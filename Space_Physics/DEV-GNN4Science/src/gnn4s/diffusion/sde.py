import abc
import torch
import numpy as np

class SDE(abc.ABC):
    """ SDE abstract class.
        Functions are designed for a mini-batch of inputs.
    """
    def __init__(self, time_steps_num: int) -> None:
        """ Initialization
        Args:
            time_steps_num: number of discretization time steps.
        """
        super().__init__()
        self.time_steps_num = time_steps_num
    
    @property
    @abc.abstractmethod
    def time_end(self) -> float:
        """ End time of the SDE. 
        Returns:
            end time
        """
        pass

    @abc.abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Stochastic differential equation 
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.time_end`)
        Returns:
            drift: drift coefficient
            diffusion: diffusion coefficient
        """
        pass

    @abc.abstractmethod
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Parameters to determine the marginal distribution of the SDE, $p_t(x)$ 
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.time_end`)
        Returns:
            log probability density
        """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape: list[int]) -> torch.Tensor:
        """ Generate one sample from the prior distribution, $p_T(x)$.
        Args:
            shape: shape of sampling points
        Returns:
            sampling points
        """
        pass

    @abc.abstractmethod
    def prior_log_prob(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute log-density of the prior distribution.
            Useful for computing the log-likelihood via probability flow ODE.
        Args:
            z: latent code
            mask: mask
        Returns:
            log probability density
        """
        pass

    def discretize(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + g_i z_i.
            Useful for reverse diffusion sampling and probability flow sampling.
            Defaults to Euler-Maruyama discretization.
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.time_end`)
        Returns:
            f: drift coefficient
            g: diffusion coefficient
        """
        dt = 1 / self.time_steps_num
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        g = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, g

    def reverse(self, score_fn, probability_flow: bool=False):
        """ Create the reverse-time SDE/ODE.
        Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        Returns:
            reverse-time SDE
        """
        time_steps_num = self.time_steps_num
        time_end = self.time_end
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self) -> None:
                """ Initialization """
                self.time_steps_num = time_steps_num
                self.probability_flow = probability_flow
            
            @property
            def time_end(self) -> float:
                """ End time of the reverse SDE.
                Returns:
                    end time
                """
                return time_end
            
            def sde(self, x, adj, mask, t, is_adj=True) -> torch.Tensor:
                """ Create the drift and diffusion functions for the reverse SDE/ODE.
                Args:
                    x: a torch tensor
                    t: a torch float representing the time step (from 0 to `self.time_end`)
                Returns:
                    drift: drift coefficient
                    diffusion: diffusion coefficient
                """
                drift, diffusion = sde_fn(adj, t) if is_adj else sde_fn(x, t)
                score = score_fn(x, adj, mask, t)
                diffusion_ = diffusion[:,None,None] if len(drift.shape)==3 else diffusion[:,None,None,None]
                drift = drift - diffusion_ ** 2 * score * (0.5 if self.probability_flow else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion
            
            def discretize(self, x, adj, mask, t, is_adj=True) -> torch.Tensor:
                """ Create discretized iteration rules for the reverse diffusion sampler.
                Args:
                    x: a torch tensor
                    t: a torch float representing the time step (from 0 to `self.time_end`)
                Returns:
                    f: drift coefficient
                    g: diffusion coefficient
                """
                f, g = discretize_fn(adj, t) if is_adj else discretize_fn(x, t)
                score = score_fn(x, adj, mask, t)
                g_ = g[:,None,None] if len(f.shape)==3 else g[:,None,None,None]
                rev_f = f - g_ ** 2 * score * (0.5 if self.probability_flow else 1.)
                rev_g = torch.zeros_like(g) if self.probability_flow else g
                return rev_f, rev_g

        return RSDE()

class VPSDE(SDE):
    def __init__(self, beta_min: float=0.1, beta_max: float=20, time_steps_num: int=1000) -> None:
        """ Construct a Variance Preserving SDE.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            time_steps_num: number of discretization steps
        """
        super().__init__(time_steps_num)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.time_steps_num = time_steps_num

        self.discrete_betas = torch.linspace(self.beta_min/self.time_steps_num,
            self.beta_max/self.time_steps_num, self.time_steps_num)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    
    @property
    def time_end(self) -> float:
        """ End time of the SDE.
        Returns:
            end time
        """
        return 1

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Stochastic differential equation 
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.time_end`)
        Returns:
            drift: drift coefficient
            diffusion: diffusion coefficient
        """
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        if len(x.shape) == 4:
            drift = -0.5 * beta_t[:, None, None, None] * x
        elif len(x.shape) == 3:
            drift = -0.5 * beta_t[:, None, None] * x
        else:
            raise NotImplementedError
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Parameters to determine the marginal distribution of the SDE, $p_t(x)$ 
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.time_end`)
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        if len(x.shape) == 4:
            mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        elif len(x.shape) == 3:
            mean = torch.exp(log_mean_coeff[:, None, None]) * x
        else:
            raise ValueError("The shape of x in marginal_prob is not correct.")
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        for i in range(len(std.shape),len(mean.shape)):
            std = std.unsqueeze(-1)
        return mean, std

    def prior_sampling(self, shape: list[int]) -> torch.Tensor:
        """ Generate one sample from the prior distribution, $p_T(x)$.
        Args:
            shape: shape of sampling points
        Returns:
            sampling points
        """
        sample = torch.randn(*shape)
        if len(shape) == 4:
            sample = torch.tril(sample, -1)
            sample = sample + sample.transpose(-1, -2)

        return sample

    def prior_log_prob(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute log-density of the prior distribution.
            Useful for computing the log-likelihood via probability flow ODE.
        Args:
            z: latent code
            mask: mask
        Returns:
            log probability density
        """
        time_steps_num = torch.sum(mask, dim=tuple(range(1, len(mask.shape))))
        logps = -time_steps_num / 2. * np.log(2 * np.pi) - torch.sum((z * mask) ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ DDPM discretization.
        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to self.time_end)
        Returns:
            f: drift coefficient
            g: diffusion coefficient
        """
        timestep = (t * (self.time_steps_num - 1) / self.time_end).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        if len(x.shape) == 4:
            f = torch.sqrt(alpha)[:, None, None, None] * x - x
        elif len(x.shape) == 3:
            f = torch.sqrt(alpha)[:, None, None] * x - x
        else:
            NotImplementedError
        g = sqrt_beta
        return f, g

class VESDE(SDE):
    def __init__(self, sigma_min: float=0.01, sigma_max: float=50., time_steps_num: int=1000) -> None:
        """ Variance exploding SDE.
        Args:
            sigma_min: lower bound of standard deviation of the normal distribution of noise
            sigma_max: upper bound of standard deviation of the normal distribution of noise
            time_steps_num: number of discretization steps
        """
        super().__init__(time_steps_num)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.time_steps_num = time_steps_num

        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), time_steps_num))

    @property
    def time_end(self):
        """ End time. """
        return 1.0

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """ Stochastic differential equation.
        Args:
            x: node features or adjacency matrices
            t: time step (between 0 to self.time_end)
        Returns:
            drift: linear drift coefficient
            diffusion: diffusion coefficient
        """
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(
            2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return drift, diffusion
    
    def marginal_prob(self, x, t):
        """ probability density function of the marginal distribution.
        Args:
            x: node features or adjacency matrices
            t: time step (between 0 to self.time_end)
        Returns:
            mean: mean of the probability density function
            std: std of the probability density function
        """
        mean = x
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        for i in range(len(std.shape),len(mean.shape)):
            std = std.unsqueeze(-1)
        return mean, std

    def prior_sampling(self, shape):
        """ Generate samples from a prior distribution.
        Args:
            shape: shape of the samples
        Returns:
            samples
        """
        return torch.randn(*shape) 

    def prior_sampling_sym(self, shape):
        """ Generate symmetric samples from a prior distribution.
        Args:
            shape: shape of the samples
        Returns:
            samples
        """
        x = torch.randn(*shape).triu(1)
        x = x + x.transpose(-1,-2)
        return x 

    def prior_log_prob(self, z):
        shape = z.shape
        time_steps_num = np.prod(shape[1:])
        return (-time_steps_num/2. * np.log(2*np.pi*self.sigma_max**2) - 
                torch.sum(z**2, dim=(1,2,3)) / (2*self.sigma_max**2))

    def discretize(self, x, t):
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.time_steps_num - 1) / self.time_end).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     self.discrete_sigmas[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        g = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, g

    def transition(self, x, t, dt):
        # -------- negative timestep dt --------
        std = torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** t) - \
            torch.square(self.sigma_min * (self.sigma_max / self.sigma_min) ** (t + dt)) 
        std = torch.sqrt(std)
        mean = x
        return mean, std

class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, time_steps_num=1000):
        """Construct the sub-VP SDE that excels at likelihoods.
        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        time_steps_num: number of discretization steps
        """
        super().__init__(time_steps_num)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.time_steps_num = time_steps_num
        self.discrete_betas = torch.linspace(beta_min / time_steps_num, beta_max / time_steps_num, time_steps_num)
        self.alphas = 1. - self.discrete_betas

    @property
    def time_end(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t[:, time_steps_numone, time_steps_numone] * x
        discount = 1. - torch.exp(-2 * self.beta_min * t - (self.beta_max - self.beta_min) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff)[:, time_steps_numone, time_steps_numone] * x
        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_sampling_sym(self, shape):
        x = torch.randn(*shape).triu(1) 
        return x + x.transpose(-1,-2)

    def prior_log_prob(self, z):
        shape = z.shape
        time_steps_num = np.prod(shape[1:])
        return -time_steps_num / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
