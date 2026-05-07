import torch

# =========================
# Gradient
# =========================
class Gradient:
    @staticmethod
    def compute_gradient(f, x):
        assert x.requires_grad, "x must require grad"

        if f.dim() == 1:
            f = f.unsqueeze(1)

        grads = []
        for i in range(f.shape[1]):
            gi = torch.autograd.grad(
                f[:, i].sum(),
                x,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]

            if gi is None:
                gi = torch.zeros_like(x)

            grads.append(gi.unsqueeze(1))

        return torch.cat(grads, dim=1)  # [N, k, d]


# =========================
# Curl & Helmholtz
# =========================
class Curl:
    @staticmethod
    def curl(E, x):
        """
        E: [N, 3]
        x: [N, 4]  (x,y,z,ω)
        """

        grad_E_full = Gradient.compute_gradient(E, x)  # [N,3,4]
        grad_E = grad_E_full[:, :, :3]                 # 只用空间梯度

        curl_E = torch.zeros_like(E)
        curl_E[:, 0] = grad_E[:, 2, 1] - grad_E[:, 1, 2]
        curl_E[:, 1] = grad_E[:, 0, 2] - grad_E[:, 2, 0]
        curl_E[:, 2] = grad_E[:, 1, 0] - grad_E[:, 0, 1]

        return curl_E

    @staticmethod
    def helmholtz_operator(E, x, epsilon, mu, omega):
        curl_E = Curl.curl(E, x)

        mu = mu.expand_as(E)
        epsilon = epsilon.expand_as(E)

        curl_mu_inv_curl_E = Curl.curl(curl_E / mu, x)

        return curl_mu_inv_curl_E - epsilon * omega**2 * E
