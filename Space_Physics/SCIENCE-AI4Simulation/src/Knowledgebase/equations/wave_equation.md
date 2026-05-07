# Equation Card: Scalar Wave Equation
## 0. Metadata
- id: wave_equation_scalar_v1
- equation_name: Scalar Wave Equation
- aliases:
  - wave equation
  - acoustic wave equation
  - u_tt - c^2 Δu
  - 波动方程
  - 声学波动方程
- tags:
  - domain: [acoustics, geophysics, applied_math]
  - pde_class: [hyperbolic]
  - linearity: [linear]
  - time: [transient]
  - form: [strong]
- keywords:
  - wave propagation
  - acoustic pressure
  - seismic wave
  - wave speed
  - source term
  - absorbing boundary
  - open domain
- related_equations:
  - heat_equation
  - poisson_equation
  - elastic_wave_equation

## 1. Background
- phenomenon:
  Describes propagation of waves in a continuous medium under linear, small-amplitude approximation.
- typical_applications:
  - acoustic wave propagation
  - seismic exploration (acoustic approximation)
  - ultrasound and nondestructive testing
- assumptions:
  - linear small-amplitude waves
  - continuous medium
  - no intrinsic dissipation unless explicitly modeled
- invalid_when:
  - strong nonlinear effects (shock waves)
  - elastic shear waves or anisotropic media dominate

### 中文辅助说明
- 该方程常用于声学、地震勘探中的简化模型
- 若存在强非线性或弹性效应，应使用更复杂的方程体系

## 2. Variables & Parameters
- independent_variables:
  - space: x ∈ Ω ⊂ R^d
  - time: t ∈ [0, T]
- dependent_fields:
  - u(x,t): scalar wave field (e.g. acoustic pressure or potential), unit: application-dependent
- parameters:
  - c(x): wave speed field, unit: m/s, typical_range: 10^2–10^4, known_or_unknown: known/unknown
  - s(x,t): source term or external forcing, unit: consistent with u, known_or_unknown: known/unknown
- observables:
  - point-wise sensor measurements u(x_s, t)
  - wave arrival times, wavefronts, spectra

## 3. Governing Equation

### 3.1 Strong form (primary)
$$
\frac{\partial^2 u}{\partial t^2}
- \nabla \cdot \left( c^2(x)\nabla u \right)
= s(x,t),
\quad x \in \Omega,\ t \in (0,T].
$$

### 3.2 Alternative forms
- Constant wave speed:
$$
u_{tt} - c^2 \Delta u = s.
$$

- First-order system (introducing v = u_t):
$$
\begin{cases}
u_t = v, \\
v_t = \nabla \cdot \left( c^2(x)\nabla u \right) + s.
\end{cases}
$$

### 3.3 Non-dimensional form (recommended)
Let x = L x', t = (L / c_0) t', u = U_0 u', c = c_0 c'. Then:
$$
u_{t't'} - \nabla' \cdot \left( (c')^2 \nabla' u' \right) = s'.
$$

- dimensionless_numbers:
  - λ / L: wavelength-to-domain ratio (controls resolution requirements)

## 4. Initial and Boundary Conditions

### 4.1 Initial conditions
- Initial displacement:
$$
u(x,0) = u_0(x).
$$

- Initial velocity:
$$
\frac{\partial u}{\partial t}(x,0) = v_0(x).
$$

### 4.2 Boundary condition catalog
- Dirichlet:
$$
u(x,t) = g(x,t), \quad x \in \partial\Omega.
$$

- Neumann:
$$
\nabla u(x,t)\cdot n = q(x,t), \quad x \in \partial\Omega.
$$

- Robin / Impedance:
$$
\alpha u + \beta\, \nabla u\cdot n = r(x,t), \quad x \in \partial\Omega.
$$

- Radiation / Absorbing boundary:
  - Enforces outgoing-wave behavior
  - Used to approximate open or unbounded domains
  - Specific formulation depends on chosen ABC or PML scheme

### 中文辅助说明
- 开域问题（声波、地震波）通常必须使用吸收边界或 PML
- 简单 Dirichlet 边界会导致非物理反射

## 5. Domain & Typical Setups
- geometry_templates:
  - 1D interval
  - 2D rectangle or circle
  - 3D box or sphere
  - heterogeneous media with spatially varying c(x)
- typical_benchmarks:
  - 1D Gaussian pulse propagation
  - 2D point source in layered velocity model

## 6. Inverse Problems / Identification
- invertible_targets:
  - wave speed field c(x)
  - source location and source time function
  - boundary impedance parameters
- required_data:
  - multi-source excitation
  - time-series measurements at multiple sensor locations
- identifiability_notes:
  - limited sources or sparse sensors lead to poor spatial resolution
  - strong correlation between c(x) and source parameters
- recommended_priors_or_regularization:
  - smoothness or TV regularization for c(x)
  - physical bounds: c_min ≤ c(x) ≤ c_max

## 7. Constraint Decomposition (for agent compilation)
- PDE_residual:
  - r_pde(x,t) = u_tt - div(c(x)^2 ∇u) - s(x,t)
- IC_residuals:
  - r_ic_u(x) = u(x,0) - u_0(x)
  - r_ic_v(x) = u_t(x,0) - v_0(x)
- BC_residuals:
  - r_bc_D = u - g
  - r_bc_N = ∇u·n - q
  - r_bc_R = αu + β∇u·n - r
  - r_bc_A = absorbing boundary residual (ABC/PML-dependent)
- data_residuals:
  - r_data = u(x_s,t) - u_obs(t)
- weighting_strategy:
  - emphasize IC/BC in early training
  - gradually increase PDE weight
  - add regularization for inverse problems

## 8. Minimal Problem Spec Checklist
- [ ] spatial dimension d, domain Ω, time window [0,T]
- [ ] physical meaning of u
- [ ] wave speed c(x): known or unknown, bounds defined
- [ ] source term or excitation mechanism
- [ ] initial conditions u_0, v_0
- [ ] boundary conditions on entire ∂Ω
- [ ] observation data specification (if inverse)
- [ ] required outputs and metrics

## 9. Minimal Example (toy specification)
- setup:
  - d = 1, Ω = [0,1], T = 1
  - c = 1
  - u_0(x) = exp(-100(x-0.2)^2), v_0(x) = 0
  - Dirichlet BC: u(0,t)=u(1,t)=0
  - s = 0
- expected_behavior:
  - wave splits and reflects at boundaries
  - total energy conserved (no damping)

## 10. References
- Standard PDE textbooks on hyperbolic equations
- Acoustics and seismology literature on absorbing boundaries and wave inversion
