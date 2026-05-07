# Equation Card: 6DOF Aircraft Dynamics with Aerodynamic Coefficient Identification

---

## 0. Metadata (REQUIRED)
- id: aircraft_6dof_aero_identification_v1
- equation_name: Six-Degree-of-Freedom Aircraft Rigid-Body Dynamics with Aerodynamic Coefficient Identification
- aliases:
  - 6DOF aircraft equations
  - rigid-body flight dynamics
  - aerodynamic coefficient identification
  - aircraft dynamics inverse problem
  - 六自由度飞机动力学
  - 气动系数辨识
- tags:
  - domain: [fluid, math]
  - pde_class: [mixed]
  - linearity: [nonlinear]
  - time: [transient]
  - form: [strong]
- keywords:
  - flight dynamics
  - aerodynamic coefficients
  - body-axis acceleration
  - inverse identification
  - PINN
  - F-16-like model
  - 刚体动力学
  - 气动物理残差
- related_equations:
  - rigid_body_euler_rotation_v1
  - body_axis_force_balance_v1
  - aerodynamic_force_moment_model_v1

---

## 1. Background (REQUIRED)
- phenomenon: This equation set models the transient motion of a rigid aircraft in body coordinates and Earth coordinates under aerodynamic forces, aerodynamic moments, gravity, and thrust. In this case, the main task is not only forward simulation, but also inverse identification of the aerodynamic coefficient functions from noisy flight measurements.
- typical_applications:
  - aerodynamic database reconstruction from flight-test data
  - physics-informed surrogate modeling for aircraft force and moment coefficients
  - flight dynamics system identification
- assumptions:
  - rigid-body aircraft model with lumped mass and inertia
  - body-axis translational and rotational equations are sufficient
  - low-order aerodynamic coefficient map is a smooth function of angle, rate, and control inputs
  - constant air density and constant thrust in the toy case
  - flat-Earth kinematics with Euler angles
- invalid_when:
  - large aeroelastic deformation, fuel slosh, or structural flexibility is significant
  - transonic/shock-dominated flow requires high-fidelity compressible aerodynamics not captured by a low-order coefficient model
  - near-singular Euler-angle attitude representation causes kinematic instability
  - rapidly varying atmosphere, variable mass, or thrust vectoring cannot be neglected

---

## 2. Variables & Parameters (REQUIRED)
- independent_variables:
  - time: $$t \in [0, T]$$
- dependent_fields:
  - $$u(t), v(t), w(t)$$: body-axis translational velocity components, unit: m/s
  - $$p(t), q(t), r(t)$$: body-axis angular rates, unit: rad/s
  - $$\phi(t), \theta(t), \psi(t)$$: roll, pitch, yaw Euler angles, unit: rad
  - $$x_E(t), y_E(t), z_E(t)$$: Earth-frame position, unit: m
  - $$C_X, C_Y, C_Z, C_l, C_m, C_n$$: aerodynamic force and moment coefficients, unit: dimensionless
- parameters:
  - $$m$$: aircraft mass, unit: kg, typical_range: aircraft-specific, known_or_unknown: known
  - $$I_x, I_y, I_z$$: principal inertias, unit: kg-m^2, typical_range: aircraft-specific, known_or_unknown: known
  - $$I_{xz}$$: product of inertia, unit: kg-m^2, typical_range: aircraft-specific, known_or_unknown: known
  - $$S$$: reference wing area, unit: m^2, typical_range: aircraft-specific, known_or_unknown: known
  - $$b$$: wing span, unit: m, typical_range: aircraft-specific, known_or_unknown: known
  - $$\bar c$$: mean aerodynamic chord, unit: m, typical_range: aircraft-specific, known_or_unknown: known
  - $$\rho$$: air density, unit: kg/m^3, typical_range: atmosphere-dependent, known_or_unknown: known
  - $$g$$: gravitational acceleration, unit: m/s^2, typical_range: constant near Earth surface, known_or_unknown: known
  - $$T$$: thrust magnitude, unit: N, typical_range: mission-dependent, known_or_unknown: known
  - $$\delta_e, \delta_a, \delta_r$$: elevator, aileron, rudder deflections, unit: rad, typical_range: actuator-limited, known_or_unknown: known
  - $$\mathcal{C}(\xi) = [C_X, C_Y, C_Z, C_l, C_m, C_n]$$: aerodynamic coefficient map, unit: dimensionless, typical_range: smooth bounded function, known_or_unknown: unknown
- observables:
  - measured state trajectory $$[u,v,w,p,q,r,\phi,\theta,\psi,x_E,y_E,z_E]$$
  - measured or numerically estimated body-axis accelerations $$[\dot u, \dot v, \dot w, \dot p, \dot q, \dot r]$$
  - control histories $$[\delta_e, \delta_a, \delta_r]$$

---

## 3. Governing Equation (REQUIRED)

### 3.1 Strong form (PRIMARY, REQUIRED)
Define airspeed, aerodynamic angles, and non-dimensional rates as

$$
V = \sqrt{u^2 + v^2 + w^2}, \qquad
\alpha = \arctan\!\left(\frac{w}{u}\right), \qquad
\beta = \arcsin\!\left(\frac{v}{V}\right),
$$

$$
\hat p = \frac{p b}{2V}, \qquad
\hat q = \frac{q \bar c}{2V}, \qquad
\hat r = \frac{r b}{2V}.
$$

Let the aerodynamic coefficient map depend on the explanatory variables

$$
\xi = [\alpha, \beta, \hat p, \hat q, \hat r, \delta_e, \delta_a, \delta_r, V_{norm}],
$$

where $$V_{norm}$$ is an optional normalized speed feature.

Dynamic pressure and body-axis forces/moments are

$$
\bar q = \frac{1}{2}\rho V^2,
$$

$$
X = \bar q S C_X + T, \qquad
Y = \bar q S C_Y, \qquad
Z = \bar q S C_Z,
$$

$$
L = \bar q S b C_l, \qquad
M = \bar q S \bar c C_m, \qquad
N = \bar q S b C_n.
$$

The translational rigid-body dynamics in body coordinates are

$$
\dot u = rv - qw + \frac{X}{m} - g\sin\theta,
$$

$$
\dot v = pw - ru + \frac{Y}{m} + g\cos\theta\sin\phi,
$$

$$
\dot w = qu - pv + \frac{Z}{m} + g\cos\theta\cos\phi.
$$

The rotational rigid-body dynamics with nonzero product of inertia $$I_{xz}$$ are

$$
\Delta_I = I_x I_z - I_{xz}^2,
$$

$$
\dot p = \frac{(I_z L + I_{xz} N) + I_{xz}(I_x - I_y + I_z)pq - (I_z^2 + I_{xz} I_x)qr}{\Delta_I},
$$

$$
\dot q = \frac{M - (I_x - I_z)pr - I_{xz}(p^2 - r^2)}{I_y},
$$

$$
\dot r = \frac{(I_x N + I_{xz} L) + I_{xz}(I_x + I_y - I_z)qr - (I_x^2 + I_{xz} I_z)pq}{\Delta_I}.
$$

The Euler-angle kinematics are

$$
\dot \phi = p + \tan\theta (q\sin\phi + r\cos\phi),
$$

$$
\dot \theta = q\cos\phi - r\sin\phi,
$$

$$
\dot \psi = \frac{q\sin\phi + r\cos\phi}{\cos\theta}.
$$

The Earth-frame position kinematics in the toy flat-Earth setup are

$$
\dot x_E = (\cos\theta\cos\psi)u + (\cos\theta\sin\psi)v - (\sin\theta)w,
$$

$$
\dot y_E = (\sin\phi\sin\theta\cos\psi - \cos\phi\sin\psi)u + (\sin\phi\sin\theta\sin\psi + \cos\phi\cos\psi)v + (\sin\phi\cos\theta)w,
$$

$$
\dot z_E = (\cos\phi\sin\theta\cos\psi + \sin\phi\sin\psi)u + (\cos\phi\sin\theta\sin\psi - \sin\phi\cos\psi)v + (\cos\phi\cos\theta)w.
$$

### 3.2 Alternative forms (OPTIONAL, recommended)
- first-order state-space form:
$$
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}_c, \mathcal{C}(\xi); \Theta),
$$
where
$$
\mathbf{x} = [u,v,w,p,q,r,\phi,\theta,\psi,x_E,y_E,z_E]^\top,
$$
$$
\mathbf{u}_c = [\delta_e,\delta_a,\delta_r]^\top,
$$
and $$\Theta = \{m,I_x,I_y,I_z,I_{xz},S,b,\bar c,\rho,g,T\}$$.

- coefficient submodel used in the toy case:
$$
C_X = C_{X0} + C_{X\alpha}\alpha + C_{Xq}\hat q + C_{X\delta_e}\delta_e,
$$
$$
C_Y = C_{Y\beta}\beta + C_{Y\delta_a}\delta_a + C_{Y\delta_r}\delta_r + C_{Yp}\hat p + C_{Yr}\hat r,
$$
$$
C_Z = C_{Z0} + C_{Z\alpha}\alpha + C_{Zq}\hat q + C_{Z\delta_e}\delta_e,
$$
$$
C_l = C_{l\beta}\beta + C_{l\delta_a}\delta_a + C_{l\delta_r}\delta_r + C_{lp}\hat p + C_{lr}\hat r,
$$
$$
C_m = C_{m0} + C_{m\alpha}\alpha + C_{mq}\hat q + C_{m\delta_e}\delta_e,
$$
$$
C_n = C_{n\beta}\beta + C_{n\delta_a}\delta_a + C_{n\delta_r}\delta_r + C_{np}\hat p + C_{nr}\hat r.
$$

### 3.3 Non-dimensional form (OPTIONAL but recommended)
A common non-dimensionalization uses

$$
\tau = \frac{V_0}{\bar c} t, \qquad \tilde u = \frac{u}{V_0}, \qquad \tilde v = \frac{v}{V_0}, \qquad \tilde w = \frac{w}{V_0},
$$

with aerodynamic rates already written as non-dimensional quantities $$\hat p, \hat q, \hat r$$.
The resulting equations preserve the same structure while reducing scale disparity between translational velocities, angular rates, and coefficient targets.

- dimensionless_numbers:
  - reduced_roll_rate: $$\hat p = pb/(2V)$$, interpretation: roll-rate normalization by convective time scale
  - reduced_pitch_rate: $$\hat q = q\bar c/(2V)$$, interpretation: pitch-rate normalization by chord-based convective time scale
  - reduced_yaw_rate: $$\hat r = rb/(2V)$$, interpretation: yaw-rate normalization by convective time scale

---

## 4. Initial and Boundary Conditions (REQUIRED)

### 4.1 Initial conditions (REQUIRED if transient)
- IC-1:
$$
[u(0), v(0), w(0), p(0), q(0), r(0)] = [u_0, v_0, w_0, p_0, q_0, r_0].
$$
- IC-2:
$$
[\phi(0), \theta(0), \psi(0), x_E(0), y_E(0), z_E(0)] = [\phi_0, \theta_0, \psi_0, x_{E0}, y_{E0}, z_{E0}].
$$

### 4.2 Boundary condition catalog (REQUIRED)
This is an initial-value ODE system rather than a spatial PDE, so classical spatial boundary conditions on $$\partial \Omega$$ are not required.

- Dirichlet:
$$
\mathbf{x}(0) = \mathbf{x}_0.
$$
- Neumann:
$$
\dot{\mathbf{x}}(0) = \mathbf{f}(\mathbf{x}_0, \mathbf{u}_c(0), \mathcal{C}(\xi_0); \Theta)
$$
used only when derivative consistency at the initial time is enforced.
- Robin/Impedance (if relevant):
$$
\text{N/A for the standard flight-dynamics initial-value formulation.}
$$
- Periodic / Symmetry / Radiation(ABC/PML) (if relevant):
Not applicable for the standard trajectory-identification formulation.

---

## 5. Domain & Typical Setups (REQUIRED)
- geometry_templates:
  - time interval $$[0,T]$$ for a single maneuver
  - multiple flight segments stitched in time for multi-condition identification
- typical_benchmarks:
  - synthetic F-16-like maneuver with rich multi-sine control excitation
  - coefficient recovery from noisy body-axis state and acceleration measurements

---

## 6. Inverse Problems / Identification (REQUIRED if meaningful; otherwise write "N/A")
- invertible_targets:
  - aerodynamic coefficient functions $$C_X, C_Y, C_Z, C_l, C_m, C_n$$
  - low-order polynomial aerodynamic derivatives such as $$C_{m\alpha}, C_{mq}, C_{Y\beta}$$
  - optionally thrust bias or sensor bias if explicitly modeled
- required_data:
  - time histories of $$u,v,w,p,q,r,\phi,\theta$$ over a sufficiently exciting maneuver
  - control histories $$\delta_e, \delta_a, \delta_r$$ over the same time interval
  - measured or estimated accelerations $$\dot u, \dot v, \dot w, \dot p, \dot q, \dot r$$ at matching time samples
- identifiability_notes:
  - poor excitation causes parameter collinearity, especially for rate derivatives and control derivatives
  - if only state measurements are available, acceleration estimation can amplify noise and degrade identifiability
  - multiple coefficient combinations may yield similar accelerations unless the maneuver spans enough angle and rate variation
  - thrust uncertainty can leak into identified axial-force coefficient estimates
- recommended_priors_or_regularization:
  - smoothness prior on coefficient functions
  - bounded coefficient magnitude
  - weak L2 regularization on predicted coefficients
  - symmetry-informed priors for lateral-directional terms when applicable

---

## 7. Constraint Decomposition (for agent compilation) (REQUIRED)
Use the same notation as above.

- PDE_residual:
  - $$r_u(t) = \dot u_{pred} - \dot u_{meas}$$
  - $$r_v(t) = \dot v_{pred} - \dot v_{meas}$$
  - $$r_w(t) = \dot w_{pred} - \dot w_{meas}$$
  - $$r_p(t) = \dot p_{pred} - \dot p_{meas}$$
  - $$r_q(t) = \dot q_{pred} - \dot q_{meas}$$
  - $$r_r(t) = \dot r_{pred} - \dot r_{meas}$$
- IC_residuals:
  - $$r_{ic,1} = \|[u(0),v(0),w(0),p(0),q(0),r(0)] - [u_0,v_0,w_0,p_0,q_0,r_0]\|^2$$
  - $$r_{ic,2} = \|[\phi(0),\theta(0),\psi(0),x_E(0),y_E(0),z_E(0)] - [\phi_0,\theta_0,\psi_0,x_{E0},y_{E0},z_{E0}]\|^2$$
- BC_residuals:
  - $$r_{bc,D} = \text{N/A}$$
  - $$r_{bc,N} = \text{N/A}$$
  - $$r_{bc,R} = \text{N/A}$$
  - $$r_{bc,open} = \text{N/A}$$
- data_residuals (optional):
  - $$r_{data,x}(t) = \|\mathbf{x}_{pred}(t) - \mathbf{x}_{meas}(t)\|^2$$
  - $$r_{data,c}(t) = \|\mathcal{C}_{pred}(t) - \mathcal{C}_{ref}(t)\|^2$$ if coefficient labels are available in synthetic data
- weighting_strategy (optional):
  - use separate translation and rotation weights to balance force-driven and moment-driven residuals
  - normalize inputs such as speed to reduce scale imbalance
  - add a small coefficient regularization term to suppress unphysical drift

A typical loss for this case is

$$
\mathcal{L} = \lambda_{trans} \mathbb{E}[r_u^2 + r_v^2 + r_w^2] + \lambda_{rot} \mathbb{E}[r_p^2 + r_q^2 + r_r^2] + \lambda_{reg} \mathbb{E}[C_X^2 + C_Y^2 + C_Z^2 + C_l^2 + C_m^2 + C_n^2].
$$

---

## 8. Minimal Problem Spec Checklist (REQUIRED)
- [ ] time window $$[0,T]$$ and sampling schedule
- [ ] physical meaning of all state variables and coefficient outputs
- [ ] mass, inertia, reference geometry, gravity, density, and thrust assumptions
- [ ] control input histories or excitation mechanism
- [ ] initial conditions for full state vector
- [ ] acceleration measurements or a trusted method to infer them
- [ ] specification of unknown coefficient targets or parameterization family
- [ ] required outputs and evaluation metrics

---

## 9. Minimal Example (toy specification, NO CODE) (REQUIRED)
- setup:
  - one transient maneuver on $$t \in [0, 20]$$
  - a rigid aircraft starts near steady level flight with small positive angle of attack
  - control inputs $$\delta_e(t), \delta_a(t), \delta_r(t)$$ are multi-sine excitations to stimulate longitudinal and lateral-directional dynamics
  - known constants $$m, I_x, I_y, I_z, I_{xz}, S, b, \bar c, \rho, g, T$$
  - unknown target is the aerodynamic coefficient map $$\mathcal{C}(\xi)$$
  - noisy measurements are available for body-axis states and body-axis accelerations
- expected_behavior:
  - the trajectory remains dynamically smooth and responds to elevator, aileron, and rudder excitation
  - the identified coefficient model should reproduce measured accelerations through the rigid-body equations
  - coefficient estimates should track the synthetic reference trends with low residual error

---

## 10. References (REQUIRED)
- Stevens, B. L., Lewis, F. L., and Johnson, E. N., *Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems*.
- Etkin, B., and Reid, L. D., *Dynamics of Flight: Stability and Control*.
- Schmidt, B. E., and others on aerodynamic system identification and flight-test parameter estimation.
- General literature on physics-informed neural networks for inverse dynamics and parameter identification.

---

## Chinese Notes (Optional)
- 这个案例更适合被理解为“**基于 6DOF 常微分方程组的逆问题**”，而不是经典空间 PDE。
- 模板中的 “Boundary Conditions” 在这里对应的是 **初值约束与时间序列观测约束**。
- 如果后续你希望把它写得更贴近你代码中的“真值多项式气动模型”，也可以再补一版只讲 $$C_X, C_Y, C_Z, C_l, C_m, C_n$$ 六个系数函数的说明卡。
