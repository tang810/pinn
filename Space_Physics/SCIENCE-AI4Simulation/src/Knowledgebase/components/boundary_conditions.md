# Boundary Conditions Component Library (Agent-Oriented)

This document defines a standardized catalog of boundary condition (BC) types
used in mathematical and physical modeling.

Boundary conditions are **reusable components** that must be combined with
governing equations to form well-posed problems.

This file is intended for **agent reasoning and problem assembly**, not for
numerical implementation.

---

## 0. Design Principles (READ FIRST)

- Boundary conditions MUST be explicitly specified on the entire boundary ∂Ω.
- BCs are classified by **mathematical form**, not by numerical method.
- Each BC entry provides:
  - canonical mathematical expression,
  - physical interpretation,
  - typical use cases,
  - applicability notes and pitfalls.
- BC definitions here are **equation-agnostic** and may be referenced by multiple equation cards.

---

## 1. Dirichlet Boundary Condition

### 1.1 Canonical Form
$$
u(x,t) = g(x,t), \quad x \in \partial\Omega.
$$

### 1.2 Physical Meaning
- Prescribes the value of the field on the boundary.
- Represents fixed state, enforced displacement, fixed temperature, fixed potential, etc.

### 1.3 Typical Use Cases
- Fixed temperature in heat conduction.
- Prescribed displacement in solid mechanics.
- Fixed potential in electrostatics.
- Sound-soft boundary in acoustics.

### 1.4 Applicability Notes
- Leads to a strongly constrained boundary.
- May introduce artificial reflections in wave problems if used on truncated domains.

### 1.5 Common Pitfalls
- Using Dirichlet BC on open-domain wave propagation problems.
- Over-constraining problems where natural flux conditions are more appropriate.

---

## 2. Neumann Boundary Condition

### 2.1 Canonical Form
$$
\nabla u(x,t) \cdot n = q(x,t), \quad x \in \partial\Omega.
$$

### 2.2 Physical Meaning
- Prescribes normal flux through the boundary.
- Represents heat flux, stress traction, electric flux, etc.

### 2.3 Typical Use Cases
- Insulated boundary in heat transfer (q = 0).
- Traction boundary in elasticity.
- Specified flux or current in electromagnetics.

### 2.4 Applicability Notes
- Often considered a "natural" boundary condition in variational formulations.
- For elliptic problems, pure Neumann BC requires compatibility conditions.

### 2.5 Common Pitfalls
- Forgetting global solvability constraints (e.g., solution uniqueness up to a constant).
- Mixing inconsistent flux definitions across coupled equations.

---

## 3. Robin (Mixed) Boundary Condition

### 3.1 Canonical Form
$$
\alpha u + \beta \nabla u \cdot n = r(x,t), \quad x \in \partial\Omega.
$$

### 3.2 Physical Meaning
- Linear combination of value and flux.
- Represents convective heat transfer, impedance boundaries, or partial constraints.

### 3.3 Typical Use Cases
- Newton cooling law in heat transfer.
- Impedance boundary in acoustics.
- Contact resistance in electro-thermal problems.

### 3.4 Applicability Notes
- Interpolates between Dirichlet (β=0) and Neumann (α=0).
- Coefficients (α, β) must have consistent physical units.

### 3.5 Common Pitfalls
- Using Robin BC without physical justification.
- Inconsistent scaling of α and β.

---

## 4. Periodic Boundary Condition

### 4.1 Canonical Form
$$
u(x_1,t) = u(x_2,t), \quad \nabla u(x_1,t) \cdot n_1 = -\nabla u(x_2,t) \cdot n_2,
$$
where \(x_1\) and \(x_2\) are corresponding points on periodic boundaries.

### 4.2 Physical Meaning
- Enforces spatial periodicity.
- Represents infinite or repeating structures.

### 4.3 Typical Use Cases
- Unit cell modeling in materials science.
- Fully developed flow in channel simulations.
- Periodic media wave propagation.

### 4.4 Applicability Notes
- Requires geometric compatibility between paired boundaries.
- Often combined with forcing terms or mean constraints.

### 4.5 Common Pitfalls
- Applying periodic BC to non-periodic geometries.
- Forgetting to constrain mean value when necessary.

---

## 5. Symmetry Boundary Condition

### 5.1 Canonical Form (Scalar Field)
$$
\nabla u(x,t) \cdot n = 0, \quad x \in \partial\Omega_{\text{sym}}.
$$

### 5.2 Physical Meaning
- No flux across symmetry plane.
- Represents mirror symmetry of the solution.

### 5.3 Typical Use Cases
- Reducing computational domain.
- Symmetric loading or geometry.

### 5.4 Applicability Notes
- Valid only when geometry, material, and loading are symmetric.
- Vector fields require component-wise symmetry definitions.

### 5.5 Common Pitfalls
- Using symmetry BC when forcing or geometry breaks symmetry.

---

## 6. Absorbing / Radiation Boundary Condition (Open Domain)

### 6.1 Conceptual Form
- Enforces outgoing-wave behavior at artificial boundaries.
- Prevents non-physical reflections in truncated domains.

### 6.2 Typical Representations
- First-order absorbing boundary (conceptual):
$$
\frac{\partial u}{\partial t} + c \nabla u \cdot n = 0
$$
- Higher-order ABCs or Perfectly Matched Layers (PML) may be used.

### 6.3 Physical Meaning
- Models an unbounded or semi-infinite domain.

### 6.4 Typical Use Cases
- Acoustic wave propagation.
- Seismic wave modeling.
- Electromagnetic scattering.

### 6.5 Applicability Notes
- ABC/PML formulation depends on the governing equation.
- Often implemented via auxiliary equations or domain extension.

### 6.6 Common Pitfalls
- Using Dirichlet or Neumann BC instead of absorbing BC for wave problems.
- Assuming ABC is exact for all incidence angles or frequencies.

---

## 7. Interface Boundary Condition (Coupled Domains)

### 7.1 Canonical Form (Example)
$$
u_1 = u_2, \quad
k_1 \nabla u_1 \cdot n = k_2 \nabla u_2 \cdot n.
$$

### 7.2 Physical Meaning
- Enforces continuity of field and flux across material interfaces.

### 7.3 Typical Use Cases
- Multi-material heat conduction.
- Fluid–solid interaction (simplified scalar coupling).
- Layered media wave propagation.

### 7.4 Applicability Notes
- Normal direction must be consistently defined.
- May require additional jump conditions in nonlinear or vector systems.

### 7.5 Common Pitfalls
- Mismatched normals or sign conventions.
- Ignoring material parameter discontinuities.

---

## 8. Boundary Condition Selection Guidance (Agent Heuristics)

- Elliptic problems:
  - Use Dirichlet/Neumann/Robin combinations.
- Parabolic problems:
  - Require BC + initial condition.
- Hyperbolic (wave-like) problems:
  - Prefer absorbing/radiation BC for truncated domains.
- Inverse problems:
  - BCs must be known or jointly inferred with strong regularization.

The agent MUST warn when:
- BCs do not cover the entire boundary.
- Selected BCs conflict with modeling assumptions.
- BC choice may lead to non-physical artifacts.

---

## 9. Chinese Notes (Optional, for Human Maintainers)

- 本文件用于统一边界条件的“概念级定义”，不涉及数值实现细节。
- equation card 中应引用这里的 BC 类型，而不是重新发明边界条件。
- 若某类新边界条件在多个方程中重复出现，应补充到本文件中。
