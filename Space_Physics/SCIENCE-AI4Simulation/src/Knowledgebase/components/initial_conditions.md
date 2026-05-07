# Initial Conditions Component Library (Agent-Oriented)

This document defines a standardized catalog of **initial condition (IC) types**
used in time-dependent mathematical and physical models.

Initial conditions specify the system state at the initial time and are
required for all transient problems.

This file provides **conceptual and mathematical definitions only**.
It does NOT include numerical or implementation-specific details.

---

## 0. Design Principles (READ FIRST)

- Initial conditions define the system state at the initial time t = 0.
- All transient problems MUST specify initial conditions explicitly.
- Initial conditions may apply to:
  - primary fields,
  - auxiliary fields,
  - time derivatives of fields.
- This document defines **canonical initial condition types**.
  Equation cards MUST NOT invent new generic IC categories.

---

## 1. Zero Initial Condition (Rest State)

### 1.1 Canonical Form
$$
u(x,0) = 0
$$

### 1.2 Physical Meaning
- System starts from a rest or equilibrium state.
- No initial excitation or stored energy.

### 1.3 Typical Use Cases
- Free response driven by boundary or source excitation.
- Benchmark and validation problems.

### 1.4 Notes
- Must be stated explicitly to avoid ambiguity.
- Often paired with nonzero boundary or source terms.

---

## 2. Prescribed Initial Field

### 2.1 Canonical Form
$$
u(x,0) = u_0(x)
$$

### 2.2 Physical Meaning
- Initial spatial distribution of the field is prescribed.
- Represents pre-existing temperature, displacement, concentration, etc.

### 2.3 Typical Use Cases
- Heat diffusion from a known temperature distribution.
- Initial deformation or stress state in solids.

### 2.4 Notes
- Regularity of \(u_0(x)\) affects solution smoothness.
- Discontinuities should be explicitly noted.

---

## 3. Initial Time-Derivative (Velocity-Type IC)

### 3.1 Canonical Form
$$
\frac{\partial u}{\partial t}(x,0) = v_0(x)
$$

### 3.2 Physical Meaning
- Specifies initial rate of change of the field.
- Represents velocity, momentum, or flux at t = 0.

### 3.3 Typical Use Cases
- Wave and vibration problems.
- Second-order dynamical systems.

### 3.4 Notes
- Required for second-order-in-time equations.
- Often omitted in first-order-in-time systems.

---

## 4. Multiple-Field Initial Conditions

### 4.1 Canonical Form
For a system with multiple fields:
$$
u_i(x,0) = u_{i,0}(x), \quad i = 1,\dots,N
$$

### 4.2 Physical Meaning
- Each field in a coupled system has its own initial state.

### 4.3 Typical Use Cases
- Multi-physics or multi-variable systems.
- First-order system reformulations of higher-order PDEs.

### 4.4 Notes
- Consistency across fields is critical.
- Redundant or conflicting ICs must be avoided.

---

## 5. Random or Perturbed Initial Condition

### 5.1 Conceptual Form
$$
u(x,0) = u_{\text{base}}(x) + \epsilon(x)
$$

### 5.2 Physical Meaning
- Initial state includes small perturbations.
- Used to probe stability or transition behavior.

### 5.3 Typical Use Cases
- Stability analysis.
- Turbulence onset studies.
- Sensitivity and robustness analysis.

### 5.4 Notes
- Statistical properties of perturbation should be specified.
- Random ICs complicate reproducibility.

---

## 6. Parameterized Initial Condition

### 6.1 Canonical Form
$$
u(x,0) = u(x,0;\theta)
$$

### 6.2 Physical Meaning
- Initial condition depends on unknown or tunable parameters.

### 6.3 Typical Use Cases
- Inverse problems.
- System identification.
- Control and optimization.

### 6.4 Notes
- Parameter bounds or priors should be stated.
- Identifiability must be assessed carefully.

---

## 7. Consistency Conditions

### 7.1 Description
Initial conditions must be consistent with:
- boundary conditions,
- physical constraints,
- governing equations.

### 7.2 Typical Examples
- Compatibility of displacement and velocity ICs in mechanics.
- Mass or charge conservation at t = 0.

### 7.3 Pitfalls
- Inconsistent IC/BC combinations leading to non-physical transients.
- Overspecified initial data.

---

## 8. Initial Condition Selection Guidance (Agent Heuristics)

- For transient problems:
  - At least one IC per time-dependent field is mandatory.
- For second-order-in-time systems:
  - Both field value and time-derivative ICs are required.
- For inverse problems:
  - ICs are preferably known; unknown ICs increase ill-posedness.
- For benchmark problems:
  - Prefer simple, smooth initial conditions.

The agent MUST warn when:
- Required initial conditions are missing.
- ICs conflict with BCs or model assumptions.
- ICs introduce unnecessary complexity without justification.

---

## 9. Chinese Notes (Optional, for Human Maintainers)

- 本文件用于统一初始条件（Initial Conditions）的概念级分类。
- equation card 中可写 canonical 形式，但不应重复解释物理含义。
- 若出现新的通用 IC 类型，应补充到本文件中统一维护。
