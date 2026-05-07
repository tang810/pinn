# Source Terms Component Library (Agent-Oriented)

This document defines a standardized catalog of **source / forcing / excitation terms**
used in mathematical and physical modeling.

Source terms represent **external inputs** to the governing equations and are
reusable modeling components shared across multiple equation cards.

This file provides **conceptual and mathematical definitions only**.
It does NOT include numerical implementation details.

---

## 0. Design Principles (READ FIRST)

- Source terms describe **how energy, mass, momentum, or signals enter the system**.
- A source term may depend on:
  - space x,
  - time t,
  - state variables u,
  - parameters.
- Source terms MUST be explicitly specified or explicitly set to zero.
- This document defines **canonical source types**.
  Equation cards MUST NOT invent new generic source types.

---

## 1. Zero Source (Homogeneous System)

### 1.1 Canonical Form
$$
f(x,t) = 0
$$

### 1.2 Physical Meaning
- No external excitation.
- System evolution is driven purely by initial and boundary conditions.

### 1.3 Typical Use Cases
- Free vibration / free wave propagation.
- Relaxation and decay processes.
- Benchmark and validation problems.

### 1.4 Notes
- Even when the source is zero, it should be stated explicitly
  to avoid ambiguity in problem specification.

---

## 2. Constant Source

### 2.1 Canonical Form
$$
f(x,t) = f_0
$$

### 2.2 Physical Meaning
- Uniform and steady input to the system.
- Represents constant heat generation, body force, or charge density.

### 2.3 Typical Use Cases
- Steady-state Poisson and heat equations.
- Gravity-like body forces in simplified models.

### 2.4 Notes
- Often combined with Dirichlet or Neumann boundary conditions.
- May lead to unbounded growth in transient systems if not balanced.

---

## 3. Spatially Distributed Source

### 3.1 Canonical Form
$$
f(x,t) = f(x)
$$

### 3.2 Physical Meaning
- Source strength varies in space but is steady in time.

### 3.3 Typical Use Cases
- Heterogeneous material heating.
- Spatially varying charge or mass distribution.

### 3.4 Notes
- Must be compatible with domain geometry Ω.
- Discontinuities should be explicitly noted.

---

## 4. Time-Dependent Source

### 4.1 Canonical Form
$$
f(x,t) = f(t)
$$

### 4.2 Physical Meaning
- Uniform excitation that varies in time.
- Represents temporal loading, pulsed input, or global forcing.

### 4.3 Typical Use Cases
- Time-modulated boundary or volume excitation.
- Control and actuation problems.

### 4.4 Notes
- Temporal smoothness affects solution regularity.
- Abrupt changes may induce high-frequency responses.

---

## 5. Separable Space–Time Source

### 5.1 Canonical Form
$$
f(x,t) = g(x)\,h(t)
$$

### 5.2 Physical Meaning
- Source profile is separable in space and time.

### 5.3 Typical Use Cases
- Localized actuator with time-dependent signal.
- Analytical and benchmark problems.

### 5.4 Notes
- Commonly used for constructing test cases.
- Facilitates interpretation and parameter studies.

---

## 6. Localized Source (Point / Compact Support)

### 6.1 Conceptual Form
$$
f(x,t) = A(t)\,\delta(x - x_0)
$$

### 6.2 Physical Meaning
- Highly localized injection at a specific location.

### 6.3 Typical Use Cases
- Point heat source.
- Acoustic or seismic point excitation.

### 6.4 Applicability Notes
- Dirac delta is an idealization.
- In practice, often approximated by a narrow smooth function.

### 6.5 Pitfalls
- Singular sources may reduce solution regularity.
- Care is needed in inverse and learning-based formulations.

---

## 7. Harmonic (Periodic) Source

### 7.1 Canonical Form
$$
f(x,t) = A(x)\sin(\omega t + \phi)
$$

### 7.2 Physical Meaning
- Periodic excitation with angular frequency ω.

### 7.3 Typical Use Cases
- Steady-state vibration analysis.
- Frequency-domain wave and EM problems.

### 7.4 Notes
- Often used to probe system resonance.
- Phase φ should be specified when relevant.

---

## 8. Pulsed / Transient Source

### 8.1 Canonical Examples
- Gaussian pulse:
$$
f(t) = A \exp\!\left(-\frac{(t-t_0)^2}{\sigma^2}\right)
$$

- Compact pulse:
$$
f(t) =
\begin{cases}
A, & t_1 \le t \le t_2 \\
0, & \text{otherwise}
\end{cases}
$$

### 8.2 Physical Meaning
- Short-duration excitation.

### 8.3 Typical Use Cases
- Wave packet generation.
- Transient response and impulse analysis.

### 8.4 Notes
- Pulse width controls spectral bandwidth.
- Common in wave and control problems.

---

## 9. State-Dependent Source (Nonlinear Forcing)

### 9.1 Canonical Form
$$
f(x,t,u) = g(u)
$$

### 9.2 Physical Meaning
- Source depends on the system state.
- Represents reaction, feedback, or nonlinear coupling.

### 9.3 Typical Use Cases
- Reaction–diffusion systems.
- Nonlinear damping or amplification.

### 9.4 Applicability Notes
- May change equation class (e.g., linear → nonlinear).
- Stability and identifiability must be assessed carefully.

---

## 10. Source Term Selection Guidance (Agent Heuristics)

- If excitation mechanism is not specified:
  - Default to zero source and state explicitly.
- For steady elliptic problems:
  - Constant or spatial sources are common.
- For transient wave-like problems:
  - Prefer pulsed or harmonic sources.
- For inverse problems:
  - Source parameters may be jointly inferred but increase ill-posedness.

The agent MUST warn when:
- Source type is incompatible with equation assumptions.
- Source leads to non-physical growth or instability.
- Source is underspecified in time or space.

---

## 11. Chinese Notes (Optional, for Human Maintainers)

- 本文件用于统一 source / forcing / excitation 的“概念级分类”。
- equation card 中可以写 canonical 公式，但不应重新解释物理意义。
- 若出现新的通用 source 类型，应补充到本文件中统一维护。
