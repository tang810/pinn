# Equation Card Template (Agent-Oriented)

> Authoring Goal:
> Create an equation-level knowledge unit that an agent can reliably use to:
> (1) select a governing equation, (2) complete a well-posed problem definition,
> (3) decompose constraints for solver/PINN compilation, and (4) identify inverse targets.

> Language Rule (STRICT):
> - All structured content MUST be in English.
> - Chinese is allowed only as inline comments or an auxiliary section titled "Chinese Notes (Optional)".
> - Aliases/keywords SHOULD include both English and Chinese when appropriate.

---

## 0. Metadata (REQUIRED)
- id: <string, unique and stable; example: wave_equation_scalar_v1>
- equation_name: <English canonical name>
- aliases:
  - <alias_1>
  - <alias_2>
  - <... include common Chinese alias if applicable>
- tags:
  - domain: [<acoustics>, <fluid>, <solid>, <thermal>, <em>, <geo>, <math>, ...]
  - pde_class: [<elliptic> | <parabolic> | <hyperbolic> | <mixed>]
  - linearity: [<linear> | <nonlinear>]
  - time: [<steady> | <transient>]
  - form: [<strong> | <weak> | <conservation>]
- keywords:
  - <keyword_1>
  - <keyword_2>
  - <... include common Chinese query keywords if useful>
- related_equations:
  - <equation_id_1>
  - <equation_id_2>

**Do**
- Keep tags and keywords short and retrieval-friendly.
- Include both canonical names and user-facing synonyms in aliases.

**Don't**
- Do not use long paragraphs in metadata.
- Do not omit id or tags.

---

## 1. Background (REQUIRED)
- phenomenon: <1–2 sentences describing what the equation models>
- typical_applications:
  - <application_1>
  - <application_2>
- assumptions:
  - <assumption_1>
  - <assumption_2>
- invalid_when:
  - <failure_mode_1>
  - <failure_mode_2>

**Do**
- State modeling assumptions explicitly (e.g., incompressible, small deformation, isotropic).
- List "invalid_when" to prevent misuse.

**Don't**
- Do not include numerical values unless they are universal constants.

---

## 2. Variables & Parameters (REQUIRED)
- independent_variables:
  - space: x ∈ Ω ⊂ R^d
  - time: t ∈ [0, T] (if transient; otherwise omit)
- dependent_fields:
  - <field_symbol>(x,t): <meaning>, unit: <...>
  - <...>
- parameters:
  - <param_symbol>: <meaning>, unit: <...>, typical_range: <...>, known_or_unknown: <known|unknown>
  - <...>
- observables:
  - <measurable_quantity_1>
  - <measurable_quantity_2>

**Do**
- Define what each symbol represents and its unit.
- Mark each parameter as known or unknown (for inverse tasks).

**Don't**
- Do not leave units unspecified unless truly application-dependent (state that explicitly).

---

## 3. Governing Equation (REQUIRED)

### 3.1 Strong form (PRIMARY, REQUIRED)
$$
<write the primary PDE in strong form; be consistent with symbols in Section 2>
$$

### 3.2 Alternative forms (OPTIONAL, recommended)
- conservation form (if applicable):
$$
<...>
$$
- first-order system form (if useful for PINN/solver):
$$
<...>
$$

### 3.3 Non-dimensional form (OPTIONAL but recommended)
$$
<...>
$$
- dimensionless_numbers:
  - <name>: <definition>, interpretation: <...>
  - <...>

**Do**
- Use LaTeX equations only with $$ ... $$.
- If coefficients vary in space/time, use the correct operator form (e.g., div(c(x)^2 grad u)).

**Don't**
- Do not invent variants not used in practice.
- Do not change symbol meanings across sections.

---

## 4. Initial and Boundary Conditions (REQUIRED)

### 4.1 Initial conditions (REQUIRED if transient)
- IC-1:
$$
<...>
$$
- IC-2:
$$
<...>
$$

### 4.2 Boundary condition catalog (REQUIRED)
Provide a short catalog of admissible BC types for this equation.
Do NOT list every BC in existence—only those commonly used and meaningful here.

- Dirichlet:
$$
<...>
$$
- Neumann:
$$
<...>
$$
- Robin/Impedance (if relevant):
$$
<...>
$$
- Periodic / Symmetry / Radiation(ABC/PML) (if relevant):
<brief description; give typical expression if standard>

**Do**
- Ensure the BC catalog covers typical closed-domain and (if needed) open-domain cases.
- Mention open-domain strategy (ABC/PML) when wave-like propagation is involved.

**Don't**
- Do not leave BC vague; at least provide canonical expressions.
- Do not assume boundary conditions are optional; a well-posed problem needs BC on entire ∂Ω.

---

## 5. Domain & Typical Setups (REQUIRED)
- geometry_templates:
  - <interval/rectangle/box/cylinder/custom_mesh/...>
- typical_benchmarks:
  - <benchmark_1>
  - <benchmark_2>

**Do**
- Provide 1–2 classic setups that are widely recognized in the community.

**Don't**
- Do not include code, datasets, or long tutorial content.

---

## 6. Inverse Problems / Identification (REQUIRED if meaningful; otherwise write "N/A")
- invertible_targets:
  - <target_1>
  - <target_2>
- required_data:
  - <data_type> at <where/when>
- identifiability_notes:
  - <pitfall_1>
  - <pitfall_2>
- recommended_priors_or_regularization:
  - <smoothness/TV/bounds/physics prior>

**Do**
- Explicitly warn about ill-posedness and what data makes it identifiable.

**Don't**
- Do not claim identifiability without stating data requirements.

---

## 7. Constraint Decomposition (for agent compilation) (REQUIRED)
Write residuals in a solver/PINN-friendly form.
Use the same symbols as Sections 2–3.

- PDE_residual:
  - r_pde(x,t) = <...>
- IC_residuals:
  - r_ic_1 = <...>
  - r_ic_2 = <...>
- BC_residuals:
  - r_bc_D = <...>
  - r_bc_N = <...>
  - r_bc_R = <...> (if used)
  - r_bc_open = <...> (if used; e.g., ABC/PML-dependent placeholder)
- data_residuals (optional):
  - r_data = <...>
- weighting_strategy (optional):
  - <curriculum/dynamic weights/normalization hints>

**Do**
- Keep residuals explicit and symbolic (no code).
- If open-boundary residual depends on a specific scheme, write a placeholder and name the scheme.

**Don't**
- Do not omit Section 7; it is critical for producing high-quality implementations later.

---

## 8. Minimal Problem Spec Checklist (REQUIRED)
These items MUST be satisfiable before compilation to code.
Unchecked items indicate missing information that must be provided by the user
or completed by the agent using safe defaults.

- [ ] spatial dimension d, domain Ω, time window [0,T] (if transient)
- [ ] physical meaning of primary field(s)
- [ ] parameters specified (known/unknown, bounds if unknown)
- [ ] source term or excitation mechanism (if applicable)
- [ ] initial conditions complete (if transient)
- [ ] boundary conditions specified on entire ∂Ω
- [ ] observation data specified (required for inverse problems)
- [ ] required outputs and evaluation metrics

**Do**
- Keep items minimal and necessary.
- Ensure checklist matches what is required for this equation.

**Don't**
- Do not treat checklist as optional feature flags.

---

## 9. Minimal Example (toy specification, NO CODE) (REQUIRED)
- setup:
  - <dimension, domain, time window>
  - <parameter values or symbolic ranges>
  - <IC/BC>
  - <source>
- expected_behavior:
  - <qualitative behavior; e.g., diffusion smooths, waves reflect, steady solves Laplace, etc.>

**Do**
- Provide a canonical toy setup that is self-contained.

**Don't**
- Do not include implementation details or training tricks.

---

## 10. References (REQUIRED)
- <textbook / review / documentation>
- <... keep 2–6 items>

**Do**
- Prefer authoritative references (textbooks, review papers, official docs).

**Don't**
- Do not paste long quotations.

---

## Chinese Notes (Optional)
<Only for human maintainers. Must not contain additional technical constraints
that are absent from the English sections.>
