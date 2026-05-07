# Math_Physics_Formulations — Agent Instruction Manual

This document defines how an AI agent should interact with the
Math_Physics_Formulations knowledge base.

This directory provides equation-level knowledge for mathematical
and physical modeling, intended to be used when no task-specific
case or implementation is directly available.

The agent MUST follow the rules below strictly.

---

## 1. Purpose and Scope

Math_Physics_Formulations serves as a **foundational equation knowledge base**.

It is used to:
- understand governing equations and modeling assumptions,
- construct mathematically complete problem definitions,
- support forward simulation and inverse/identification tasks,
- bridge incomplete user queries to executable problem specifications.

This directory is **not**:
- a code repository,
- a tutorial collection,
- a numerical implementation guide.

---

## 2. When to Use This Knowledge Base

The agent SHOULD enter Math_Physics_Formulations when **any** of the following is true:

1. No existing case or example matches the user query.
2. The user asks a general modeling question (e.g. “how to model…”, “what equation governs…”).
3. The user query specifies a physical phenomenon but lacks a formal equation.
4. The agent needs to verify completeness or correctness of a problem definition.
5. The task involves constructing a PDE, IC/BC, or inverse formulation from scratch.

The agent SHOULD NOT use this directory if:
- a fully matching, task-specific case already exists and is applicable without modification.

---

## 3. Knowledge Units and Their Roles

The knowledge base is composed of the following unit types:

### 3.1 Equation Cards (`equations/*.md`)
Each equation card represents a **single governing equation** or tightly coupled equation system.

Equation cards define:
- governing equations (strong form, optional alternatives),
- variables and parameters,
- modeling assumptions and invalid regimes,
- admissible initial and boundary conditions,
- inverse/identification targets,
- constraint decomposition for compilation.

Each equation card is treated as an **atomic modeling primitive**.

---

### 3.2 Components (`components/*.md`)
Components provide **reusable modeling elements**, such as:
- boundary condition types,
- initial condition patterns,
- source/excitation terms,
- nondimensional numbers,
- sampling strategies.

Components are not complete models by themselves.
They are referenced by equation cards and assembled into problems.

---

### 3.3 Templates (`templates/*.md`)
Templates define **authoring standards** for adding new content.

The agent MUST assume that all valid equation cards and components
conform to the templates.

---

### 3.4 Registry (`registry/*.json`)
Registry files provide **lightweight indexing** for retrieval:
- equation identifiers,
- aliases and keywords,
- file paths.

The agent SHOULD consult the registry before scanning full documents.

---

## 4. Agent Workflow (Mandatory)

When using Math_Physics_Formulations, the agent MUST follow this workflow:

### Step 1 — Identify Modeling Intent
From the user query, determine:
- physical phenomenon,
- task type: forward / inverse / control / analysis,
- spatial and temporal nature (steady vs transient).

---

### Step 2 — Select Governing Equation(s)
- Search the registry for matching equations using aliases and keywords.
- Select the **most appropriate equation card**.
- If multiple candidates exist, prefer the simplest valid model.

---

### Step 3 — Extract Modeling Requirements
From the selected equation card, extract:
- unknown fields,
- required parameters,
- admissible IC/BC types,
- source/excitation options,
- inverse targets (if applicable).

---

### Step 4 — Validate Problem Completeness
Check the **Minimal Problem Spec Checklist** defined in the equation card.

For each unchecked item:
- determine whether it can be safely assumed,
- or must be explicitly requested from the user.

Unchecked items indicate **missing but required information**.

---

### Step 5 — Complete Missing Information
Missing items MUST be handled by one of the following strategies:

1. **User clarification**  
   Ask targeted, minimal questions.

2. **Default safe assumptions**  
   Use canonical benchmark settings when appropriate
   (e.g. unit domain, zero initial velocity).

3. **Parameterized placeholders**  
   Introduce symbolic parameters with explicit bounds or priors.

The agent MUST document which strategy was used.

---

### Step 6 — Produce a Structured Problem Specification
The final output of this workflow is a **Problem Specification** that includes:
- governing equation identifier,
- fields and parameters (known/unknown),
- domain and time window,
- initial and boundary conditions,
- source terms,
- data/observations (if any),
- objectives and outputs.

This specification MUST be internally consistent and complete.

---

## 5. Interpretation Rules (Critical)

### 5.1 Checklist Semantics
- `[ ]` indicates a required item that is not yet satisfied.
- `[x]` indicates the item is fully specified.

Unchecked items MUST NOT be ignored.

---

### 5.2 Language Convention
- Structured content is in **English**.
- Chinese text is auxiliary and MUST NOT affect reasoning.
- Aliases and keywords MAY include both English and Chinese.

---

### 5.3 Assumptions and Validity
The agent MUST:
- respect modeling assumptions explicitly stated in equation cards,
- warn the user when assumptions are likely violated,
- avoid over-claiming model validity.

---

## 6. Inverse and Identification Tasks

For inverse problems, the agent MUST:
- identify which parameters are invertible,
- verify data sufficiency qualitatively,
- recommend regularization or priors when ill-posedness is likely.

The agent SHOULD avoid constructing inverse problems
that are fundamentally non-identifiable.

---

## 7. Output Discipline

The agent MUST NOT:
- generate code directly from equation cards,
- invent equations not present in the knowledge base without justification,
- skip IC/BC specification.

The agent SHOULD:
- remain equation-centric,
- prioritize correctness and completeness over brevity,
- clearly separate assumptions from user-provided facts.

---

## 8. Design Philosophy (For Agent Alignment)

Math_Physics_Formulations represents **first-principles modeling knowledge**.

The agent’s role is to:
- reason like a mathematical modeler,
- assemble equations like a domain expert,
- and only then delegate implementation to downstream systems.

This knowledge base is the **source of truth** for modeling logic.
