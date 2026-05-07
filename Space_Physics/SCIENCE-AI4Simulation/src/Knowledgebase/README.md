# 📘 Math_Physics_Formulations

**Agent-Oriented Mathematical & Physical Equation Knowledge Base**

---

## 1. 项目定位（What & Why）

`Math_Physics_Formulations` 是 **AI4PDE / XIMUAlpha** 体系中的**基础方程知识库**，其目标不是存放“教材式公式”，而是：

> **为 AI Agent 提供可学习、可检索、可组合的物理与数学建模原子知识**，
> 从而在 **未命中具体案例（fallback / adaptive）时**，依然能够：
>
> * 构造**专业、规范、物理合理**的控制方程；
> * 自动选择合适的 **边界条件 / 初始条件 / 源项**；
> * 支撑 PINN / 数值方法 / 反问题 等建模流程。

该知识库是 **Agent 的“物理常识层”**，不是用户文档，也不是代码仓库。

---

## 2. 设计原则（Design Principles）

### 2.1 面向 Agent，而非人类教材

* 所有内容：

  * 偏重 **建模结构**，而非推导细节；
  * 强调 **可迁移性 / 可组合性 / 可复用性**。
* 默认读者是 **LLM Agent**，而不是初学者。

---

### 2.2 Equation Card = 最小完备建模单元

每一个 `equations/*.md` 文件都必须是一个**独立、完备、可复用的方程卡片**，而不是某个项目的附属说明。

---

### 2.3 Components 是跨方程共享的“原子库”

* Boundary Conditions
* Initial Conditions
* Source / Forcing Terms

**只能在 `components/` 中定义一次**，
Equation Card 只能 **引用或选择**，不能重复发明。

---

### 2.4 Agent 负责“组装”，知识库负责“约束”

* 知识库：

  * 提供 **合法的选项空间**
  * 提供 **建模范式**
* Agent：

  * 根据问题语义选择
  * 拼接成完整 PDE 建模方案

---

## 3. 目录结构说明

```text
Math_Physics_Formulations/
├── README.md                # 本文件（总览与进度）
├── agent.md                 # Agent 行为契约（如何使用该知识库）
│
├── equations/               # 方程知识卡片（Equation Cards）
│   ├── poisson_equation.md
│   ├── wave_equation.md
│   ├── heat_equation.md
│   ├── maxwell_equations.md
│   └── navier_stokes_incompressible.md
│
├── components/              # 可复用建模组件库
│   ├── boundary_conditions.md
│   ├── initial_conditions.md
│   └── source_terms.md
│
└── templates/
    └── equation_card.template.md
```

---

## 4. Equation Card 规范（强制）

所有 `equations/*.md` **必须严格遵循**
`templates/equation_card.template.md`，包括：

* Metadata（id / type / domain）
* Governing Equation
* Variables & Physical Meaning
* Domain & Dimensionality
* Boundary / Initial / Source（**引用 components**）
* Notes / Assumptions
* Inverse / Learning Remarks（如适用）

**禁止自由发挥结构。**

---

## 5. 已完成方程（✅）

### 基础椭圆 / 抛物 / 双曲方程

* ✅ **Poisson Equation**
* ✅ **Heat Equation**
* ✅ **Scalar Wave Equation**

### 电磁与场论

* ✅ **Maxwell Equations**

### 流体力学

* ✅ **Incompressible Navier–Stokes Equations**

---

## 6. 进行中 / 已规划方程（🟡 / ⬜）

### ⭐ 第一优先级（Agent 泛化核心）

* ⬜ **Advection–Diffusion Equation**
* ⬜ **Burgers’ Equation**
* ⬜ **Stokes Equation**

### ⭐⭐ 第二优先级（工程 / 工业标准）

* ⬜ **Linear Elasticity (Navier–Cauchy)**
* ⬜ **Helmholtz Equation**
* ⬜ **Advection–Reaction–Diffusion Equation**

### ⭐⭐⭐ 第三优先级（研究 / PINN Benchmark）

* ⬜ **KdV Equation**
* ⬜ **Laplace Equation**
* ⬜ **Schrödinger Equation (Linear)**

---

## 7. Components 状态（✅）

* ✅ Boundary Conditions Catalog
* ✅ Initial Conditions Catalog
* ✅ Source / Forcing Terms Catalog

> 所有 Equation Card **必须从这里选型**，
> 不允许在方程卡片中重新定义通用 BC / IC / Source。

---

## 8. 与 AI4PDE Workflow 的关系

在 AI4PDE 的 `Coding(Action)` 中：

* 当 **案例检索未命中** 或 **Adaptive Mode 需要泛化** 时：

  1. 使用 `EquationRetriever` 检索本知识库；
  2. 构造 `eq_context`；
  3. 将其注入：

     * `FALLBACK_STAGE1_PROMPT`
     * `ADAPTIVE_STAGE1_PROMPT`
  4. 引导 Agent 构造 **高质量 PDE 建模方案**。

---
## 9. 核心目标

> **让 AI Agent 在“没有现成答案”的情况下，依然像一个受过严格训练的工程与物理研究人员那样，构建方程。**

