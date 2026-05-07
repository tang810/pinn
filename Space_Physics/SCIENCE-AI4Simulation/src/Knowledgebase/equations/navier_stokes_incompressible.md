# Equation Card: Incompressible Navier–Stokes Equations

## 0. Metadata (REQUIRED)
- id: navier_stokes_incompressible
- title: Incompressible Navier–Stokes Equations
- aliases: 不可压缩Navier–Stokes方程, 非压缩流体方程
- type: Nonlinear PDE System
- canonical_fields: u, p
- keywords: fluid dynamics, incompressible flow, viscous flow, laminar, turbulence precursor
- typical_dimensions: 2D, 3D
- related_components:
  - components/boundary_conditions.md
  - components/initial_conditions.md
  - components/source_terms.md

## 1. Background (REQUIRED)
The incompressible Navier–Stokes equations describe the motion of **viscous,
incompressible Newtonian fluids**. They form the foundation of classical
fluid mechanics and appear in aerodynamics, hydrodynamics, environmental
flows, and microfluidic systems.

These equations are nonlinear due to the convective acceleration term and
pose significant challenges for both numerical solvers and physics-informed
neural networks.

## 2. Variables & Parameters (REQUIRED)

### 2.1 Primary Fields
- Velocity field:
$$
\mathbf{u}(\mathbf{x},t) = (u_1, u_2 [, u_3])
$$
- Pressure field:
$$
p(\mathbf{x},t)
$$

### 2.2 Physical Parameters
- Density (constant for incompressible flow):
$$
\rho
$$
- Dynamic viscosity:
$$
\mu
$$
- Kinematic viscosity:
$$
\nu = \frac{\mu}{\rho}
$$

### 2.3 External Forcing
- Body force per unit volume:
$$
\mathbf{f}(\mathbf{x},t)
$$
(e.g. gravity, electromagnetic forcing, buoyancy approximation)

## 3. Governing Equations (REQUIRED)

### 3.1 Momentum Conservation
On domain $$\Omega \subset \mathbb{R}^d$$:
$$
\rho\left(
\frac{\partial \mathbf{u}}{\partial t}
+ \mathbf{u}\cdot\nabla \mathbf{u}
\right)
=
- \nabla p
+ \mu \Delta \mathbf{u}
+ \mathbf{f},
\quad \mathbf{x}\in\Omega
$$

### 3.2 Mass Conservation (Incompressibility Constraint)
$$
\nabla \cdot \mathbf{u} = 0
$$

This constraint couples velocity and pressure and must be enforced
explicitly in PINN formulations.

### 3.3 Non-dimensional Form (Optional)
Using characteristic velocity $$U$$ and length $$L$$:
$$
\frac{\partial \mathbf{u}}{\partial t}
+ \mathbf{u}\cdot\nabla \mathbf{u}
=
- \nabla p
+ \frac{1}{Re}\Delta \mathbf{u}
$$
where
$$
Re = \frac{UL}{\nu}
$$
is the Reynolds number.

## 4. Initial and Boundary Conditions (REQUIRED)

### 4.1 Initial Conditions
$$
\mathbf{u}(\mathbf{x},0) = \mathbf{u}_0(\mathbf{x})
$$

Pressure initial conditions are usually **not required** and can be
determined up to a constant.

### 4.2 Boundary Conditions
On $$\partial\Omega$$, common choices include:

**(A) No-slip Wall**
$$
\mathbf{u} = \mathbf{0}
$$

**(B) Prescribed Velocity (Inflow)**
$$
\mathbf{u} = \mathbf{u}_{in}(\mathbf{x},t)
$$

**(C) Stress / Pressure Outlet**
$$
\left(
- p\mathbf{I}
+ \mu (\nabla\mathbf{u} + \nabla\mathbf{u}^T)
\right)\mathbf{n}
= \mathbf{t}_{out}
$$

**(D) Periodic Boundary**
$$
\mathbf{u}(\mathbf{x}_1,t) = \mathbf{u}(\mathbf{x}_2,t)
$$

## 5. Domain & Typical Setups (REQUIRED)

### 5.1 Typical Domains
- Channels and pipes
- Cavities (lid-driven cavity)
- External flow around obstacles
- Microfluidic geometries

### 5.2 Typical Flow Regimes
- Creeping flow ($$Re \ll 1$$)
- Laminar flow
- Transitional flow (precursor to turbulence)

## 6. Inverse Problems / Identification (REQUIRED if meaningful)
Common inverse problems include:
- Estimation of viscosity $$\nu$$
- Inference of unknown forcing $$\mathbf{f}$$
- Boundary condition identification
- Data-driven closure modeling (precursor to RANS/LES)

Observations may include:
$$
\mathbf{u}(\mathbf{x}_m,t_n), \quad p(\mathbf{x}_m,t_n)
$$

## 7. Constraint Decomposition (for agent compilation) (REQUIRED)

### 7.1 Momentum Residual
$$
\mathbf{r}_{mom}
=
\rho\left(
\frac{\partial \mathbf{u}}{\partial t}
+ \mathbf{u}\cdot\nabla \mathbf{u}
\right)
+ \nabla p
- \mu \Delta \mathbf{u}
- \mathbf{f}
$$

### 7.2 Continuity Residual
$$
r_{div} = \nabla \cdot \mathbf{u}
$$

### 7.3 Boundary Residuals
- Velocity BC:
$$
\mathbf{r}_{bc} = \mathbf{u} - \mathbf{u}_b
$$

### 7.4 Initial Residual
$$
\mathbf{r}_{ic} = \mathbf{u}(\mathbf{x},0) - \mathbf{u}_0(\mathbf{x})
$$

## 8. Minimal Problem Spec Checklist (REQUIRED)
Before solving, specify:
1) Spatial dimension and geometry $$\Omega$$  
2) Density $$\rho$$ and viscosity $$\mu$$ or $$\nu$$  
3) Reynolds number (if nondimensionalized)  
4) Initial velocity field $$\mathbf{u}_0$$  
5) Boundary condition types for all boundaries  
6) Forcing term $$\mathbf{f}$$ (or zero)  
7) Outputs of interest (velocity, pressure, forces)  
8) Observation data (if inverse problem)

## 9. Minimal Example (toy specification, NO CODE) (REQUIRED)

**Toy: 2D Lid-Driven Cavity**
- Domain: $$\Omega=[0,1]\times[0,1]$$
- Top lid: $$\mathbf{u}=(1,0)$$
- Other walls: no-slip
- Initial: $$\mathbf{u}=0$$
- Parameters: constant $$\rho,\mu$$
- Goal: steady-state velocity and pressure fields

## 10. References (REQUIRED)
- P. Moin, *Fundamentals of Engineering Numerical Analysis*
- Batchelor, *An Introduction to Fluid Dynamics*
- Temam, *Navier–Stokes Equations*
- Karniadakis et al., *Physics-Informed Machine Learning*

## Chinese Notes (Optional)
- 不可压缩 Navier–Stokes 的核心难点在于 **非线性对流项 + 散度自由约束**。
- PINN 中常将连续性方程作为独立残差项。
- 高雷诺数问题通常需要时间加权、分区采样或多尺度建模。
