# Equation Card: Heat Equation (Transient Conduction)

## 0. Metadata (REQUIRED)
- id: heat_equation_transient
- title: Heat Equation (Transient Conduction)
- aliases: 热传导方程, 热扩散方程, Heat Diffusion Equation
- type: Parabolic PDE
- canonical_fields: T
- keywords: heat conduction, diffusion, parabolic, transient, steady-state limit
- typical_dimensions: 1D, 2D, 3D
- related_components:
  - components/boundary_conditions.md
  - components/initial_conditions.md
  - components/source_terms.md

## 1. Background (REQUIRED)
Heat equation describes the **temporal evolution of temperature fields** driven by
conduction and internal heat sources. It is the canonical **parabolic PDE** and
appears across thermal engineering, materials science, geophysics, and
electro-thermal coupling problems.

In steady-state limits, the heat equation reduces to the **Poisson equation**.
This equation is often used as a benchmark for time-dependent PINN and diffusion solvers.

## 2. Variables & Parameters (REQUIRED)

### 2.1 Primary Field
- Temperature:
$$
T(\mathbf{x},t)
$$

### 2.2 Material Parameters
- Density:
$$
\rho(\mathbf{x})
$$
- Specific heat capacity:
$$
c_p(\mathbf{x})
$$
- Thermal conductivity:
$$
k(\mathbf{x})
$$

### 2.3 Source Term
- Volumetric heat generation:
$$
Q(\mathbf{x},t)
$$

### 2.4 Derived Quantity
- Thermal diffusivity:
$$
\alpha(\mathbf{x}) = \frac{k(\mathbf{x})}{\rho(\mathbf{x})c_p(\mathbf{x})}
$$

## 3. Governing Equation (REQUIRED)

### 3.1 Transient Heat Equation (General Form)
On domain $$\Omega \subset \mathbb{R}^d$$:
$$
\rho c_p \frac{\partial T}{\partial t}
=
\nabla \cdot \left( k \nabla T \right) + Q(\mathbf{x},t),
\quad \mathbf{x}\in\Omega,\ t>0
$$

### 3.2 Simplified Constant-Coefficient Form
If $$\rho, c_p, k$$ are constants:
$$
\frac{\partial T}{\partial t}
=
\alpha \Delta T + \frac{Q}{\rho c_p}
$$

### 3.3 Steady-State Limit
If $$\partial T/\partial t = 0$$:
$$
- \nabla \cdot (k \nabla T) = Q(\mathbf{x})
$$
which reduces to the **Poisson equation**.

## 4. Initial and Boundary Conditions (REQUIRED)

### 4.1 Initial Condition (Required for Transient Problems)
$$
T(\mathbf{x},0) = T_0(\mathbf{x})
$$

### 4.2 Boundary Conditions
On $$\partial\Omega$$, typical choices include:

**(A) Dirichlet (Prescribed Temperature)**
$$
T(\mathbf{x},t) = T_b(\mathbf{x},t)
$$

**(B) Neumann (Prescribed Heat Flux / Insulated)**
$$
- k \nabla T \cdot \mathbf{n} = q(\mathbf{x},t)
$$
Insulated boundary corresponds to $$q=0$$.

**(C) Robin / Convective Boundary**
$$
- k \nabla T \cdot \mathbf{n}
= h\left(T - T_\infty\right)
$$
(Newton’s cooling law)

## 5. Domain & Typical Setups (REQUIRED)

### 5.1 Typical Domains
- Rods, plates, and solids (1D–3D)
- Heterogeneous materials with spatially varying $$k$$
- Composite or layered structures

### 5.2 Typical Problem Types
- **Transient heating/cooling**
- **Thermal diffusion**
- **Steady-state thermal equilibrium**
- **Electro-thermal or thermo-mechanical coupling**

## 6. Inverse Problems / Identification (REQUIRED if meaningful)
Common inverse problems include:
- Identification of $$k(\mathbf{x})$$ or $$\alpha(\mathbf{x})$$
- Estimation of unknown heat sources $$Q(\mathbf{x},t)$$
- Boundary heat transfer coefficient $$h$$ estimation
- Thermal property inference from sparse temperature sensors

Observations typically include:
$$
T(\mathbf{x}_m,t_n)
$$
at discrete sensor locations.

## 7. Constraint Decomposition (for agent compilation) (REQUIRED)

### 7.1 PDE Residual
$$
r_{pde}
=
\rho c_p \frac{\partial T}{\partial t}
- \nabla \cdot (k \nabla T)
- Q
$$

### 7.2 Boundary Residuals
- Dirichlet:
$$
r_{bc}^{D} = T - T_b
$$
- Neumann:
$$
r_{bc}^{N} = -k\nabla T\cdot \mathbf{n} - q
$$
- Robin:
$$
r_{bc}^{R} = -k\nabla T\cdot \mathbf{n} - h(T-T_\infty)
$$

### 7.3 Initial Residual
$$
r_{ic} = T(\mathbf{x},0) - T_0(\mathbf{x})
$$

## 8. Minimal Problem Spec Checklist (REQUIRED)
Before solving, specify:
1) Spatial dimension and domain $$\Omega$$  
2) Material parameters $$\rho, c_p, k$$ (constant or variable)  
3) Time window $$[0,T]$$  
4) Source term $$Q(\mathbf{x},t)$$ (or zero)  
5) Initial temperature $$T_0(\mathbf{x})$$  
6) Boundary conditions covering $$\partial\Omega$$  
7) Outputs of interest (field, flux, averages)  
8) Inverse targets (if any)

## 9. Minimal Example (toy specification, NO CODE) (REQUIRED)

**Toy: 2D Plate Cooling**
- Domain: $$\Omega=[0,L_x]\times[0,L_y]$$
- Parameters: constant $$\rho,c_p,k$$
- Initial: $$T_0(\mathbf{x}) = T_{hot}$$
- Boundary:
  - Left/right: insulated
  - Top/bottom: convective cooling
- Source: $$Q=0$$
- Goal: transient temperature evolution and steady equilibrium

## 10. References (REQUIRED)
- J. Crank, *The Mathematics of Diffusion*
- H. S. Carslaw, J. C. Jaeger, *Conduction of Heat in Solids*
- Özisik, *Heat Conduction*
- Incropera et al., *Fundamentals of Heat and Mass Transfer*

## Chinese Notes (Optional)
- 热方程是最典型的抛物型 PDE，**必须给初值**。
- 稳态热问题不需要初值，应使用 Poisson equation card。
- 在 PINN 中，时间尺度与扩散尺度不匹配时常需非均匀采样或时间加权损失。
