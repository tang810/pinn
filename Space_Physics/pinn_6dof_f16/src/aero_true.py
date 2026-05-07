from __future__ import annotations

def aero_true(alpha: float, beta: float, p_hat: float, q_hat: float, r_hat: float,
              de: float, da: float, dr: float):
    """合成“类 F-16”气动模型（多项式形式），用于生成训练数据真值。

    返回: (CX, CY, CZ, Cl, Cm, Cn)
    """
    CX0   = -0.1
    CXa   = -0.3
    CXq   = -0.5
    CXde  =  0.0

    CYb   = -0.8
    CYda  =  0.08
    CYdr  =  0.105
    CYp   = -0.03
    CYr   =  0.21

    CZ0   = -0.3
    CZa   = -5.5
    CZq   = -8.0
    CZde  = -0.9

    Clb   = -0.12
    Clda  =  0.16
    Cldr  =  0.105
    Clp   = -0.5
    Clr   =  0.25

    Cm0   =  0.05
    Cma   = -1.5
    Cmq   = -20.0
    Cmde  = -1.0

    Cnb   =  0.25
    Cnda  =  0.06
    Cndr  = -0.2
    Cnp   = -0.02
    Cnr   = -0.3

    CX = CX0 + CXa*alpha + CXq*q_hat + CXde*de
    CY = CYb*beta + CYda*da + CYdr*dr + CYp*p_hat + CYr*r_hat
    CZ = CZ0 + CZa*alpha + CZq*q_hat + CZde*de
    Cl = Clb*beta + Clda*da + Cldr*dr + Clp*p_hat + Clr*r_hat
    Cm = Cm0 + Cma*alpha + Cmq*q_hat + Cmde*de
    Cn = Cnb*beta + Cnda*da + Cndr*dr + Cnp*p_hat + Cnr*r_hat

    return CX, CY, CZ, Cl, Cm, Cn
