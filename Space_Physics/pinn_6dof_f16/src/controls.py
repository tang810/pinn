import math
from typing import Iterable, Tuple

def multi_sine(t: float, amps: Iterable[float], freqs: Iterable[float], phases: Iterable[float]) -> float:
    """多正弦叠加信号。"""
    y = 0.0
    for A, f, p in zip(amps, freqs, phases):
        y += A * math.sin(2.0 * math.pi * f * t + p)
    return y

def control_inputs(t: float) -> Tuple[float, float, float]:
    """给定时间 t，返回 (de, da, dr) 舵面偏转量，单位：弧度。"""
    # 升降舵 elevator
    de_amps   = [10.0*math.pi/180.0, 5.0*math.pi/180.0]
    de_freqs  = [0.2, 0.7]
    de_phases = [0.0, 1.0]

    # 副翼 aileron
    da_amps   = [8.0*math.pi/180.0, 4.0*math.pi/180.0]
    da_freqs  = [0.15, 0.55]
    da_phases = [0.5, 2.0]

    # 方向舵 rudder
    dr_amps   = [6.0*math.pi/180.0, 3.0*math.pi/180.0]
    dr_freqs  = [0.25, 0.9]
    dr_phases = [1.0, 0.3]

    de = multi_sine(t, de_amps, de_freqs, de_phases)
    da = multi_sine(t, da_amps, da_freqs, da_phases)
    dr = multi_sine(t, dr_amps, dr_freqs, dr_phases)
    return de, da, dr
