#!/usr/bin/env python
"""
Raptor 3 Engine Full Reactive Flow Solver
===========================================
Internal: Quasi-1D isentropic nozzle with CH4/LOX combustion
External: Analytical plume model with Mach diamonds
Output:    VTK (ParaView), PNG plots, GIF animation

Usage:    python main.py
"""

import os
import time

import numpy as np

from src.config import Config
from src.geometry import init_geometry
from src.nozzle import Quasi1DNozzle
from src.plume import PlumeModel
from src.vtk_writer import FullVTKWriter
from src.vtk_writer_vtk import VTKWriter as VTKWriter2
from src.postprocess import (create_summary_plots, create_thrust_curve,
                             create_gif_from_vtk)


def main():
    print('=' * 60)
    print('  Raptor 3 Engine - Full Reactive Flow Solver')
    print('  Internal: Quasi-1D CH4/LOX Nozzle')
    print('  External: Analytical Plume (Mach diamonds)')
    print('=' * 60)

    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ---- Calibrate geometry for target mass flow ----
    init_geometry(cfg)

    # ---- Internal Nozzle ----
    print(f'\n[1] Solving internal nozzle flow ({cfg.NX_INT} cells)...')
    t0 = time.time()
    nozzle = Quasi1DNozzle(cfg)
    nozzle.solve()
    nozzle.print_summary()
    print(f'    Time: {time.time() - t0:.1f}s')

    # ---- External Plume ----
    print(f'\n[2] Generating analytical plume (Mach diamonds)...')
    plume = PlumeModel(cfg, nozzle, nx_plume=cfg.NX_EXT, ny=cfg.NY)
    print(f'    Diamond spacing: {plume.diamond_spacing:.2f} m')
    print(f'    Jet exit: M={nozzle.M[-1]:.2f}, P={nozzle.P[-1] / 1e3:.0f}kPa')

    # ---- VTK Output (prefer vtk module, fallback to hand-written) ----
    try:
        vtk_writer = VTKWriter2(cfg, nozzle, plume)
        print(f'\n[3] Writing VTM output (vtk module)...')
    except Exception as e:
        vtk_writer = FullVTKWriter(cfg, nozzle, plume)
        print(f'\n[3] Writing VTM output (hand-written, vtk module failed: {e})...')
    base_name = os.path.join(cfg.OUTPUT_DIR, 'raptor3_full')
    vtk_writer.write(base_name + '.vtm', 0)

    # ---- Animation frames ----
    print(f'\n[4] Generating animation frames...')
    for frame in range(20):
        phase = frame * np.pi / 10.0
        plume._compute_plume_with_phase(phase)
        plume._compute_species()
        fname = os.path.join(cfg.OUTPUT_DIR,
                             f'raptor3_full_{frame:04d}.vtm')
        vtk_writer.write(fname, frame)

    # ---- Post-Processing ----
    print('\n[5] Creating summary plots...')
    create_summary_plots(cfg, nozzle)
    create_thrust_curve(cfg, nozzle)

    print('\n[6] Creating GIF animation...')
    create_gif_from_vtk(cfg)

    # ---- Final Report ----
    F, Fm, Fp, mdot = nozzle.thrust()
    isp_val = nozzle.isp()

    print(f'\n{"=" * 55}')
    print(f'  FINAL PERFORMANCE REPORT')
    print(f'{"=" * 55}')
    print(f'  Chamber:    P={cfg.P_CHAMBER / 1e6:.1f} MPa  '
          f'T={cfg.T_CHAMBER:.0f} K')
    print(f'  O/F ratio:  {cfg.OF_RATIO:.2f}  '
          f'(CH4={cfg.M_CH4:.0f} LOX={cfg.M_LOX:.0f}) kg/s')
    print(f'  Mass flow:  {mdot:.1f} kg/s')
    print(f'  Exit Mach:  {nozzle.M[-1]:.2f}')
    print(f'  Exit u:     {nozzle.u[-1]:.0f} m/s')
    print(f'  Exit P:     {nozzle.P[-1] / 1e3:.1f} kPa')
    print(f'  Expansion:  {nozzle.A[-1] / nozzle.A_star:.1f}:1')
    print(f'  {"-" * 50}')
    print(f'  THRUST:     {F / 1e6:.3f} MN  '
          f'({F / 1e3:.0f} kN = {F / 1e3 / 9.81:.0f} tonnes)')
    print(f'  Isp (sea):  {isp_val:.1f} s')
    print(f'{"=" * 55}')

    print(f'\n  Output files in: {cfg.OUTPUT_DIR}/')
    for fn in ['raptor3_full.vtm', 'raptor3_full_S2.vts',
               'raptor3_full_S3.vts', 'raptor3_full_S4.vts',
               'raptor3_summary.png', 'raptor3_thrust.png',
               'raptor3_plume.gif']:
        fpath = os.path.join(cfg.OUTPUT_DIR, fn)
        if os.path.exists(fpath):
            print(f'    {fn}')

    return nozzle, plume


if __name__ == '__main__':
    main()
