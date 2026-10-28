# -*- coding: utf-8 -*-
"""
Reproduce Grimholt & Skogestad (2018) Table 3 for all four benchmark processes.
Improvements over the first attempt:
  * safe objective (large penalty for nan/inf / unstable closed loop)
  * multi-start (several initial guesses, keep the best valid result)
  * pure-delay process optimised as PI (tau_d = 0), matching the paper
"""
import math
import numpy as np
from scipy import signal
from scipy.optimize import minimize

TAU_F = 0.03
N_PADE = 12
BIG = 1000.0


def pade_coeffs(theta, n):
    c = [math.factorial(2*n - j)*math.factorial(n) /
         (math.factorial(2*n)*math.factorial(j)*math.factorial(n - j))
         for j in range(n + 1)]
    num = np.array([c[j]*((-theta) ** j) for j in range(n + 1)][::-1], dtype=float)
    den = np.array([c[j]*(theta ** j) for j in range(n + 1)][::-1], dtype=float)
    return num, den


def build_plant(k, tau, theta, integrating):
    pnum, pden = pade_coeffs(theta, N_PADE)
    Gn = k * pnum
    if integrating:
        Gd = np.polymul(pden, np.array([1.0, 0.0]))
    elif tau == 0:
        Gd = pden
    else:
        Gd = np.polymul(pden, np.array([tau, 1.0]))
    return Gn, Gd


def serial_pid(kc, ti, td, tf=TAU_F):
    return kc*np.array([ti*td, ti+td, 1.0]), np.array([ti*tf, ti, 0.0])


def serial_pi(kc, ti):
    return kc*np.array([ti, 1.0]), np.array([ti, 0.0])


def closed_loop(Gn, Gd, Kn, Kd):
    Ln = np.polymul(Gn, Kn)
    Ld = np.polymul(Gd, Kd)
    Sd = np.polyadd(Ld, Ln)
    return Ld, Sd, np.polymul(Gn, Ld), np.polymul(Gd, Sd), Ln, Sd


def step_iae(num, den, t_end=150.0, npts=30000):
    t = np.linspace(0, t_end, npts)
    _, y = signal.step((num, den), T=t)
    v = np.trapz(np.abs(y), t)
    return float(v)


def hinf_peak(num, den):
    w = np.logspace(-4, 4, 20000)
    jw = 1j*w
    return float(np.abs(np.polyval(num, jw)/np.polyval(den, jw)).max())


def evaluate(Gn, Gd, kc, ti, td, IAE0_dy, IAE0_du):
    if td == 0:
        Kn, Kd = serial_pi(kc, ti)
    else:
        Kn, Kd = serial_pid(kc, ti, td, TAU_F)
    Sn, Sd, GSn, GSd, Tn, Td = closed_loop(Gn, Gd, Kn, Kd)
    poles = np.roots(Sd)
    if np.any(~np.isfinite(poles)) or np.any(np.real(poles) >= -1e-6):   # stability: finite poles, Re < 0
        return None
    i_dy = step_iae(Sn, Sd)
    i_du = step_iae(GSn, GSd)
    if not (np.isfinite(i_dy) and np.isfinite(i_du)):
        return None
    J = 0.5*(i_dy/IAE0_dy + i_du/IAE0_du)
    Ms = hinf_peak(Sn, Sd)
    MT = hinf_peak(Tn, Td)
    return i_dy, i_du, J, Ms, MT, max(Ms, MT)


def safe_J(Gn, Gd, kc, ti, td, IAE0_dy, IAE0_du):
    r = evaluate(Gn, Gd, kc, ti, td, IAE0_dy, IAE0_du)
    if r is None or not np.isfinite(r[2]):
        return BIG
    return r[2]


def safe_MST(Gn, Gd, kc, ti, td, IAE0_dy, IAE0_du):
    r = evaluate(Gn, Gd, kc, ti, td, IAE0_dy, IAE0_du)
    if r is None or not np.isfinite(r[5]):
        return BIG
    return r[5]


def optimise(Gn, Gd, IAE0_dy, IAE0_du, starts, td_bounds=(0.0, 50.0)):
    best = None
    for x0 in starts:
        obj = lambda p: safe_J(Gn, Gd, p[0], p[1], p[2], IAE0_dy, IAE0_du)
        con = lambda p: 1.59 - safe_MST(Gn, Gd, p[0], p[1], p[2], IAE0_dy, IAE0_du)
        try:
            res = minimize(obj, x0, method='SLSQP',
                           bounds=[(1e-4, 50), (1e-4, 50), td_bounds],
                           constraints={'type': 'ineq', 'fun': con},
                           options={'ftol': 1e-8, 'maxiter': 300})
        except Exception:
            continue
        r = evaluate(Gn, Gd, res.x[0], res.x[1], res.x[2], IAE0_dy, IAE0_du)
        if r is None:                          # unstable or non-finite
            continue
        _, _, J, Ms, MT, MST = r
        if MST > 1.59 + 1e-3:                  # robustness constraint Ms,MT <= 1.59
            continue
        if not res.success:                    # optimizer did not converge
            continue
        if best is None or J < best[0]:
            best = (J, res.x)
    return best


processes = [
    # name, k, tau, theta, integrating, IAE0_dy, IAE0_du, paper opt, paper iSIMC, starts
    ('e^-s',        1.0, 0.0,  1.0, False, 1.61, 1.61,   (0.20, 0.32, 0.00, 1.00), (0.21, 0.33, 0.00, 1.00),
     [(0.20, 0.32, 0.0), (0.10, 0.20, 0.0), (0.30, 0.50, 0.0)]),
    ('e^-s/(s+1)',  1.0, 1.0,  1.0, False, 2.07, 2.02,   (0.42, 0.61, 0.61, 0.74), (0.62, 1.00, 0.33, 0.79),
     [(0.50, 0.60, 0.60), (0.40, 0.70, 0.50)]),
    ('e^-s/(8s+1)', 1.0, 8.0,  1.0, False, 2.17, 1.13,   (4.34, 2.63, 0.49, 0.81), (4.92, 6.50, 0.33, 1.00),
     [(4.00, 3.00, 0.50), (5.00, 6.00, 0.30), (3.00, 2.00, 0.50), (6.00, 8.00, 0.30)]),
    ('e^-s/s',      1.0, None, 1.0, True,  2.17, 15.10,  (0.53, 3.18, 0.51, 0.89), (0.59, 6.81, 0.33, 1.09),
     [(0.50, 3.00, 0.50), (0.60, 6.00, 0.30), (0.40, 4.00, 0.60)]),
]

print('Table 3 reproduction (ours vs paper) — tau_f=%.2f, Pade order %d, multi-start' % (TAU_F, N_PADE))

for name, k, tau, theta, integ, IAE0_dy, IAE0_du, opt_p, isimc_p, starts in processes:
    Gn, Gd = build_plant(k, tau, theta, integ)

    td_b = (0.0, 0.0) if name == 'e^-s' else (0.0, 50.0)   # pure delay -> PI only
    best = optimise(Gn, Gd, IAE0_dy, IAE0_du, starts, td_bounds=td_b)
    if best is None:
        kc_o, ti_o, td_o, J_o = np.nan, np.nan, np.nan, np.nan
    else:
        J_o, x = best
        kc_o, ti_o, td_o = x
        r = evaluate(Gn, Gd, kc_o, ti_o, td_o, IAE0_dy, IAE0_du)
        J_o = r[2] if r else np.nan

    J_i = evaluate(Gn, Gd, isimc_p[0], isimc_p[1], isimc_p[2], IAE0_dy, IAE0_du)
    J_i = J_i[2] if J_i else np.nan

    print()
    print('  %-12s' % name)
    print('    Optimal PID : ours  kc=%.3f  ti=%.3f  td=%.3f  J=%.3f' % (kc_o, ti_o, td_o, J_o))
    print('                  paper kc=%.2f  ti=%.2f  td=%.2f  J=%.2f' % opt_p)
    print('    iSIMC       : ours J=%.3f' % J_i)
    print('                  paper kc=%.2f  ti=%.2f  td=%.2f  J=%.2f' % isimc_p)
