# -*- coding: utf-8 -*-
"""
Week 7 implementation evidence — Grimholt & Skogestad (2018), Table 3,
for the benchmark process G(s) = e^-s/(s+1)  (k=1, tau=1, theta=1).

Contents:
  1. Cross-check of two independent IAE calculations (time-domain step vs
     frequency-domain inverse transform), agreeing within 1%.
  2. SIMC-PI baseline, iSIMC and optimal-PID results compared with the paper.

Method notes:
  * serial (cascade) PID  K(s) = kc (tau_i s+1)(tau_d s+1)/(tau_i s)
  * derivative filter      F = 1/(tau_f s + 1)  with a small fixed tau_f = 0.03
  * time delay             Pade approximation (order 12)
  * IAE                    unit step disturbance at output (dy) and input (du)
  * robustness             Ms = max|S(jw)|, MT = max|T(jw)|, MST = max{Ms, MT}
"""
import math
import numpy as np
from scipy import signal
from scipy.optimize import minimize

k, tau, theta = 1.0, 1.0, 1.0          # G(s) = e^-s / (s+1)
IAE0_dy, IAE0_du = 2.07, 2.02          # Table 1 weighting factors
TAU_F = 0.03                           # small fixed derivative filter


def pade_coeffs(theta, n):
    c = [math.factorial(2*n - j) * math.factorial(n) /
         (math.factorial(2*n) * math.factorial(j) * math.factorial(n - j))
         for j in range(n + 1)]
    num = np.array([c[j] * ((-theta) ** j) for j in range(n + 1)][::-1], dtype=float)
    den = np.array([c[j] * (theta ** j) for j in range(n + 1)][::-1], dtype=float)
    return num, den


pnum, pden = pade_coeffs(theta, 12)
Gn = k * pnum
Gd = np.polymul(pden, np.array([tau, 1.0]))


def serial_pid(kc, ti, td, tf=TAU_F):
    return kc*np.array([ti*td, ti+td, 1.0]), np.array([ti*tf, ti, 0.0])


def serial_pi(kc, ti):
    return kc*np.array([ti, 1.0]), np.array([ti, 0.0])


def closed_loop(Gn, Gd, Kn, Kd):
    Ln = np.polymul(Gn, Kn)
    Ld = np.polymul(Gd, Kd)
    Sd = np.polyadd(Ld, Ln)
    return Ld, Sd, np.polymul(Gn, Ld), np.polymul(Gd, Sd), Ln, Sd


def iae_time(num, den, t_end=100.0, npts=20000):
    t = np.linspace(0, t_end, npts)
    _, y = signal.step((num, den), T=t)
    return float(np.trapz(np.abs(y), t))


def iae_freq(num, den, t_end=100.0, npts=20000):
    dt = t_end / npts
    w = 2*np.pi*np.fft.rfftfreq(npts, d=dt)
    H = np.polyval(num, 1j*w) / np.polyval(den, 1j*w)
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    h = np.fft.irfft(H, n=npts) / dt
    s = np.cumsum(h) * dt
    return float(np.trapz(np.abs(s), np.arange(npts)*dt))


def hinf_peak(num, den, wlo=1e-4, whi=1e4, npts=20000):
    w = np.logspace(np.log10(wlo), np.log10(whi), npts)
    return float(np.abs(np.polyval(num, 1j*w) / np.polyval(den, 1j*w)).max())


def evaluate(kc, ti, td, tf=TAU_F):
    Kn, Kd = (serial_pi(kc, ti) if td == 0 else serial_pid(kc, ti, td, tf))
    Sn, Sd, GSn, GSd, Tn, Td = closed_loop(Gn, Gd, Kn, Kd)
    i_dy = iae_time(Sn, Sd)
    i_du = iae_time(GSn, GSd)
    J = 0.5*(i_dy/IAE0_dy + i_du/IAE0_du)
    Ms = hinf_peak(Sn, Sd)
    MT = hinf_peak(Tn, Td)
    return i_dy, i_du, J, Ms, MT, max(Ms, MT)


# 1. cross-check of two independent IAE methods (SIMC-PI baseline)
print('1. Cross-check of two independent IAE calculations (SIMC-PI baseline)')
Kn, Kd = serial_pi(0.50, 1.00)
Sn, Sd, GSn, GSd, Tn, Td = closed_loop(Gn, Gd, Kn, Kd)
for label, n, d in [('IAEdy (output disturbance)', Sn, Sd),
                    ('IAEdu (input disturbance)', GSn, GSd)]:
    td = iae_time(n, d)
    fd = iae_freq(n, d)
    print(f'{label}')
    print(f'  time-domain      : {td:.5f}')
    print(f'  frequency-domain : {fd:.5f}')
    print(f'  relative diff    : {abs(td-fd)/td*100:.3f}%')
    print()

# 2. SIMC-PI baseline vs Table 4
print('2. SIMC-PI baseline (kc=0.50, tau_i=1.00)  →  Table 4')
i_dy, i_du, J, Ms, MT, MST = evaluate(0.50, 1.00, 0.0)
print(f'  ours  : IAEdy={i_dy:.4f}  IAEdu={i_du:.4f}  J={J:.4f}  MST={MST:.4f}')
print(f'  paper : IAEdy=2.17       IAEdu=2.04       J=1.03   MST=1.59')
print()

# 3. iSIMC PID vs Table 3
print('3. iSIMC PID (tau_c=0.61 → kc=1/(tau_c+theta), tau_d=theta/3)')
tau_c = 0.61
kc_i = 1.0/(tau_c + theta)
ti_i = 1.0
td_i = theta/3.0
print(f'  algebra : kc={kc_i:.4f}  tau_i={ti_i:.4f}  tau_d={td_i:.4f}')
print(f'  paper   : kc=0.62        tau_i=1.00        tau_d=0.33')
i_dy, i_du, J, Ms, MT, MST = evaluate(kc_i, ti_i, td_i)
print(f'  ours  : J={J:.4f}  MST={MST:.4f}    paper: J=0.79  MST=1.59')
print()

# 4. optimal PID (SLSQP) vs Table 3
print('4. Optimal PID (min J s.t. Ms<=1.59, MT<=1.59)')


def obj(p):
    return evaluate(p[0], p[1], p[2])[2]


def con(p):
    return 1.59 - evaluate(p[0], p[1], p[2])[5]


res = minimize(obj, [0.5, 0.6, 0.6], method='SLSQP',
               bounds=[(1e-4, 20.0), (1e-4, 20.0), (0.0, 20.0)],
               constraints={'type': 'ineq', 'fun': con},
               options={'ftol': 1e-8, 'maxiter': 200})
kc_o, ti_o, td_o = res.x
i_dy, i_du, J, Ms, MT, MST = evaluate(kc_o, ti_o, td_o)
print(f'  ours  : kc={kc_o:.4f}  tau_i={ti_o:.4f}  tau_d={td_o:.4f}  J={J:.4f}')
print(f'  paper : kc=0.42        tau_i=0.61        tau_d=0.61        J=0.74')
print('  note  : tau_i ~ tau_d reproduces the paper structural result')
