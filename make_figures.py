# -*- coding: utf-8 -*-
"""
Generate two figures for Week 8:
  fig1_step.png  — step responses (ours vs paper) for the four benchmark processes
  fig2_tauf.png  — J (and Ms) vs derivative-filter tau_f for e^-s/(s+1)
Colorblind-safe Okabe-Ito palette; solid = ours, dashed = paper (secondary encoding).
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

TAU_F = 0.03
N_PADE = 12

# Okabe-Ito colorblind-safe
BLUE = '#0072B2'
ORANGE = '#E69F00'
GRID = '#cccccc'


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


def serial_pid(kc, ti, td, tf):
    return kc*np.array([ti*td, ti+td, 1.0]), np.array([ti*tf, ti, 0.0])


def serial_pi(kc, ti):
    return kc*np.array([ti, 1.0]), np.array([ti, 0.0])


def closed_loop(Gn, Gd, Kn, Kd):
    Ln = np.polymul(Gn, Kn)
    Ld = np.polymul(Gd, Kd)
    Sd = np.polyadd(Ld, Ln)
    return Ld, Sd, np.polymul(Gn, Ld), np.polymul(Gd, Sd), Ln, Sd


def step_resp(Gn, Gd, kc, ti, td, tf, t_end, npts=5000):
    if td == 0:
        Kn, Kd = serial_pi(kc, ti)
    else:
        Kn, Kd = serial_pid(kc, ti, td, tf)
    Ln = np.polymul(Gn, Kn)
    Ld = np.polymul(Gd, Kd)
    Sd = np.polyadd(Ld, Ln)              # T = Ln / Sd  (setpoint response)
    t = np.linspace(0, t_end, npts)
    _, y = signal.step((Ln, Sd), T=t)
    return t, y


def step_iae(num, den, t_end=150.0, npts=30000):
    t = np.linspace(0, t_end, npts)
    _, y = signal.step((num, den), T=t)
    return float(np.trapz(np.abs(y), t))


def hinf_peak(num, den):
    w = np.logspace(-4, 4, 20000)
    jw = 1j*w
    return float(np.abs(np.polyval(num, jw)/np.polyval(den, jw)).max())


def evaluate(Gn, Gd, kc, ti, td, tf, IAE0_dy, IAE0_du):
    Kn, Kd = (serial_pi(kc, ti) if td == 0 else serial_pid(kc, ti, td, tf))
    Sn, Sd, GSn, GSd, Tn, Td = closed_loop(Gn, Gd, Kn, Kd)
    i_dy = step_iae(Sn, Sd)
    i_du = step_iae(GSn, GSd)
    J = 0.5*(i_dy/IAE0_dy + i_du/IAE0_du)
    Ms = hinf_peak(Sn, Sd)
    MT = hinf_peak(Tn, Td)
    return J, Ms, MT, max(Ms, MT)


# process data: name, k, tau, theta, integrating, ours(kc,ti,td), paper(kc,ti,td), t_end
procs = [
    ('e⁻ˢ',          1.0, 0.0, 1.0, False, (0.198, 0.319, 0.000), (0.20, 0.32, 0.00), 25.0),
    ('e⁻ˢ/(s+1)',   1.0, 1.0, 1.0, False, (0.417, 0.620, 0.620), (0.42, 0.61, 0.61), 30.0),
    ('e⁻ˢ/(8s+1)',  1.0, 8.0, 1.0, False, (4.274, 2.806, 0.491), (4.34, 2.63, 0.49), 30.0),
    ('e⁻ˢ/s',       1.0, None, 1.0, True, (0.533, 3.691, 0.486), (0.53, 3.18, 0.51), 60.0),
]

# Figure 1: step responses
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
for ax, (name, k, tau, theta, integ, ours, paper, t_end) in zip(axes.flat, procs):
    Gn, Gd = build_plant(k, tau, theta, integ)
    t1, y1 = step_resp(Gn, Gd, *ours, TAU_F, t_end)
    t2, y2 = step_resp(Gn, Gd, *paper, TAU_F, t_end)
    ax.plot(t1, y1, color=BLUE, lw=1.8, label='ours (SLSQP)')
    ax.plot(t2, y2, color=ORANGE, lw=1.4, ls='--', label='paper')
    ax.set_title('G(s) = ' + name, fontsize=11)
    ax.set_xlabel('time t', fontsize=9)
    ax.set_ylabel('output y(t)', fontsize=9)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.tick_params(labelsize=8)

axes[0, 0].legend(fontsize=9, frameon=False, loc='upper left')
fig.suptitle('Step responses: our reproduced optimal PID vs the paper (Table 3)', fontsize=12, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('fig1_step.png', dpi=160, bbox_inches='tight')
print('saved fig1_step.png')

# Figure 2: tau_f sensitivity (e^-s/(s+1))
Gn, Gd = build_plant(1.0, 1.0, 1.0, False)
IAE0_dy, IAE0_du = 2.07, 2.02
tf_grid = np.concatenate([np.linspace(0.005, 0.02, 4), np.linspace(0.03, 0.20, 12)])
Js, Msts = [], []
for tf in tf_grid:
    J, Ms, MT, MST = evaluate(Gn, Gd, 0.42, 0.61, 0.61, tf, IAE0_dy, IAE0_du)
    Js.append(J)
    Msts.append(MST)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
ax1.plot(tf_grid, Js, color=BLUE, lw=1.8, marker='o', ms=3)
ax1.axhline(0.74, color=ORANGE, lw=1.2, ls='--', label='paper J = 0.74')
ax1.axvline(TAU_F, color='#555555', lw=0.8, ls=':')
ax1.set_xlabel('derivative filter τf', fontsize=9)
ax1.set_ylabel('J', fontsize=9)
ax1.set_title('J vs τf', fontsize=10)
ax1.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax1.legend(fontsize=8, frameon=False)
ax1.tick_params(labelsize=8)

ax2.plot(tf_grid, Msts, color=BLUE, lw=1.8, marker='o', ms=3)
ax2.axhline(1.59, color=ORANGE, lw=1.2, ls='--', label='MST target 1.59')
ax2.axvline(TAU_F, color='#555555', lw=0.8, ls=':')
ax2.set_xlabel('derivative filter τf', fontsize=9)
ax2.set_ylabel('MST = max(Ms, MT)', fontsize=9)
ax2.set_title('MST vs τf', fontsize=10)
ax2.grid(True, color=GRID, lw=0.5, alpha=0.6)
ax2.legend(fontsize=8, frameon=False)
ax2.tick_params(labelsize=8)

fig2.suptitle('Derivative-filter sensitivity for e⁻ˢ/(s+1) (paper optimal PID kc=0.42, τi=τd=0.61)',
              fontsize=11, y=0.99)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig('fig2_tauf.png', dpi=160, bbox_inches='tight')
print('saved fig2_tauf.png')
