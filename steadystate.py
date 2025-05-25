# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:43:32 2025

@author: ئسەر
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thermalbuilding import TC
import dm4bem

controller = True
neglect_air_glass_capacity = False
imposed_time_step = True
Δt = 498    # s, imposed time step


# MODEL
# =====
# Thermal circuit

# by default TC['G']['q11'] = 0, i.e. Kp -> 0, no controller (free-floating)
if controller:
    TC['G']['q9'] = 1000    # Kp -> ∞, almost perfect controller
    TC['G']['q20'] = 0
#if neglect_air_glass_capacity:
   # TC['C']['θ11'] = TC['C']['θ5'] = 0
    # or
   # TC['C'].update({'θ11': 0, 'θ5': 0})


# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)


bss = np.zeros(21)        # temperature sources b for steady state
bss[[0,1,7,8,11,12,18,19]] = 10      # outdoor temperature
bss[[9,20]] = 20          # indoor set-point temperature

fss = np.zeros(12)         # flow-rate sources f for steady state

print(bss)
A = TC['A']
G = TC['G']
diag_G = pd.DataFrame(np.diag(G), index=G.index, columns=G.index)

θss = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
#θss= np.full((12, 1), 10)
print(f'θss = {np.around(θss, 2)} °C')


bss = np.zeros(21)        # temperature sources b for steady state

fss = np.zeros(12)         # flow-rate sources f for steady state
fss[[5,11]] = 0,1000

θssQ = np.linalg.inv(A.T @ diag_G @ A) @ (A.T @ diag_G @ bss + fss)
print(f'θssQ = {np.around(θssQ, 2)} °C')


bT = np.array([10, 10, 10, 10,20,10,10,10,10,20])     # [To, To, To, Tisp]
fQ = np.array([0, 0, 0, 0,0,0])         # [Φo, Φi, Qa, Φa]
uss = np.hstack([bT, fQ])           # input vector for state space
print(f'uss = {uss}')


inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)
yss = (-Cs @ inv_As @ Bs + Ds) @ uss

yss = float(yss.values[0])
print(f'yss = {yss:.2f} °C')


print(f'Error between DAE and state-space: {abs(θss[5] - yss):.2e} °C')


bT = np.array([0, 0, 0, 0,0,0,0,0,0,0])         # [To, To, To, Tisp]
fQ = np.array([0, 0, 0, 0,0,1000])      # [Φo, Φi, Qa, Φa]
uss = np.hstack([bT, fQ])

inv_As = pd.DataFrame(np.linalg.inv(As),
                      columns=As.index, index=As.index)
yssQ = (-Cs @ inv_As @ Bs + Ds) @ uss

yssQ = float(yssQ.values[0])
print(f'yssQ = {yssQ:.2f} °C')


print(f'Error between DAE and state-space: {abs(θssQ[5] - yssQ):.2e} °C')


# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As


# time step
Δtmax = 2 * min(-1. / λ)    # max time step for stability of Euler explicit
dm4bem.print_rounded_time('Δtmax', Δtmax)

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(Δtmax)
dm4bem.print_rounded_time('dt', dt)


if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")
# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)


# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}s")

To = 10 * np.ones(n)        # outdoor temperature
Ti_sp = 20 * np.ones(n)     # indoor temperature set point
Φa = 0 * np.ones(n)         # solar radiation absorbed by the glass
Qa = Φo = Φi = Φa           # auxiliary heat sources and solar radiation

data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': 0, 'Φi': 0, 'Qa': 0, 'Φa': 0,'Φo': 0, 'Φi': 0, 'Qa': 0, 'Φa': 0}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)


# Initial conditions
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 0.0                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])
    

# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T


# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)

ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {Δtmax:.0f} s')
plt.show()



print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {float(θss[5]):.4f} °C')
print(f'- state-space model: {float(yss):.4f} °C')
print("Available columns in y_exp:", y_exp.columns.tolist())
print(f'- steady-state response to step input: \
{y_exp["θ5"].tail(1).values[0]:.4f} °C')

# Create input_data_set
# ---------------------
# time vector
n = int(np.floor(duration / dt))    # number of time steps

# Create a DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start="2000-01-01 00:00:00",
                           periods=n, freq=f"{int(dt)}s")
# Create input_data_set
To = 0 * np.ones(n)         # outdoor temperature
Ti_sp =  20 * np.ones(n)     # indoor temperature set point
Φa = 0 * np.ones(n)         # solar radiation absorbed by the glass
Φo = Φi = Φa                # solar radiation
Qa = 1000 * np.ones(n)      # auxiliary heat sources
data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa,'Φo': Φo, 'Φi':Φi , 'Qa': Qa, 'Φa': Φa}
input_data_set = pd.DataFrame(data, index=time)

# Get inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)


# Initial conditions
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])
    

# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T


# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)
ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {Δtmax:.0f} s')
plt.show()


print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {float(θssQ[5]):.4f} °C')
print(f'- state-space model: {float(yssQ):.4f} °C')
print(f'- steady-state response to step input: \
{y_exp["θ5"].tail(1).values[0]:.4f} °C')


#In cell [2], consider:

controller = False
neglect_air_glass_capacity = False
imposed_time_step = False


# Now, in cell [2], consider:

controller = False
neglect_air_glass_capacity = False
imposed_time_step = True
Δt = 498    # s, imposed time step


controller = False
neglect_air_glass_capacity = True
imposed_time_step = False

controller = True
neglect_air_glass_capacity = True
imposed_time_step = False


controller = True
neglect_air_glass_capacity = False
imposed_time_step = False


if controller:
    TC['G']['q9'] = 1e3        # Kp -> ∞, almost perfect controller
    TC['G']['q20'] = 1e3

if controller:
    TC['G']['q11'] = 1e5        # Kp -> ∞, almost perfect controller
    TC['G']['q20'] = 1e5


    