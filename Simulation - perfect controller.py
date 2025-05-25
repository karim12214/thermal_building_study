import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ctrl
import time

import dm4bem

controller = True
Kp1 = 1e3    # W/°C, controller gain
Kp2 = 1e3

neglect_air_capacity = False
neglect_glass_capacity = False

explicit_Euler = True

imposed_time_step = False
Δt = 3600    # s, imposed time step 

# MODEL
# =====
# Thermal circuits
TC = dm4bem.file2TC('./model/TC.csv', name='', auto_number=False)

# by default  0 # Kp -> 0, no controller (free-floating
if controller:
    TC['G']['q9'] = 1e3     # G9 = Kp, conductance of edge q9
    TC['G']['q20'] = 1e3                      # Kp -> ∞, almost perfect controller
if neglect_air_capacity:
    TC['C']['θ1'] = 0
    TC['C']['θ3'] = 0        #
if neglect_glass_capacity:
    TC['C']['θ7'] = 0
    TC['C']['θ9'] = 0       # 

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
dm4bem.print_TC(TC)

λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As
dtmax = 2 * min(-1. / λ)    # max time step for Euler explicit stability
dt = dm4bem.round_time(dtmax)

if imposed_time_step:
    dt = Δt

dm4bem.print_rounded_time('dt', dt)


# INPUT DATA SET
# ==============
input_data_set = pd.read_csv('./model/input_data_set.csv',
                             index_col=0,
                             parse_dates=True,
                             dayfirst=True)
print(input_data_set)

# Resample Input DataSet
input_data_set = input_data_set.resample(
    str(dt) + 's').interpolate(method='linear')
input_data_set.head()

# Input vector in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)
u.head()

# Initial conditions
θ0 = 20.0                   # °C, initial temperatures
θ = pd.DataFrame(index=u.index)
θ[As.columns] = θ0          # fill θ with initial valeus θ0

I = np.eye(As.shape[0])     # identity matrix

if explicit_Euler:
    for k in range(u.shape[0] - 1):
        θ.iloc[k + 1] = (I + dt * As) @ θ.iloc[k] + dt * Bs @ u.iloc[k]
else:
    for k in range(u.shape[0] - 1):
        θ.iloc[k + 1] = np.linalg.inv(
            I - dt * As) @ (θ.iloc[k] + dt * Bs @ u.iloc[k])
        
# outputs
y = (Cs @ θ.T + Ds @  u.T).T

# Controller gains (assumed equal here)
TC['G']['q9'] = Kp1      # room 1
TC['G']['q20'] = Kp2      # room 2

# Surface areas (assumed toy house with 9 m² and 20 m² respectively)
S1 = S2 = 9     # m², Room  and 2


# HVAC heat flux density (W/m²) for Room 1 only
q_HVAC1 = Kp1 * (u['q9'] - y['θ5']) / S1

print(q_HVAC1)

# Combine data for Room 1 only
data = pd.DataFrame({
    'To': input_data_set['To'],     # outdoor temperature
    'θi1': y['θ5'],                 # Room 1 indoor temperature
    'Etot': input_data_set['Etot'], # total solar radiation
    'q_HVAC1': q_HVAC1              # HVAC power density for Room 1
})

# Plotting for Room 1 only
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# Temperature plot
data[['To', 'θi1']].plot(ax=axs[0],
                         xticks=[],
                         ylabel='Temperature, $θ$ / °C')
axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor,1}$'],
              loc='upper right')

# HVAC power plot
data[['Etot', 'q_HVAC1']].plot(ax=axs[1],
                               ylabel='Heat rate, $q$ / (W·m⁻²)')
axs[1].set(xlabel='Time')
axs[1].legend(['$E_{total}$', '$q_{HVAC,1}$'],
              loc='upper right')

plt.tight_layout()
plt.show()

# Save simulation result
data.to_csv("simulation_output_theta5_only.csv")
