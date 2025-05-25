import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import dm4bem

controller = True
neglect_air_glass_capacity = True # Always true in this model
imposed_time_step = False
Δt = 1800    # s, imposed time step

# MODEL
# =====
# Thermal circuit
TC = dm4bem.file2TC('./model/TC.csv', name='', auto_number=False)

# Kp -> 0, no controller (free-floating)

if controller:
    TC['G']['q6'] = 1e3        # Kp -> ∞, almost perfect controller
    TC['G']['q14'] = 1e3 

# State-space
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

# Eigenvalues analysis
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As
# print(f'λ = {λ}')

# time step
Δtmax = 2 * min(-1 / λ)    # max time step for stability of Euler explicit
dm4bem.print_rounded_time('Δtmax', Δtmax)

imposed_time_step = True

if imposed_time_step:
    dt = Δt
else:
    dt = dm4bem.round_time(Δtmax)

if dt < 10:
    raise ValueError("Time step is too small. Stopping the script.")

dm4bem.print_rounded_time('dt', dt)

# settling time
t_settle = 4 * max(-1 / λ)
dm4bem.print_rounded_time('t_settle', t_settle)

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
dm4bem.print_rounded_time('duration', duration)

# Define the start and end dates for the simulation
start_date = '04-01 12:00:00'
end_date = '04-05 12:00:00'

start_date = '2025-' + start_date
end_date = '2025-' + end_date
print(f'{start_date} \tstart date')
print(f'{end_date} \tend date')

# Lyon weather data

filename = './weather_data/FRA_Lyon.074810_IWEC.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=None)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
To = weather['temp_air']
del data
weather.index = weather.index.map(lambda t: t.replace(year=2025))
weather = weather.loc[start_date:end_date]

# Temperature sources are refred from the weather data
To = weather['temp_air']

# Solar radiation absorbed by the glass is neglected
# Solar radiation values Φo1, Φi1, Φi2, Φo2 ae now to be calculated from the weather data and from modeling part

# radiative properties 

τ_gSW = 0.30  # from modeling part

wall_out = pd.read_csv('./bldg/walls_out.csv')
w1 = wall_out[wall_out['ID'] == 'w0'] #concrete
w2 = wall_out[wall_out['ID'] == 'w1'] #insulation
wall_in = pd.read_csv('./bldg/walls_in.csv')
w3 = wall_in[wall_in['ID'] == 'w0'] #concrete
surface_orientation = {'slope': w1['β'].values[0],
                       'azimuth': w1['γ'].values[0],
                       'latitude': 45}

rad_surf = dm4bem.sol_rad_tilt_surf(
    weather, surface_orientation, w1['albedo'].values[0])

Etot = rad_surf.sum(axis=1)

# solar radiation absorbed by the outdoor surface of the wall
Φo1 = w1['α1'].values[0] * w1['Area'].values[0] * Etot  
Φo2 = w2['α1'].values[0] * w2['Area'].values[0] * Etot 


# solar radiation absorbed by the indoor surface of the wall
S_glass = 12 # m² surface area of the glass wall, see assignment1
Φi1 = τ_gSW * w2['α0'].values[0] * S_glass * Etot    
Φi2 = τ_gSW * w3['α0'].values[0] * S_glass * Etot    
Qa = 0

# Indoor air temperature set-point
Ti_day, Ti_night = 20, 16

Ti_sp = pd.Series(
    [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
    index=To.index)

# No auxiliary heat sources in this model

# Input data set

input_data_set = pd.DataFrame({'To': To, 'Ti_sp': Ti_sp, 'Φo1': Φo1, 'Φi1': Φi1, 'Φi2': Φi2, 'Φo2': Φo2, 'Etot': Etot, 'Qa': Qa })

input_data_set.to_csv('./model/input_data_set.csv')



