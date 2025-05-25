# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:26:36 2025

@author: ئسەر
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dm4bem



l = 3               # m length of the cubic room
L = 6
Sg = l**2           # m² surface area of the glass wall
Sc = Si = L*l    # m² surface area of concrete & insulation of the 5 walls

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])
concrete = {'Conductivity': 0.8,          # W/(m·K)
            'Density': 1750.0,              # kg/m³
            'Specific heat': 900,           # J/(kg⋅K)
            'Width': 0.4,                   # m
            'Surface': 5.55}            # m²

insulation = {'Conductivity': 0.023,        # W/(m·K)
              'Density': 30.0,              # kg/m³
              'Specific heat': 1400,        # J/(kg⋅K)
              'Width': 0.02,                # m
              'Surface': 5.55}          # m²

glass = {'Conductivity': 0.02,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.02,                     # m
         'Surface': 6.48}                   # m²
door = {'Conductivity': 0.79,
        'Density': 2710,
        'Specific heat': 900,
        'Width': 0.03,
        'Surface': 2.35*1.15}
wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass,
                               'Door':door},
                              orient='index')
wall

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant

h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)
h



# conduction
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
pd.DataFrame(G_cd, columns=['Conductance'])

# convection
Gw = h * wall['Surface'].iloc[0]     # wall
Gg = h * wall['Surface'].iloc[2]     # glass
Gis = h * wall['Surface'].iloc[1]
Gd = h * wall['Surface'].iloc[3]

# view factor wall-glass
Fwg = glass['Surface'] / concrete['Surface']



T_int = 273.15 + np.array([0, 40])
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For 0°C < (T/K - 273.15)°C < 40°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + np.array([10, 30])
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For 10°C < (T/K - 273.15)°C < 30°C, 4σT³/[W/(m²·K)] ∈ {coeff}')

T_int = 273.15 + 20
coeff = np.round((4 * σ * T_int**3), 1)
print(f'For (T/K - 273.15)°C = 20°C, 4σT³ = {4 * σ * T_int**3:.1f} W/(m²·K)')


# long wave radiation
Tm = 20 + 273   # K, mean temp for radiative exchange

GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['Layer_in']
GLW12 = 4 * σ * Tm**3 * Fwg * wall['Surface']['Layer_in']
GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['Glass']

GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2)

Va1 = l**3                   # m³, volume of air
Va2 = l**3
ACH = 1                     # 1/h, air changes per hour
Va_dot1 = ACH / 3600 * Va1   # m³/s, air infiltration
Va_dot2 = ACH / 3600 * Va2

Gv1 = air['Density'] * air['Specific heat'] * Va_dot1
Gv2 = air['Density'] * air['Specific heat'] * Va_dot2



# P-controler gain
# Kp = 1e4            # almost perfect controller Kp -> ∞
# Kp = 1e-3           # no controller Kp -> 0
Kp1 = 0
Kp2 = 0





# glass: convection outdoor & conduction
Ggs = float(1 / (1 / Gg.loc['h', 'out'] + 6.48 + 1 / Gg.loc['h','in']))
Ggsi = float (1 / (1 / Gg.loc['h', 'in'] + 6.48 + 1 / Gg.loc['h','in']))
#wall: convection outdoor & conduction 
Gws = float(1/ (1 / Gw.loc['h', 'out'] + 1 / (G_cd['Layer_out'])))
# isolant: conduction and indoor convection
Giss = Gws = float(1/ (1 / Gw.loc['h', 'in'] + 1 / (G_cd['Layer_in'])))
#door: convection outdoor & conduction & indoor convection
Gds = float(1/ (1 / Gd.loc['h','out']  +0.79*2.35*1.15+ 1 / Gd.loc['h','in']))



C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
pd.DataFrame(C, columns=['Capacity'])



C['Air'] = air['Density'] * air['Specific heat'] * Va1
pd.DataFrame(C, columns=['Capacity'])



# temperature nodes
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8','θ9','θ10','θ11']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12','q13','q14','q15','q16','q17','q18','q19','q20']

# temperature nodes
nθ = 12      # number of temperature nodes
θ = [f'θ{i}' for i in range(12)]

# flow-rate branches
nq = 21     # number of flow branches
q = [f'q{i}' for i in range(21)]

A = np.zeros([21, 12])       # n° of branches X n° of nodes
A[0, 5] = 1                 # branch 0: -> node 0
A[1, 0] = 1    # branch 1: node 0 -> node 1
A[2, 0], A[2, 1] = -1, 1    # branch 2: node 1 -> node 2
A[3, 1], A[3, 2] = -1, 1    # branch 3: node 2 -> node 3
A[4, 2], A[4, 3] = -1, 1    # branch 4: node 3 -> node 4
A[5, 3], A[5, 4] = -1, 1    # branch 5: node 4 -> node 5
A[6, 4], A[6, 5] = -1, 1    # branch 6: node 4 -> node 6
A[7, 5] = 1    # branch 7: node 5 -> node 6
A[8, 5] = 1                 # branch 8: -> node 7
A[9, 5] = 1    # branch 9: node 5 -> node 7
A[10, 5], A[10,11] = -1, -1                # branch 10: -> node 6
A[11, 11] = 1                # branch 11: -> node 6
A[12,6] = 1
A[13,6], A[13,7] = -1, 1
A[14,7], A[14,8] = -1, 1
A[15,8], A[15,9] =  -1, 1
A[16,9], A[16,10] = -1, 1
A[17,10], A[17,11] =1, 1
A[18,11] = 1
A[19,11] = 1
A[20,11] = 1

pd.DataFrame(A, index=q, columns=θ)



G = np.array(np.hstack(
    [Gds,
     Gw['out'],
     2*G_cd['Layer_out'],
     2*G_cd['Layer_out'],
     2*G_cd['Layer_in'],
     2*G_cd['Layer_in'],
     Gis['in'],
     Ggs,
     Gv1,
     Kp1,
     Ggsi,
     Gds,
     Gw['out'],
     2*G_cd['Layer_out'],
     2*G_cd['Layer_out'],
     2*G_cd['Layer_in'],
     2*G_cd['Layer_in'],
     Gis['in'],
     Ggs,
     Gv2,
     Kp2]))

# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
pd.DataFrame(G, index=q)


neglect_air_glass = False

if neglect_air_glass:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, C['Air'],
                  0, C['Layer_out'],0,C['Layer_in'],0,C['Air']])
else:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                  0, 0,0,0,0,0])

# pd.set_option("display.precision", 3)
pd.DataFrame(C, index=θ)



b = pd.Series(['To', 'To', 0, 0, 0, 0, 0, 'To', 'To', 'Ti_sp', 0, 'To','To',0,0,0,0,0,'To','To','Ti_sp'],
              index=q)

f = pd.Series(['Φo', 0, 0, 0, 'Φi', 'Qa', 'Φo', 0,0,0,'Φi','Qa'],
              index=θ)



y = np.zeros(12)         # nodes
y[[5]] = 1    # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)




# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}



# TC = dm4bem.file2TC('./toy_model/TC.csv', name='', auto_number=False)



# TC['G']['q11'] = 1e3  # Kp -> ∞, almost perfect controller
TC['G']['q9'] = 0      # Kp -> 0, no controller (free-floating)
TC['G']['q20'] = 0



[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
us


