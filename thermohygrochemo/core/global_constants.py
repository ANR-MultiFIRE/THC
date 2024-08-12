# Global Constants
from fenics import *

T_0_K         = Constant(273.15)          # Conversion to Kelvin Scale
T_cr          = Constant(647.3)           # Critical Temperature of Water
T_ref_1       = Constant(293.15)          # Reference Temperature 1 (Ambient)
T_ref_2       = Constant(298.15)          # Reference Temperature 2 (Ambient)
Pg_ref        = Constant(101325)          # Reference Gas Pressure (Ambient)
Cp_l          = Constant(4181)            # Heat Capacity of Liquid Water
Cp_v          = Constant(1805)            # Heat Capacity of Water Vapour
Cp_a          = Constant(1005.7)          # Heat Capacity of Dry Air
sigma_SB      = Constant(5.67e-8)         # Stefan-Boltzmann Constant
R             = Constant(8.31441)         # Ideal Gas Constant
M_v           = Constant(0.01801528)      # Molar Mass of Water Vapour
M_a           = Constant(0.0289645)       # Molar Mass of Dry Air
D_v_0         = Constant(2.58e-5)         # Initial Mass Diffusivity