from thermohygrochemo.core.thc_model import *
from thermohygrochemo.core.constant_constitutive_laws import *
from thermohygrochemo.materials.materials_constitutive_laws import *
import sys
sys.path.append('../../')

# ======================== Directory for Output Files ========================

dir_output = "/ya_kanema_2D_axi_manual_7_days"
dir_backup = '/backup'

# Redirect stdout to log file
dir_output_full = thc_model_dir + '/results' + dir_output

if not path.exists(dir_output_full):
    makedirs(dir_output_full)
if not path.exists(dir_output_full + dir_backup):
    makedirs(dir_output_full + dir_backup)


sys.stdout = Logger(thc_model_dir + '/results' + dir_output
                    + f'/{dir_output[1:]}.log')

# =========================== Time Discretization =============================
t = 0                         # Initial time
minute = 60                   # Number of seconds in 1 minute
hour = 60 * minute            # Number of seconds in 1 hour
day = 24 * hour               # Number of seconds in 1 day
year = 365 * day

t_curing = 3 * day
t_aging = 7 * day

T_ramp_0 = 20                                          # Initial T
T_plateau = 300                                        # Plateau T
T_rate = 1 / minute                                    # Heating rate in °C/min
t_ramp = (T_plateau - T_ramp_0) / T_rate               # Time of heating ramp
t_plateau = 1 * hour                                   # Time of fixed T
t_heating_begin = 240                                  # Heating time begin
t_heating = 0.5 * hour                                 # Heating time begin
t_heating_cr = (t_ramp + t_plateau) - (t_heating_begin + t_heating)  # Heating time critical
t_cooling = 2                                          # Cooling time

t_total = (t_curing + t_aging
           + t_heating_begin + t_heating_cr
           + t_cooling)           # Total time in seconds


dt_curing = 1200
dt_aging = 7200
dt_heating_begin = 1
dt_heating = 1
dt_heating_critical = 10
dt_cooling = 100

tau_curing = 11               # Number of iterations for dt transition
tau_aging = 11                # Number of iterations for dt transition
tau_heating_begin = 3         # Number of iterations for dt transition
tau_heating = 11              # Number of iterations for dt transition
tau_heating_critical = 11     # Number of iterations for dt transition

rate_curing = 1               # Rate of logistic dt transition
rate_aging = 2                # Rate of logistic dt transition
rate_heating_begin = 1.5      # Rate of logistic dt transition
rate_heating = 1              # Rate of logistic dt transition
rate_heating_critical = 1     # Rate of logistic dt transition


stages_cfg = {'stage_1': {'name': 'curing',
                          'case': 'curing',
                          'duration': t_curing,
                          'stepsizeselector': 'manual',
                          'dt': dt_curing,
                          'tau': tau_curing,
                          'rate': rate_curing},
              'stage_2': {'name': 'aging',
                          'case': 'aging',
                          'duration': t_aging,
                          'stepsizeselector': 'manual',
                          'dt': dt_aging,
                          'tau': tau_aging,
                          'rate': rate_aging},
              'stage_3': {'name': 'heating_begin',
                          'case': 'heating',
                          'duration': t_heating_begin,
                          'stepsizeselector': 'manual',
                          'dt': dt_heating_begin,
                          'tau': tau_heating_begin,
                          'rate': rate_heating_begin},
              'stage_4': {'name': 'heating_1',
                          'case': 'heating',
                          'duration': t_heating,
                          'stepsizeselector': 'manual',
                          'dt': dt_heating,
                          'tau': tau_heating,
                          'rate': rate_heating},
              'stage_5': {'name': 'heating_critical',
                          'case': 'heating',
                          'duration': t_heating_cr,
                          'stepsizeselector': 'manual',
                          'dt': dt_heating_critical,
                          'tau': tau_heating_critical,
                          'rate': rate_heating_critical},
              'stage_6': {'name': 'cooling',
                          'case': 'cooling',
                          'duration': t_cooling,
                          'stepsizeselector': 'manual-set',
                          'dt': dt_cooling}}

freq_out = 5                # Frequency of Writing on the Output Files

# =========================== Space Discretization ============================
# Mesh
mesh_name = '2D_Cylinder'
mesh_input = Mesh()
with XDMFFile(f'{thermo_dir}/meshes/{mesh_name}.xdmf') as infile:
    infile.read(mesh_input)

mvc = MeshValueCollection("size_t", mesh_input, 2)
with XDMFFile(f'{thermo_dir}/meshes/{mesh_name}_MeshFunction.xdmf') as infile:
    infile.read(mvc, "Boundary")
boundaries_input = MeshFunction('size_t', mesh_input, mvc)

# ============================ Initial Conditions =============================
RH_0 = 0.9825                        # Initial relative humidity
T_0 = 273.15 + 20                    # Initial temperature
Pg_0 = 101325                        # Initial gas pressure (ambient pressure)
Pc_0 = Pc_Keq(RH_0, Pg_0, T_0)       # Initial capillary pressure
G_0 = 0                              # Initial Hydration Degree

# ============================ Boundary Conditions ============================
h_g = 0.018 / 100                    # Mass transfer coefficient
h_T_curing = 8.3                     # Thermal film coeficient during curing
h_T_cooling = 10.0                   # Thermal film coeficient during cooling
epsilon = 0.5                        # Surface emissivity
T_inf = T_0                          # Temperature at the far field
Pg_inf = Pg_0                        # Gas Pressure at the far field

# Densities at the far field sorrounding gas
RH_inf_aging = 0.8     # Relative Humidity far field sorrounding gas
Pc_inf_aging = Pc_Keq(RH_inf_aging, Pg_inf, T_inf)
rho_v_inf_aging = float(rho_v(Pg_0, Pc_inf_aging, T_0))

RH_inf_heating = 0.50  # Relative Humidity far field sorrounding gas
Pc_inf_heating = Pc_Keq(RH_inf_heating, Pg_inf, T_inf)
rho_v_inf_heating = float(rho_v(Pg_0, Pc_inf_heating, T_0))
rho_v_inf_cooling = rho_v_inf_heating


def rho_v_inf_heating_trans(t):
    '''
    Smooth transition between the density of vapor
    at the far field from the aging stage to
    the heating stage.

    Parameters:
    ------------
    t: time, float
        Time in seconds

    Returns:
    ---------
    rho_v_inf_val: 
        Density at the far field.
    '''

    t_0 = t_curing + t_aging
    theta = t - t_0
    rate = 0.5e-2
    theta_tr = 1800
    theta_0 = theta_tr / 2
    delta_rho_v_inf = rho_v_inf_heating - rho_v_inf_aging
    rho_v_inf_val = (rho_v_inf_aging + delta_rho_v_inf
                     / (1 + np.exp(- rate * (theta - theta_0))))
    return rho_v_inf_val


def T_heating(t):
    '''
    Temperature evolution during the hating stage.

    Parameters:
    ------------
    t: time, float
        Time in seconds

    Returns:
    ---------
    T_val: 
        Temperature value considered.
    '''
    t_0 = t_curing + t_aging
    theta = t - t_0
    delay = 50
    if theta <= delay:
        T_val = T_ramp_0 + 273.15
    elif theta <= t_ramp and theta > delay:
        T_val = (theta - delay) * T_rate + T_ramp_0 + 273.15
    else:
        T_val = T_plateau + 273.15
    return T_val


def T_cooling(t):
    '''
    Temperature evolution during the cooling stage.

    Parameters:
    ------------
    t: time, float
        Time in seconds

    Returns:
    ---------
    T_val: 
        Temperature value considered.
    '''
    theta = t - (t_curing + t_aging
                 + t_heating_begin + t_heating_cr)
    T_val = T_plateau + 273.15 - theta * T_rate
    return T_val


# Dictionary specifying  Boundary Conditions to be used
# Pg_BC options:
# 1. 'dirichlet'
# 2. 'robin'
# 3. 'impermeable'
# 4. 'symmetry'
Pg_BC_cfg = {'curing': {'top': {'condition': 'impermeable',
                                'marker': 10},
                        'bottom': {'condition': 'impermeable',
                                   'marker': 11},
                        'external': {'condition': 'impermeable',
                                     'marker': 12},
                        'internal': {'condition': 'symmetry',
                                     'marker': 13}},
             'aging': {'top': {'condition': 'dirichlet',
                               'Pg_fix_val': Pg_0,
                               'marker': 10},
                       'bottom': {'condition': 'dirichlet',
                                  'Pg_fix_val': Pg_0,
                                  'marker': 11},
                       'external': {'condition': 'dirichlet',
                                    'Pg_fix_val': Pg_0,
                                    'marker': 12},
                       'internal': {'condition': 'symmetry',
                                    'marker': 13}},
             'heating': {'top': {'condition': 'dirichlet',
                                 'Pg_fix_val': Pg_0,
                                 'marker': 10},
                         'bottom': {'condition': 'dirichlet',
                                    'Pg_fix_val': Pg_0,
                                    'marker': 11},
                         'external': {'condition': 'dirichlet',
                                      'Pg_fix_val': Pg_0,
                                      'marker': 12},
                         'internal': {'condition': 'symmetry',
                                      'marker': 13}},
             'cooling': {'top': {'condition': 'dirichlet',
                                 'Pg_fix_val': Pg_0,
                                 'marker': 10},
                         'bottom': {'condition': 'dirichlet',
                                    'Pg_fix_val': Pg_0,
                                    'marker': 11},
                         'external': {'condition': 'dirichlet',
                                      'Pg_fix_val': Pg_0,
                                      'marker': 12},
                         'internal': {'condition': 'symmetry',
                                      'marker': 13}}}

# Pc_BC options:
# 1. 'dirichlet'
# 2. 'robin'
# 3. 'impermeable'
# 4. 'symmetry'
Pc_BC_cfg = {'curing': {'top': {'condition': 'impermeable',
                                'marker': 10},
                        'bottom': {'condition': 'impermeable',
                                   'marker': 11},
                        'external': {'condition': 'impermeable',
                                     'marker': 12},
                        'internal': {'condition': 'impermeable',
                                     'marker': 13}},
             'aging': {'top': {'condition': 'robin',
                               'rho_v_inf_val': rho_v_inf_aging,
                               'h_g_val': h_g,
                               'marker': 10},
                       'bottom': {'condition': 'robin',
                                  'rho_v_inf_val': rho_v_inf_aging,
                                  'h_g_val': h_g,
                                  'marker': 11},
                       'external': {'condition': 'robin',
                                    'rho_v_inf_val': rho_v_inf_aging,
                                    'h_g_val': h_g,
                                    'marker': 12},
                       'internal': {'condition': 'symmetry',
                                    'marker': 13}},
             'heating': {'top': {'condition': 'robin',
                                 'rho_v_inf_val': rho_v_inf_heating_trans,
                                 'h_g_val': h_g,
                                 'marker': 10},
                         'bottom': {'condition': 'robin',
                                    'rho_v_inf_val': rho_v_inf_heating_trans,
                                    'h_g_val': h_g,
                                    'marker': 11},
                         'external': {'condition': 'robin',
                                      'rho_v_inf_val': rho_v_inf_heating_trans,
                                      'h_g_val': h_g,
                                      'marker': 12},
                         'internal': {'condition': 'symmetry',
                                      'marker': 13}},
             'cooling': {'top': {'condition': 'robin',
                                 'rho_v_inf_val': rho_v_inf_cooling,
                                 'h_g_val': h_g,
                                 'marker': 10},
                         'bottom': {'condition': 'robin',
                                    'rho_v_inf_val': rho_v_inf_cooling,
                                    'h_g_val': h_g,
                                    'marker': 11},
                         'external': {'condition': 'robin',
                                      'rho_v_inf_val': rho_v_inf_cooling,
                                      'h_g_val': h_g,
                                      'marker': 12},
                         'internal': {'condition': 'symmetry',
                                      'marker': 13}}}


# T_BC options:
# 1. 'dirichlet'
# 2. 'robin-conv'
# 3. 'robin-rad'
# 4. 'robin-conv-rad'
# 5. 'adiabatic'
# 6. 'symmetry'

T_BC_cfg = {'curing': {'top': {'condition': 'robin-conv',
                               'T_inf_val': T_inf,
                               'h_T_val': h_T_curing,
                               'marker': 10},
                       'bottom': {'condition': 'robin-conv',
                                  'T_inf_val': T_inf,
                                  'h_T_val': h_T_curing,
                                  'marker': 11},
                       'external': {'condition': 'robin-conv',
                                    'T_inf_val': T_inf,
                                    'h_T_val': h_T_curing,
                                    'marker': 12},
                       'internal': {'condition': 'symmetry',
                                    'marker': 13}},
            'aging': {'top': {'condition': 'robin-conv',
                              'T_inf_val': T_inf,
                              'h_T_val': h_T_curing,
                              'marker': 10},
                      'bottom': {'condition': 'robin-conv',
                                 'T_inf_val': T_inf,
                                 'h_T_val': h_T_curing,
                                 'marker': 11},
                      'external': {'condition': 'robin-conv',
                                   'T_inf_val': T_inf,
                                   'h_T_val': h_T_curing,
                                   'marker': 12},
                      'internal': {'condition': 'symmetry',
                                   'marker': 13}},
            'heating': {'top': {'condition': 'dirichlet',
                                'T_fix_val': T_heating,
                                'marker': 10},
                        'bottom': {'condition': 'dirichlet',
                                   'T_fix_val': T_heating,
                                   'marker': 11},
                        'external': {'condition': 'dirichlet',
                                     'T_fix_val': T_heating,
                                     'marker': 12},
                        'internal': {'condition': 'symmetry',
                                     'marker': 13}},
            'cooling': {'top': {'condition': 'robin-conv-rad',
                                'T_inf_val': T_cooling,
                                'h_T_val': h_T_cooling,
                                'epsilon_val': epsilon,
                                'marker': 10},
                        'bottom': {'condition': 'robin-conv-rad',
                                   'T_inf_val': T_cooling,
                                   'h_T_val': h_T_cooling,
                                   'epsilon_val': epsilon,
                                   'marker': 11},
                        'external': {'condition': 'robin-conv-rad',
                                     'T_inf_val': T_cooling,
                                     'h_T_val': h_T_cooling,
                                     'epsilon_val': epsilon,
                                     'marker': 12},
                        'internal': {'condition': 'symmetry',
                                     'marker': 13}}}


# ========================== Printing Initial Values ==========================
print('|--- Initial Conditions:')
print(f'   |--- Pg_0  = {round(Pg_0 * 1e-6, 2)} MPa')
print(f'   |--- Pc_0  = {round(Pc_Keq(RH_0, Pg_0, T_0) * 1e-6, 2)} MPa')
print(f'   |--- T_0   = {T_0} K')
print(f'   |--- G_0   = {G_0} [-]')
S_l_0 = float(S_l(Pg_0, Pc_Keq(RH_0, Pg_0, T_0), T_0, G_0))
print(f'   |--- S_l_0 = {round(S_l_0, 2)} [-]')


# ============================ Running Simulation =============================
TH_MODEL = thc_model_axisymmetric(mesh_input, boundaries_input, t_total,
                                  Pg_0, Pc_0, T_0, G_0,
                                  Pg_BC_cfg, Pc_BC_cfg, T_BC_cfg,
                                  dir_output, dir_backup, __file__,
                                  freq_out=freq_out,
                                  stages_cfg=stages_cfg)
TH_MODEL.run()

print('\nThat\'s all folks!')
