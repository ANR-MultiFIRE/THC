from fenics import *
from thermohygrochemo.core.global_constants import *
from thermohygrochemo.core.constant_constitutive_laws import *

# Young age model properties
# Power's Volume Model
rho_c = 3150                        # Density of dry cement
rho_sf = 2200                       # Density of silica fume
rho_l_pm = 1000                     # Density of liquid water at RT

gravel = 960 + 89                   # Gravel content [kg/m3]
sand = 740                          # Sand content [kg/m3]
cement = 400                        # Cement content [kg/m3]
silica_fume = 0                     # Silica fume content [kg/m3]
water = 177                         # Water content [kg/m3]
aggr = gravel + sand                # Agregates content [kg/m3]
wc_ratio = water / cement           # Water/Cement Ratio
sc_ratio = silica_fume / cement     # Silica Fume/cement Ratio
wb_ratio = water / (cement + silica_fume)     # Silica Fume/cement Ratio
Omega = (cement / rho_c) + (water / rho_l_pm) + (silica_fume / rho_sf)  # Cement paste vol.



# Hydration process data
xsi_inf = (1.031 * wc_ratio) / (0.194 + wc_ratio)  # Max. Hydration Degree
m_hyd_inf = 0.228 * cement * xsi_inf               # Total chemically bound water
E_a = 4500 * R                                     # Hydration activation energy

a_B = 5                             # AKA. AHR  in Cast3M code
A_i = 2                             # AKA. AFF1 in Cast3M code
A_P = 17.5                          # AKA. AFF2 in Cast3M code
Gamma_P = 0.13                      # AKA. AFF3 in Cast3M code
zeta = 18                           # AKA. AFF4 in Cast3M code

# Final porosity, phi^cp_inf calculation
p = wc_ratio / (wc_ratio + rho_l_pm / rho_c + (rho_l_pm / rho_sf) * (sc_ratio))
k = 1 / (1 + (rho_c / rho_sf) * sc_ratio)
phi_cp_inf = p - k * (0.52 - 0.69 * sc_ratio) * (1 - p) * xsi_inf
a_phi_val = k * (0.52 - 0.69 * sc_ratio) * (1 - p) * xsi_inf

H_dehyd = 58.04e6                    # Enthalpy of Dehydration


def ppo(x):
    '''
    Positive Part Operator.

    Parameters:
    ------------
    x: float, array
        Value to take the positive part from
    '''
    return (abs(x) + x) / 2


def A(G):
    '''
    Concrete's Chemical Affinity, A [-]

    Parameters
    ----------
    G [-] : array_like, UFL function
    An array with nodal degree of reaction or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    A_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the concrete's chemical affinity.

    Notes
    -----
    The original Equation in the papers has the power as 4.0 instead of 2.5.
    '''
    A_vals = (A_i + (A_P - A_i) * sin(pi / 2 * (1 - ppo((Gamma_P - G) / Gamma_P)))) \
            / (1 + zeta * ppo((G - Gamma_P) / (1 - Gamma_P))**4) \
            - (A_P / (1 + zeta)) * ppo((G - Gamma_P) / (1 - Gamma_P))
    return A_vals


def B(Pg, Pc, T):
    '''
    RH effect on the degree of reaction, B [-]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    B_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the RH effect on the concrete's degree of reaction.

    Notes
    -----
    No notes.
    '''

    RH = Pv(Pg, Pc, T) / Pvps(T)
    B_vals = ((1 + (a_B - a_B * RH)**4)**(-1))
    return B_vals


def dGammadt(Pg, Pc, T, G):
    r'''
    Time derivative of degree of reaction, \Gamma [-]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    VIT_HYDR_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the concrete's degree of reaction.

    Notes
    -----
    No notes.
    '''
    HR_EFF = B(Pg, Pc, T)
    TK_EFF = exp(-E_a / (R * T))
    CH_AFF = A(G)
    VIT_HYDR_vals = CH_AFF * HR_EFF * TK_EFF
    return VIT_HYDR_vals


def F(T_k):
    '''
    Dehydration rate, F [kg/s] - AKA. FIRE
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    F_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the dehydration rate.
    Notes
    -----
    No notes.
    '''
    T = T_k - T_0_K
    cond = ((1 + sin(pi / 2 * (1 - 2 * exp(-0.008 * (T - 105))))) / 2)
    f_d_vals = conditional(lt(T, 105), 0, cond)
    return f_d_vals


def dFdT(T_k):
    '''
    Dehydration rate, F [kg/s] - AKA. DFIRE
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    F_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the dehydration rate as a function
    of the temperature.
    Notes
    -----
    No notes.
    '''
    T = T_k - T_0_K
    cond = (cos(pi / 2 * (1 - 2 * exp(-0.008 * (T - 105)))) *
            pi * 0.008 / 2 * exp(-0.008 * (T - 105)))
    df_ddT_vals = conditional(lt(T, 105), 0, cond)
    return df_ddT_vals


def Gamma_tilde(Pg, Pc, T, G):
    r'''
    Effective hydration degree, $\tilde{Gamma}
    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    G [-] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    Gamma_tilde_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the effective hydration degree.
    '''
    Gamma_tilde_vals = (1 - F(T)) * G
    return Gamma_tilde_vals


def phi(Pg, Pc, T, G):
    '''
    Solid's porosity, phi [-]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    phi_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the solid's porosity.

    Notes
    -----
    phi_As = 0.0 as the average porosity of the aggregates is NEGLECTED
    '''
    phi_As = 0.0
    phi_vals = phi_cp_inf * Omega  + a_phi_val * (1 - Gamma_tilde(Pg, Pc, T, G)) * Omega + phi_As * (1 - Omega)
    return phi_vals


def a_phi(Pg, Pc, T, G):
    dphidT_vals = a_phi_val
    return dphidT_vals



# Solid's Density, $\rho_s$ and its Derivative with respect to Temperature.
def rho_s(T):
    '''
    Solid's density, rho_s [kg/m^3]

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rho_s_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the solid's density.

    Notes
    -----
    No notes.
    '''
    rho_s_vals = 2366.0
    return rho_s_vals


def drho_sdT(T):
    '''
    Derivative of solid's density w/ respect to T, drho_sdT [kg/(m^3 . K)]

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_sdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the solid's density with respect to
    T.

    Notes
    -----
    No notes.
    '''
    A_s = 0
    drho_sdT_vals = A_s
    return drho_sdT_vals



# Intrinsic Permeabiliy, $K$.
def K(Pg, Pc, T, G):
    '''
    Intrinsic permeability, K [m^2]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    K_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the intrisic permeability.

    Notes
    -----
    No notes.
    '''
    K_0 = 7.5e-20
    A_k = 1.25
    K_vals = K_0 * 10**(A_k * (1 - Gamma_tilde(Pg, Pc, T, G)))
    return K_vals

K_G = K_L = K

def k_rl(Pg, Pc, T, G):
    '''
    Liquid water relative permeability, k_rl [-]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    k_rl_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the liquid water relative permeability.

    Notes
    -----
    No notes
    '''
    A = 0.6
    k_rl_vals = S_l(Pg, Pc, T, G)**0.5 * (1 - (1 - S_l(Pg, Pc, T, G)**(1 / A))**A)**2
    return k_rl_vals


# Gas Relative Permeability, $k_{rg}$.
def k_rg(Pg, Pc, T, G):
    '''
    Gas relative permeability, k_rg [-]

    Parameters
    ----------
    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    k_rg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the gas relative permeability.

    Notes
    -----
    No notes.
    '''
    A = 0.6
    S_l_T_cr = S_l(Pg, Pc, T_cr - 0.5, G)
    S_l_real = S_l(Pg, Pc, T, G)
    S_l_vals = conditional(le(S_l_real, S_l_T_cr), S_l_T_cr, S_l_real)
    k_rg_vals = (1 - S_l_vals)**0.5 * (1 - S_l_vals**(1 / A))**(2 * A)
    return k_rg_vals


# Mass Diffusivity , $D_{eff}$.
def D_va(Pg, T):
    '''
    Diffusivity of vapor in air, D_va [m^2/s]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    D_va_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the diffusivity of vapor in air.

    Notes
    -----
    Other diffusivity laws may be added.
    '''
    D_v_0 = 2.58e-5
    A_v = 1.667
    D_va_vals = D_v_0 * (T / T_0_K)**A_v * (Pg_ref / Pg)
    return D_va_vals


def f_s(Pg, Pc, T, G):
    '''
    Turtuosity and the available pore space for gas, f_s [-]

    Parameters
    ----------
    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    f_s_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the turtuosity and available pore space for gas.

    Notes
    -----
    It is used to derive the effective diffusivity, D_eff.
    '''
    tau = phi(T, Pc, T, G)**(1 / 3) * (1 - S_l(Pg, Pc, T, G))**(7 / 3)
    pore_space_gas = phi(T, Pc, T, G) * (1 - S_l(Pg, Pc, T, G))
    f_s_vals = tau * pore_space_gas
    return f_s_vals

def D_eff(Pg, Pc, T, G):
    '''
    Effective diffusivity of vapor in air, D_eff [m^2/s]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    D_eff_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the effective diffusivity of vapor in air.

    Notes
    -----
    Other diffusivity laws may be added.
    '''
    D_eff_vals = f_s(Pg, Pc, T, G) * D_va(Pg, T)
    return D_eff_vals


# Thermal Conductivity, $\lambda$.
def lambda_eff(Pg, Pc, T, G):
    '''
    Effective thermal conductivity of moist concrete,
    lambda_eff [W/(m . K)]

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function
    for defining variational formulations for FEniCS FEM library.

    Returns
    -------
    lambda_eff_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the effective thermal conductivity of moist
    concrete.

    Notes
    -----
    No notes.
    '''
    T_C = T - T_0_K
    lambda_inf = 1.0 - (0.136 - 0.0057 * (T_C / 100)) * (T_C / 100)
    lambda_eff = lambda_inf
    return lambda_eff



# Specific Heat of Solid Concrete, $C_ps$
def Cp_s(T):
    '''
    Specific heat of solid concrete, Cp_s [J/(kg . K)]

    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    Cp_s_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the solid concrete specific heat.

    Notes
    -----
    ATTENTION: THE LAW IS DEFINED IN CELSIUS, T [K] IS THUS CONVERTED TO
    T_C [C].
    '''
    u = 0.0
    T_C = T - 273.15
    Cp_s_20 = 915
    if u == 0.0:
        Cp_s_100_115 = 915
    elif u == 1.5:
        Cp_s_100_115 = 1450
    elif u == 3.0:
        Cp_s_100_115 = 2030
    elif u == 6.0:
        Cp_s_100_115 = 3675
    elif u == 9.0:
        Cp_s_100_115 = 5100
    Cp_s_115_200 = Cp_s_100_115 + (1000 - Cp_s_100_115) / 85 * (T_C - 115)
    Cp_s_200_400 = 1000 + (T_C - 200) / 2
    Cp_s_400_inf = 1105
    Cp_s_vals = conditional(le(T_C, 100),
                            Cp_s_20,
                            conditional(le(T_C, 115),
                                        Cp_s_100_115,
                                        conditional(le(T_C, 200),
                                                    Cp_s_115_200,
                                                    conditional(le(T_C, 400),
                                                                Cp_s_200_400,
                                                                Cp_s_400_inf))))
    return Cp_s_vals


def rhoCp(Pg, Pc, T, G):
    '''
    Effective thermal capacity of concrete, rhoCp [J/(K . m^3)]

    Parameters
    ----------
    P_g [Pa] : array_like, UFL function
    An array with nodal gas pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    P_c [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rho_Cp_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the effective thermal capacity of concrete.

    Notes
    -----
    Cp_g might vary in other implementions.
    '''
    Cp_g = (rho_g(Pg, Pc, T) * Cp_a + rho_v(Pg, Pc, T) * (Cp_v - Cp_a))
    rhoCp_s = (1 - phi(Pg, Pc, T, G)) * rho_s(T) * Cp_s(T)
    rhoCp_l = phi(Pg, Pc, T, G) * S_l(Pg, Pc, T, G) * rho_l(T) * Cp_l
    rhoCp_g = phi(Pg, Pc, T, G) * (1 - S_l(Pg, Pc, T, G)) * Cp_g
    rho_Cp_vals = rhoCp_s + rhoCp_l + rhoCp_g
    return rho_Cp_vals




# Saturation as a function of $T$ and $P_c$, $S_l(T, P_c)$ and its derivative
def gamma_w(T):
    '''
    Surface Tension of Water  [N/m = kg/s^2]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    gamma_w_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the surface tension of water.

    Notes
    -----
    No notes.
    '''

    gamma_w_val_1 = (235.8 * (1 - T / T_cr)**(1.256) * (1 - 0.625 * (1 - T / T_cr))) * 1e-3
    gamma_w_val_2 = 0
    gamma_w_vals = conditional(lt(T, T_cr), gamma_w_val_1, gamma_w_val_2)
    return gamma_w_vals


def S_l(Pg, Pc, T, G):
    '''
    Saturation of liquid water as a function of temperature and
    capillary pressure, S_l [-]

    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    S_l_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation with liquid water as a function of
    temperature and pressure.

    Notes
    -----
    No notes.
    '''
    a = (-113 * wc_ratio + 82.9) * 1e6
    b = 2.11
    c_Gamma = 1.5
    Gamma_i = 0.1
    z = 0.05
    gamma_w_20 = 0.072739843
    gamma_effect =((gamma_w(T) + z * gamma_w_20) / ((1 + z) * gamma_w_20))**-1
    hyd_eff = ((Gamma_tilde(Pg, Pc, T, G) + Gamma_i) / (1 + Gamma_i))**-c_Gamma
    S_l_vals_unbound = conditional(lt(T, T_cr), ((Pc / a * hyd_eff * gamma_effect)**(b / (b - 1)) + 1)**(-1 / b), 0)
    S_l_vals = conditional(le(S_l_vals_unbound, 1e-3),
                           1e-3,
                           conditional(le(S_l_vals_unbound, 0.999), 
                                       S_l_vals_unbound, 
                                       0.999))
    return S_l_vals


def dS_ldT(Pg, Pc, T, G):
    '''
    Derviative of saturation as a function of temperature and
    capillary pressure, dS_ldT [1/K]

    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dS_ldT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the saturation with liquid water
    with respect to temperature.

    Notes
    -----
    No notes.
    '''
    delta = 1e-4
    dT = delta * T
    dS_ldT_vals = ((S_l(Pg, Pc, T + dT / 2, G) - S_l(Pg, Pc, T - dT / 2, G)) / (dT))
    return dS_ldT_vals


def dS_ldPc(Pg, Pc, T, G):
    '''
    Derviative of saturation as a function of temperature and
    capillary pressure, dS_ldPc [1/Pa]

    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dS_ldPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the saturation with liquid water
    with respect to capillary pressure.

    Notes
    -----
    WARNING: There is a necessary conditional here to enforce that dS_ldPc does
    not diverge.
    '''
    delta = 1e-8
    dP = delta * Pc
    dS_ldPc_vals = ((S_l(Pg, Pc + dP / 2, T, G) - S_l(Pg, Pc - dP / 2, T, G)) / (dP))
    return dS_ldPc_vals


def dS_ldGamma(Pg, Pc, T, G):
    '''
    Derviative of saturation as a function of temperature and
    capillary pressure, dS_ldPc [1/Pa]
    Parameters
    ----------
    Pc [Pa] : array_like, UFL function
    An array with nodal capillary pressures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dS_ldGamma_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the saturation with liquid water
    with respect to the hydration degree.

    Notes
    -----
    No notes.
    '''
    delta = 1e-4
    dG = delta * (G + 1e-6)
    dS_ldGamma_vals = ((S_l(Pg, Pc, T, G + dG / 2) - S_l(Pg, Pc, T, G - dG / 2)) / (dG))
    return dS_ldGamma_vals
