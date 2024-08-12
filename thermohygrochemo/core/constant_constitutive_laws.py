from fenics import *
from thermohygrochemo.core.global_constants import *
from numpy import log as np_log


def Pc_Keq(RH, Pg, T):
    '''
    Kelvin Equation for setting initial capillary pressure, Pc_0 [Pa]
    Parameters
    ----------
    RH [-] : float, array_like
    A float or an array with relative humidity.

    T [K] : float, array_like
    A float or an array with temperatures.

    Returns
    -------
    Pc_Keq_vals :  float, array_like
    A flot or array object that represents the initial capillary pressure to be
    interpolated on the FEM space.

    Notes
    -----
    Can be considere as a helper function.
    '''
    np_R = float(R)
    np_M_v = float(M_v)
    Pc_Keq_vals   = Pg - Pvps(T) + ((-1.) * (np_R * T *  rho_l(T) * (np_log(RH)) / np_M_v))   
    return Pc_Keq_vals


def Pvps(T):
    '''
    Saturation vapor pressure, Pvps [Pa]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    Pvps_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the saturation vapor pressure.

    Notes
    -----
    No notes.
    '''
    C_1 = -5800.2206
    C_2 = 1.3914993
    C_3 = -4.8640239e-2
    C_4 = 4.1764768e-5
    C_5 = -1.4452093e-8
    C_6 = 6.5459673
    Pvps_vals = exp(C_1 / T + C_2 + C_3 * T + C_4 * T**2 + C_5 * T**3 +
                    C_6 * ln(T))
    return Pvps_vals


def dPvpsdT(T):
    '''
    Derivative of the saturation vapor pressure with respect to temperature,
    dPvpsdT [Pa/K]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPvpsdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of the saturation vapor pressure with
    resoect to temperature.

    Notes
    -----
    No notes.
    '''
    C_1 = -5800.2206
    C_3 = -4.8640239e-2
    C_4 = 4.1764768e-5
    C_5 = -1.4452093e-8
    C_6 = 6.5459673
    dPvpsdT_vals = Pvps(T) * (- C_1 / T**2 + C_3 + 2 * C_4 * T +
                              3 * C_5 * T**2 + C_6 / T)
    return dPvpsdT_vals


def Pv(Pg, Pc, T):
    '''
    Water vapor pressure, Pv [Pa]
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
    Pv_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the water vapor pressure.

    Notes
    -----
    It is assumed an ideal gas behavior.
    '''
    Pv_vals = Pvps(T) * exp(- M_v / (R * T * rho_l(T)) * (Pc - Pg + Pvps(T)))
    return Pv_vals


def rho_v(Pg, Pc, T):
    '''
    Water vapor density, rho_v [kg/m^3]
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
    rho_v_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the water vapor density.

    Notes
    -----
    It is assumed an ideal gas behavior.
    '''
    rho_v_vals = M_v / (R * T) * Pv(Pg, Pc, T)
    return rho_v_vals


def Pa(Pg, Pc, T):
    '''
    Dry air pressure, Pa [Pa]
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
    Pa_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the dry air pressure.

    Notes
    -----
    If the gas pressure is lower then the vapor pressure, an air pressure of
    100 Pa is assumed.
    '''
    cond = Pg - Pv(Pg, Pc, T)
    Pa_vals = conditional(ge(cond, 0), Pg - Pv(Pg, Pc, T), 0.1)
    return Pa_vals


def rho_a(Pg, Pc, T):
    '''
    Dry air density, rho_a [kg/m^3]
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
    rho_a_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the dry air density.

    Notes
    -----
    It is assumed an ideal gas behavior.
    '''
    cond = Pg - Pv(Pg, Pc, T)
    rho_a_vals = conditional(ge(cond, 0), (M_a / (R * T) * Pa(Pg, Pc, T)), 1e-3)
    return rho_a_vals


def rho_g(Pg, Pc, T):
    '''
    Gas density, rho_g [kg/m^3]
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
    rho_g_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the gas density.

    Notes
    -----
    Dalton's law.
    '''
    FGPGA0 = (M_a / (R * T))
    FGPGW0 = (M_v / (R * T))
    CHPGSC0 = Pg
    CHPGWSC0 = Pv(Pg, Pc, T)
    CHPGSCM1 = 1 / Pg
    RHOG0    = (FGPGA0 * CHPGSC0) + ((FGPGW0 - FGPGA0) * CHPGWSC0) ;
    rho_g_vals = RHOG0
    return rho_g_vals


def rho_l(T_k):
    '''
    Liquid water density, rho_l [kg/m^3]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    rho_l_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the liquid water density.

    Notes
    -----
    After the critical temperature the liquid density is set constant.
    '''
    T = T_k - T_0_K
    T_cr_C = T_cr - T_0_K
    a_0 = 4.8863e-7
    a_1 = -1.6528e-9
    a_2 = 1.8621e-12
    a_3 = 2.4266e-13
    a_4 = -1.5996e-15
    a_5 = 3.3703e-18
    b_0 = 1021.3e0
    b_1 = -7.7377e-1
    b_2 = 8.7696e-3
    b_3 = -9.2118e-5
    b_4 = 3.3534e-7
    b_5 = -4.4034e-10
    p_l_1 = 1.0e7
    p_l_ref = 2.0e7
    cond_1 = (b_0 + b_1 * T + b_2 * T**2 + b_3 * T**3 + b_4 * T**4 +
              b_5 * T**5) + ((p_l_1 - p_l_ref) * (a_0 + a_1 * T + a_2 * T**2 +
                                                  a_3 * T**3 + a_4 * T**4 +
                                                  a_5 * T**5))
    cond_2 = ((b_0 + b_1 * T_cr_C + b_2 * T_cr_C**2 + b_3 * T_cr_C**3 +
               b_4 * T_cr_C**4 + b_5 * T_cr_C**5) +
              ((p_l_1 - p_l_ref) * (a_0 + a_1 * T_cr_C + a_2 * T_cr_C**2 +
                                    a_3 * T_cr_C**3 + a_4 * T_cr_C**4 +
                                    a_5 * T_cr_C**5)))
    rho_l_vals = conditional(lt(T, T_cr_C), cond_1, cond_2)
    return rho_l_vals


def drho_ldT(T_k):
    '''
    Derivative of liquid water density, drho_ldT [kg/(m^3 K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_ldT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of liquid water density with respect to
    temperature.

    Notes
    -----
    After the critical temperature the liquid density is set constant.
    '''
    T = T_k - T_0_K
    T_cr_C = T_cr - T_0_K
    a_1 = -1.6528e-9
    a_2 = 1.8621e-12
    a_3 = 2.4266e-13
    a_4 = -1.5996e-15
    a_5 = 3.3703e-18
    b_1 = -7.7377e-1
    b_2 = 8.7696e-3
    b_3 = -9.2118e-5
    b_4 = 3.3534e-7
    b_5 = -4.4034e-10
    p_l_1 = 1.0e7
    p_l_ref = 2.0e7
    cond_1 = ((b_1 + 2 * b_2 * T + 3 * b_3 * T**2 + 4 * b_4 * T**3 +
               5 * b_5 * T**4) + (p_l_1 - p_l_ref) *
              (a_1 + 2 * a_2 * T + 3 * a_3 * T**2 + 4 * a_4 * T**3 +
               5 * a_5 * T**4))
    cond_2 = ((b_1 + 2 * b_2 * T_cr_C + 3 * b_3 * T_cr_C**2 +
               4 * b_4 * T_cr_C**3 + 5 * b_5 * T_cr_C**4) + (p_l_1 - p_l_ref) *
              (a_1 + 2 * a_2 * T_cr_C + 3 * a_3 * T_cr_C**2 +
               4 * a_4 * T_cr_C**3 + 5 * a_5 * T_cr_C**4))
    drho_ldT_vals = conditional(lt(T, T_cr_C), cond_1, 0)
    return drho_ldT_vals


def mu_v(T):
    '''
    Viscosity of water vapor, mu_v [Pa s]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    mu_v_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the water vapor's viscosity.

    Notes
    -----
    No notes.
    '''
    mu_v_0 = 8.85e-6
    alpha_v = 3.53e-8
    mu_vals_temp = mu_v_0 + alpha_v * (T - T_0_K)
    mu_vals = conditional(le(T, 370 + 273.15), mu_vals_temp, mu_v_0 + alpha_v * ((370 + 273.15) - T_0_K))
    return mu_vals


def mu_a(T):
    '''
    Viscosity of dry air, mu_a [Pa s]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    mu_a_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the dry air viscosity.

    Notes
    -----
    No notes.
    '''
    mu_a_0 = 17.17e-6
    alpha_a = 4.733e-8
    beta_a = -2.222e-11
    mu_a_vals = mu_a_0 + alpha_a * (T - T_0_K) + beta_a * (T - T_0_K)**2
    return mu_a_vals


def mu_g(Pg, Pc, T):
    '''
    Viscosity of gas, mu_g [Pa s]
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
    mu_g_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the overall gas air viscosity.

    Notes
    -----
    No notes.
    '''
    FGPGA0 = (M_a / (R * T))
    FGPGW0 = (M_v / (R * T))
    CHPGSC0 = Pg
    CHPGWSC0 = Pv(Pg, Pc, T)
    CHPGSCM1 = 1 / Pg
    MUGW0 = mu_v(T)
    MUGA0 = mu_a(T)
    MASQ1  = conditional(gt((CHPGSC0 - CHPGWSC0), 0), 1, 0)
    CHXX   = (1. - (CHPGWSC0 * CHPGSCM1)) * MASQ1 ;
    MUG0   = MUGW0 + ((MUGA0 - MUGW0) * (CHXX ** 0.6083)) ;
    mu_g_vals = MUG0
    return mu_g_vals

def mu_l(T):
    '''
    Viscosity of liquid water, mu_l [Pa s]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    mu_l_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the liquid water viscosity.

    Notes
    -----
    No notes.
    '''
    mu_l_vals_temp = 0.6612 * (T - 229)**(-1.562)
    mu_l_vals = conditional(le(T, 370 + 273.15), mu_l_vals_temp,  0.6612 * ((370 + 273.15) - 229)**(-1.562))
    return mu_l_vals


def H_vap(T):
    '''
    Enthalpy of vaporization of liquid water, H_vap [J/kg]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    H_vap_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents enthalpy of vaporization of liquid water.

    Notes
    -----
    No notes.
    '''
    return conditional(lt(T, T_cr), 2.672e5 * ((T_cr - T)**0.38), 0.0)


def dH_vapdT(T):
    '''
    Derivative of the enthalpy of vaporization of liquid water with respect to
    temperature, dH_vapdT [J/(kg K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dH_vapdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents enthalpy of vaporization of liquid water.

    Notes
    -----
    No notes.
    '''
    dH_vapdT_vals = conditional(lt(T, T_cr),
                                0.38 * 2.6725e5 * (T_cr - T)**(0.38 - 1),
                                0.0)
    return dH_vapdT_vals


def M_g(Pg, Pc, T):
    '''
    Molar Mass of the gas mixture, M_g [kg/mol]
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
    M_g_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents molar mass of the gas mixture.

    Notes
    -----
    No notes.
    '''
    cond_1 = Pg - Pv(Pg, Pc, T)
    M_g_vals_1 = M_a + (M_v - M_a) * Pv(Pg, Pc, T) / Pg
    return M_g_vals_1


def Cp_g(Pg, Pc, T):
    '''
    Specific heat of gas mixture, Cp_g [kg/mol]
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
    Cp_g_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the specific heat of the gas mixture.

    Notes
    -----
    No notes.
    '''
    Cp_g_vals = (rho_g(Pg, Pc, T) *  Cp_a + rho_v(Pg, Pc, T) * (Cp_v - Cp_a))
    return Cp_g_vals


def drho_vdPc(Pg, Pc, T):
    '''
    Derivative of the water vapor's density with respect to capillary pressure,
    drho_vdPc [kg/(m^3 K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_vdPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's density with respect to
    capillary pressure.

    Notes
    -----
    No notes.
    '''
    drho_vdPc_vals = M_v / (R * T) * dPvdPc(Pg, Pc, T)
    return drho_vdPc_vals


def dPvdPc(Pg, Pc, T):
    '''
    Derivative of the water vapor's pressure w/ respect to capillary pressure,
    drho_vdPc [-]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_vdPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's pressure with respect to
    capillary pressure.

    Notes
    -----
    No notes.
    '''
    dPvdPc_vals = - rho_v(Pg, Pc, T) / rho_l(T)
    return dPvdPc_vals


def drho_vdPg(Pg, Pc, T):
    '''
    Derivative of the water vapor's density w/ respect to gas pressure,
    drho_vdPg [kg/(m^3 Pa)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_vdPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's density with respect to
    gas pressure.

    Notes
    -----
    No notes.
    '''
    drho_vdPg_vals = M_v / (R * T) * dPvdPg(Pg, Pc, T)
    return drho_vdPg_vals


def dPvdPg(Pg, Pc, T):
    '''
    Derivative of the water vapor's pressure w/ respect to gas pressure,
    dPvdPg [-]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPvdPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's pressure with respect to
    gas pressure.

    Notes
    -----
    No notes.
    '''
    dPvdPg_vals = rho_v(Pg, Pc, T) / rho_l(T)
    return dPvdPg_vals


def drho_vdT(Pg, Pc, T):
    '''
    Derivative of the water vapor's density w/ respect to gas pressure,
    dPvdPg [kg/(m^3 K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPvdPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's density with respect to
    gas pressure.

    Notes
    -----
    No notes.
    '''
    return (M_v / (R * T)) * dPvdT(Pg, Pc, T) - rho_v(Pg, Pc, T) / T


def dPvdT(Pg, Pc, T):
    '''
    Derivative of the water vapor's pressure w/ respect to temperature,
    dPvdT [Pa/K]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPvdT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of water vapor's pressure with respect to
    temperature.

    Notes
    -----
    '''
    CHPVSAT0 = Pvps(T)
    FGPGW0   = M_v / (R * T)
    RHOW0    = rho_l(T)
    CHXX     = -1. * FGPGW0 * (RHOW0 ** -1) ;
    CHPRE    = Pc - Pg + CHPVSAT0
    CHYY     = exp(CHXX * CHPRE) ;
    CHPGWSC0 = CHPVSAT0 * CHYY;
    DPGWDPG0 = -1. * CHXX * CHPGWSC0;
    DPGWDPC0 = -1. * DPGWDPG0;
    DPSATDT0 = dPvpsdT(T)
    DRHOWDT0 = drho_ldT(T)
    CHTKSCM1 = 1 / T
    TEMP1    = ((CHPGWSC0 * (CHPVSAT0**-1.)) + DPGWDPC0) * DPSATDT0;
    TEMP2    =  DPGWDPC0 * (-1. * CHPRE) * (((RHOW0**-1)* DRHOWDT0)
                + CHTKSCM1);
    dPvdT_vals  = TEMP1 + TEMP2;
    return dPvdT_vals


def drho_adPc(Pg, Pc, T):
    '''
    Derivative of the dry air's density with respect to capillary pressure,
    drho_adPc [kg/(m^3 K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_adPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's density with respect to
    capillary pressure.

    Notes
    -----
    No notes.
    '''
    drho_adPc_vals = M_a / (R * T) * dPadPc(Pg, Pc, T)
    return drho_adPc_vals


def dPadPc(Pg, Pc, T):
    '''
    Derivative of the dry air's pressure w/ respect to capillary pressure,
    dPadPc [-]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPadPc_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's pressure with respect to
    capillary pressure.

    Notes
    -----
    No notes.
    '''
    dPadPc = rho_v(Pg, Pc, T) / rho_l(T)
    return dPadPc


def drho_adPg(Pg, Pc, T):
    '''
    Derivative of the dry air's density w/ respect to gas pressure,
    drho_adPg [kg/(m^3 Pa)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_adPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's density with respect to
    gas pressure.

    Notes
    -----
    No notes.
    '''
    drho_adPg = M_a / (R * T) * dPadPg(Pg, Pc, T)
    return drho_adPg


def dPadPg(Pg, Pc, T):
    '''
    Derivative of the dry air's pressure w/ respect to temperature,
    dPadPg [-]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPadPg_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's pressure with respect to
    temperature.

    Notes
    -----
    No notes.
    '''
    dPadPg = 1 - rho_v(Pg, Pc, T) / rho_l(T)
    return dPadPg


def drho_adT(Pg, Pc, T):
    '''
    Derivative of the dry air's density w/ respect to temperature,
    drho_adT [kg/(m^3K)]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    drho_adT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's density with respect to
    temperature.

    Notes
    -----
    No notes.
    '''
    drho_adT = M_a / (R * T) * dPadT(Pg, Pc, T) - rho_a(Pg, Pc, T) / T
    return drho_adT


def dPadT(Pg, Pc, T):
    '''
    Derivative of the dry air's pressure w/ respect to temperature,
    dPadT [Pa/K]
    Parameters
    ----------
    T [K] : array_like, UFL function
    An array with nodal temperatures or an UFL function for defining
    variational formulations for FEniCS FEM library.

    Returns
    -------
    dPadT_vals :  array_like, UFL function
    An array object or an UFL function (depending on the input)
    that represents the derivative of dry air's pressure with respect to
    temperature.

    Notes
    -----
    No notes.
    '''
    dPadT_vals = - dPvdT(Pg, Pc, T)
    return dPadT_vals
