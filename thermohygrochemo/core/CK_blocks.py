from fenics import *
from thermohygrochemo.core.constant_constitutive_laws import *
from thermohygrochemo.core.global_constants import *
from thermohygrochemo.materials.materials_constitutive_laws import *

# Coefficients
def C_cc(Pg, Pc, T, G):
    '''
    Capacitance Matrix of P_c with terms differentiated with respect to P_c.
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
    C_cc_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    P_c.

    Notes
    -----
    None.
    '''
    C_cc_vals = (phi(Pg, Pc, T, G) * ((1 - S_l(Pg, Pc, T, G)) * drho_vdPc(Pg, Pc, T)
                      + (rho_l(T) - rho_v(Pg, Pc, T)) * dS_ldPc(Pg, Pc, T, G)))
    return C_cc_vals


def C_cg(Pg, Pc, T, G):
    '''
    Capacitance Matrix of P_c with terms differentiated with respect to P_g.
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
    C_cg_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    P_g.

    Notes
    -----
    None.
    '''
    C_cg_vals = ((1 - S_l(Pg, Pc, T, G)) * phi(Pg, Pc, T, G) * drho_vdPg(Pg, Pc, T))
    return C_cg_vals


def C_ct(Pg, Pc, T, G):
    '''
    Capacitance Matrix of P_c with terms differentiated with respect to T.
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
    C_ct_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to
    T.

    Notes
    -----
    None.
    '''
    C_ct_vals = (phi(Pg, Pc, T, G) * ((rho_l(T) - rho_v(Pg, Pc, T)) * dS_ldT(Pg, Pc, T, G)
                 + (1 - S_l(Pg, Pc, T, G)) * drho_vdT(Pg, Pc, T)
                 + S_l(Pg, Pc, T, G) * drho_ldT(T)))
    return C_ct_vals

def K_cc(Pg, Pc, T, G):
    '''
    Conductance Matrix of P_c with terms differentiated with respect to P_c.
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
    K_cc_vals :  array_like, UFL function
    The conductance matrix of P_c with terms differentiated with respect to 
    P_c.

    Notes
    -----
    None.
    '''
    K_cc_vals = - (K_L(Pg, Pc, T, G) * (rho_l(T) * k_rl(Pg, Pc, T, G) / mu_l(T))
                   + D_eff(Pg, Pc, T, G) * (M_v * M_a) / (M_g(Pg, Pc, T) * R * T) *
                   rho_v(Pg, Pc, T) / rho_l(T))
    return K_cc_vals


def K_cg(Pg, Pc, T, G):
    '''
    Conductance Matrix of P_c with terms differentiated with respect to P_g.
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
    K_cg_vals :  array_like, UFL function
    The conductance matrix of P_c with terms differentiated with respect to 
    P_g.

    Notes
    -----
    None.
    '''
    K_cg_vals = (K_L(Pg, Pc, T, G) * rho_l(T) * k_rl(Pg, Pc, T, G) / mu_l(T)
                 + K_G(Pg, Pc, T, G) * rho_v(Pg, Pc, T) * k_rg(Pg, Pc, T, G)
                             / mu_g(Pg, Pc, T)
                 + D_eff(Pg, Pc, T, G) * (M_v * M_a) / (M_g(Pg, Pc, T) * R * T)
                 * (rho_v(Pg, Pc, T) / rho_l(T) - Pv(Pg, Pc, T) / Pg))
    return K_cg_vals


def K_ct(Pg, Pc, T, G):
    '''
    Conductance Matrix of P_c with terms differentiated with respect to T.
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
    K_ct_vals :  array_like, UFL function
    The conductance matrix of P_c with terms differentiated with respect to 
    T.

    Notes
    -----
    None.
    '''
    K_ct_vals = (D_eff(Pg, Pc, T, G) * (M_v * M_a) / (M_g(Pg, Pc, T) * R * T)
                 * dPvdT(Pg, Pc, T))
    return K_ct_vals


def C_gg(Pg, Pc, T, G):
    '''
    Capacitance Matrix of P_g with terms differentiated with respect to P_g.
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
    C_gg_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    P_g.

    Notes
    -----
    None.
    '''
    C_gg_vals = ((1 - S_l(Pg, Pc, T, G)) * phi(Pg, Pc, T, G) * drho_adPg(Pg, Pc, T))
    return C_gg_vals


def C_gc(Pg, Pc, T, G):
    '''
    Capacitance Matrix of P_g with terms differentiated with respect to P_c.
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
    C_gc_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    P_c.

    Notes
    -----
    None.
    '''
    C_gc_vals = ((1 - S_l(Pg, Pc, T, G)) * phi(Pg, Pc, T, G) * drho_adPc(Pg, Pc, T)
                 - rho_a(Pg, Pc, T) * phi(Pg, Pc, T, G) * dS_ldPc(Pg, Pc, T, G))
    return C_gc_vals


def C_gt(Pg, Pc, T, G):
    '''
    Capacitance Matrix of P_g with terms differentiated with respect to T.
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
    C_gt_vals :  array_like, UFL function
    The capacitance matrix of P_c with terms differentiated with respect to 
    T.

    Notes
    -----
    None.
    '''
    C_gt_vals = (phi(Pg, Pc, T, G) * ((1 - S_l(Pg, Pc, T, G)) * drho_adT(Pg, Pc, T)
                             - rho_a(Pg, Pc, T) * dS_ldT(Pg, Pc, T, G)))
    return C_gt_vals

def K_gg(Pg, Pc, T, G):
    '''
    Conductance Matrix of P_g with terms differentiated with respect to T.
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
    K_gg_vals :  array_like, UFL function
    The conductance matrix of P_g with terms differentiated with respect to 
    T.

    Notes
    -----
    None.
    '''
    K_gg_vals = (K_G(Pg, Pc, T, G) * (rho_a(Pg, Pc, T) * k_rg(Pg, Pc, T, G) 
                             / mu_g(Pg, Pc, T))
                 - D_eff(Pg, Pc, T, G) * (M_v * M_a) / (M_g(Pg, Pc, T) * R * T) *
                 (rho_v(Pg, Pc, T) / rho_l(T) - Pv(Pg, Pc, T) / Pg))
    return K_gg_vals


def K_gc(Pg, Pc, T, G):
    '''
    Conductance Matrix of P_g with terms differentiated with respect to P_c.
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
    K_gc_vals :  array_like, UFL function
    The conductance matrix of P_g with terms differentiated with respect to 
    P_c.

    Notes
    -----
    None.
    '''
    K_gc_vals = (D_eff(Pg, Pc, T, G) * (M_v * M_a) / (M_g(Pg, Pc, T) * R * T) *
                 (rho_v(Pg, Pc, T) / rho_l(T)))
    return K_gc_vals


def K_gt(Pg, Pc, T, G):
    '''
    Conductance Matrix of P_g with terms differentiated with respect to T.
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
    K_gt_vals :  array_like, UFL function
    The conductance matrix of P_g with terms differentiated with respect to 
    T.

    Notes
    -----
    None.
    '''
    K_gt_vals = - (D_eff(Pg, Pc, T, G) * (M_v * M_a) / (M_g(Pg, Pc, T) * R * T) *
                   dPvdT(Pg, Pc, T))
    return K_gt_vals


def C_tt(Pg, Pc, T, G):
    '''
    Capacitance Matrix of T with terms differentiated with respect to T.
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
    C_tt_vals :  array_like, UFL function
    The capacitance matrix of T with terms differentiated with respect to 
    T.

    Notes
    -----
    None.
    '''
    C_tt_vals = (- H_vap(T) * (phi(Pg, Pc, T, G) * (S_l(Pg, Pc, T, G) * drho_ldT(T) 
                                           + rho_l(T) * dS_ldT(Pg, Pc, T, G))
                               )
                 + rhoCp(Pg, Pc, T, G))
    return C_tt_vals


def C_tc(Pg, Pc, T, G):
    '''
    Capacitance Matrix of T with terms differentiated with respect to P_c.
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
    C_tc_vals :  array_like, UFL function
    The capacitance matrix of T with terms differentiated with respect to 
    P_c.

    Notes
    -----
    None.
    '''
    C_tc_vals = - (H_vap(T) * rho_l(T) * phi(Pg, Pc, T, G) * dS_ldPc(Pg, Pc, T, G))
    return C_tc_vals


def C_tg(Pg, Pc, T, G):
    '''
    Capacitance Matrix of T with terms differentiated with respect to P_g.
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
    C_tg_vals :  array_like, UFL function
    The capacitance matrix of T with terms differentiated with respect to 
    P_g.

    Notes
    -----
    This term is null.
    '''
    C_tg_vals = 0
    return C_tg_vals


def K_tt(Pg, Pc, T, G):
    '''
    Conductance Matrix of T with terms differentiated with respect to T.
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
    K_ct_vals :  array_like, UFL function
    The conductance matrix of T with terms differentiated with respect to 
    T.

    Notes
    -----
    None
    '''
    return lambda_eff(Pg, Pc, T, G)


def K_tc(Pg, Pc, T, G):
    '''
    Conductance Matrix of T with terms differentiated with respect to P_c.
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
    K_tc_vals :  array_like, UFL function
    The conductance matrix of T with terms differentiated with respect to 
    Pc.

    Notes
    -----
    None.
    '''
    K_tc_vals = (H_vap(T) * K_L(Pg, Pc, T, G) * rho_l(T) * k_rl(Pg, Pc, T, G) / mu_l(T))
    return K_tc_vals


def K_tg(Pg, Pc, T, G):
    '''
    Conductance Matrix of T with terms differentiated with respect to g.
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
    K_tg_vals :  array_like, UFL function
    The conductance matrix of T with terms differentiated with respect to 
    P_g.

    Notes
    -----
    None.
    '''
    K_tg_vals = - (H_vap(T) * K_L(Pg, Pc, T, G) * rho_l(T) * k_rl(Pg, Pc, T, G) / mu_l(T))
    return K_tg_vals


def F_g(Pg, Pc, T, G):
    '''
    The vector F with respect to g.
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
    F_g_vals :  array_like, UFL function
    The vector F_g.

    Notes
    -----
    None.
    '''
    F_g_vals_1 = (a_phi(Pg, Pc, T, G) * rho_a(Pg, Pc, T) * (1 - S_l(Pg, Pc, T, G))
                  + phi(Pg, Pc, T, G) * rho_a(Pg, Pc, T) * dS_ldGamma(Pg, Pc, T, G))
    F_g_vals = F_g_vals_1
    return F_g_vals


def F_c(Pg, Pc, T, G):
    '''
    The vector F with respect to g.
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
    F_c_vals :  array_like, UFL function
    The vector F_c.

    Notes
    -----
    None.
    '''
    F_c_vals_1 = ((rho_l(T) * S_l(Pg, Pc, T, G))
                  + (rho_v(Pg, Pc, T) * (1 - S_l(Pg, Pc, T, G)))) * a_phi(Pg, Pc, T, G)
    F_c_vals_2 = - m_hyd_inf
    F_c_vals_3 = - phi(Pg, Pc, T, G) * (rho_l(T)
                                    - rho_v(Pg, Pc, T)) * dS_ldGamma(Pg, Pc, T, G)
    F_c_vals = (F_c_vals_1
                + F_c_vals_2
                + F_c_vals_3)
    return F_c_vals


def F_t(Pg, Pc, T, G):
    '''
    The vector F with respect to g.
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
    F_g_vals :  array_like, UFL function
    The vector F_t.

    Notes
    -----
    None.
    '''

    F_t_vals_1 = H_dehyd
    F_t_vals_2 = H_vap(T) * m_hyd_inf
    F_t_vals_3 = H_vap(T) * rho_l(T) * phi(Pg, Pc, T, G) * dS_ldGamma(Pg, Pc, T, G)
    F_t_vals_4 = - H_vap(T) * rho_l(T) * S_l(Pg, Pc, T, G) * a_phi(Pg, Pc, T, G)
    F_t_vals = (F_t_vals_1 + F_t_vals_2
                + F_t_vals_3 + F_t_vals_4)
    return F_t_vals
