from fenics import *
from thermohygrochemo.core.global_constants import *
from thermohygrochemo.core.CK_blocks import *
from thermohygrochemo.core.constant_constitutive_laws import *
from thermohygrochemo.materials.materials_constitutive_laws import *
from os import path, makedirs
from numpy import array, savez
from datetime import datetime
from time import sleep
import meshio
import shutil
import pathlib
import csv
import numpy as np
import sys
from signal import signal, SIGINT


core_dir = pathlib.Path(__file__).parent.absolute()
thermo_dir_ = pathlib.Path(core_dir).parent.absolute()
thermo_dir = str(thermo_dir_)
thc_model_dir = str(pathlib.Path(thermo_dir_).parent.absolute())

'''
The core of the thermohygrochemical model. There are 1 class the first is the
base class, all others are derived from it. They are:
1) class thc_model_core: Base class for the model in Cartesian coordinate system
2) thc_model_axisymmetric: Class for axisymmetric model.
'''


class thc_model_core(object):
    def __init__(self, mesh_input, boundaries_input, t_total,
                 Pg_0, Pc_0, T_0, G_0,
                 Pg_BC_cfg, Pc_BC_cfg, T_BC_cfg,
                 dir_output, dir_backup, case_file, freq_out=10,
                 stages_cfg=None, load_cfg=None,
                 FEniCS_DEBUG=False):
        '''
        Initialization of the thc_model_core object.
        It receives as input all the parameters defined
        in the `case_input.py` file.
        It creates all the instances of the object (the variables
        with a self. as a prefix).
        '''

        self.mesh_input = mesh_input    # Mesh information
        # Boundaries information
        self.boundaries_input = boundaries_input

        self.t_total = t_total          # Total time
        self.t = 0                      # Current time
        self.dt = Constant(0.0)         # Time step

        self.Pg_0 = Pg_0                # Initial Pg
        self.Pc_0 = Pc_0                # Initial Pc
        self.T_0 = T_0                  # Initial T
        self.G_0 = G_0                  # Initial G

        self.f_evol = {}                # Dict w/ time dependent functions
        self.f_evol['h_g'] = {}         # Subdict w/ time dependent h_g
        self.f_evol['h_T'] = {}         # Subdict w/ time dependent h_T
        self.f_evol['rho_a_inf'] = {}   # Subdict w/ time dependent rho_a_inf
        self.f_evol['rho_v_inf'] = {}   # Subdict w/ time dependent rho_v_inf
        self.f_evol['Pg_fix'] = {}      # Subdict w/ time dependent Pg_fix
        self.f_evol['Pc_fix'] = {}      # Subdict w/ time dependent Pc_fix
        self.f_evol['T_rad_inf'] = {}   # Subdict w/ time dependent T_rad_inf
        self.f_evol['T_conv_inf'] = {}  # Subdict w/ time dependent T_conv_inf
        self.f_evol['T_inf'] = {}       # Subdict w/ time dependent T_inf
        self.f_evol['T_fix'] = {}       # Subdict w/ time dependent T_fix

        self.Pg_BC_cfg = Pg_BC_cfg      # BC configuration for Pg
        self.Pc_BC_cfg = Pc_BC_cfg      # BC configuration for Pc
        self.T_BC_cfg = T_BC_cfg        # BC configuration for T

        self.dir_output = dir_output    # Directory to write the results
        self.dir_backup = dir_backup    # Directory to write the backup
        self.case_file = case_file      # Case file name
        self.stages_cfg = stages_cfg    # Cases configuration
        self.freq_out = freq_out        # Frequency of writing output
        # Flag to show debug information on Newton Solver
        self.FEniCS_DEBUG = FEniCS_DEBUG

        self.breakTimeLoop = False      # Flag to break time loop
        self.terminateStatus = None     # Flag to know how the loop finished

        if self.FEniCS_DEBUG:           # Set FEniCS log level
            set_log_level(20)           # Prints Newton's solver info
        else:
            set_log_level(50)           # Prints nothing


    def initiate_mesh(self):
        '''
        Loads the mesh from case file.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''
        self.mesh = self.mesh_input

    def initiate_boundary_markers(self):
        '''
        Loads the boundary markers from case file.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        self.boundaries = self.boundaries_input

    def generate_function_spaces(self):
        '''
        Create the FE function space for the primary variables.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        # 2nd Degree Polynomial FE
        P2 = FiniteElement('P', self.mesh.ufl_cell(), 1)
        # Mixed Element for the primery variables
        element = MixedElement([P2, P2, P2, P2])
        # Function Space defined over mesh w. FE
        self.V = FunctionSpace(self.mesh, element)

    def generate_functions(self):
        '''
        Create the trial functions to define the residual.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        # Function defined over V
        self.u = Function(self.V)
        # Spliting u into the 4 primary variables
        self.Pg, self.Pc, self.T, self.G = split(self.u)
        # Creating auxiliary functions for tentative time step
        self.u_low = Function(self.V)
        self.u_high_1 = Function(self.V)
        self.u_high_2 = Function(self.V)
        self.T_max = project(self.T_0, self.V.sub(2).collapse())

    def generate_IC_functions(self):
        '''
        Create the initial condtions functions

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        self.Pg_n = interpolate(Constant(self.Pg_0), self.V.sub(0).collapse())
        self.Pc_n = interpolate(Constant(self.Pc_0), self.V.sub(1).collapse())
        self.T_n = interpolate(Constant(self.T_0), self.V.sub(2).collapse())
        self.G_n = interpolate(Constant(self.G_0), self.V.sub(3).collapse())
        self.Pg_n.rename("Gas Pressure [Pa]", "Pg")
        self.Pc_n.rename("Capillary Pressure [Pa]", "Pc")
        self.T_n.rename("Temperature [K]", "T")
        self.G_n.rename("Hyd. Degree [-]", "G")

    def define_total_residual(self, STAGE=None, DEBUG=True):
        '''
        Create the total residual for the current stage.

        Parameters
        ----------
        STAGE : str
            The current stage of the simulation. If None, starts with
            the first stage.

        DEBUG : bool
            If True, print information about where each boundary
            condition is added.

        Returns
        -------
        None.
        '''

        if STAGE is None:
            STAGE = self.stages_cfg['stage_1']['case']

        # Substitute the ds measure with a new measure w. the marked boundaries
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        self.ds = ds
        # Test functions of V
        v_1, v_2, v_3, v_4 = TestFunctions(self.V)
        # Create local variables just for readability's sake
        (Pg, Pc, T, G) = (self.Pg, self.Pc, self.T, self.G)
        (Pg_n, Pc_n, T_n, G_n) = (self.Pg_n, self.Pc_n, self.T_n, self.G_n)
        T_max = self.T_max
        dt = self.dt

        dGamma_tilde_dt = ((1 - F(T)) * dGammadt(Pg_n, Pc_n, T_n, G_n)
                           - G_n * dFdT(T) * ((T - T_n) / dt))
        # "Dry Air Equation"
        MBA = C_gg(Pg_n, Pc_n, T_n, G_n) * ((Pg - Pg_n) / dt) * v_1 * dx
        MBA += C_gc(Pg_n, Pc_n, T_n, G_n) * ((Pc - Pc_n) / dt) * v_1 * dx
        MBA += C_gt(Pg_n, Pc_n, T_n, G_n) * ((T - T_n) / dt) * v_1 * dx
        MBA += inner(K_gg(Pg_n, Pc_n, T_n, G_n) * grad(Pg), grad(v_1)) * dx
        MBA += inner(K_gc(Pg_n, Pc_n, T_n, G_n) * grad(Pc), grad(v_1)) * dx
        MBA += inner(K_gt(Pg_n, Pc_n, T_n, G_n) * grad(T), grad(v_1)) * dx
        MBA += - F_g(Pg_n, Pc_n, T_n, G_n) * dGamma_tilde_dt * v_1 * dx

        def Pg_neumann_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pg
                Neumann BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal MBA, DEBUG, v_1, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pg Neumann Condition on the '
                       f'{boundary.capitalize()}'))

            # The value of fixed air flux applied at this boundary
            q_bar_a = entry['q_bar_a_val']

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            MBA += - q_bar_a * v_1 * ds(marker)

        def Pg_robin_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pg
                Robin BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal MBA, DEBUG, Pg_n, Pc_n, T_n, v_1, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pg Convection on the '
                       f'{boundary.capitalize()}'))

            # The air density defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['rho_a_inf_val']):
                self.f_evol['rho_a_inf'][boundary] = {}
                (self.f_evol['rho_a_inf']
                            [boundary]
                            ['evol']) = entry['rho_a_inf_val']
                (self.f_evol['rho_a_inf']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=entry['rho_a_inf_val'](self.t))

            else:
                rho_a_val = entry['rho_a_inf_val']
                self.f_evol['rho_a_inf'][boundary] = {}
                (self.f_evol['rho_a_inf']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=rho_a_val)

            # The mass transfer coefficient defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['h_g_val']):
                self.f_evol['h_g'][boundary] = {}
                h_g = entry['h_g_val']
                (self.f_evol['h_g']
                            [boundary]
                            ['evol']) = h_g
                (self.f_evol['h_g']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=h_g(self.t))

            else:
                self.f_evol['h_g'][boundary] = {}
                h_g = entry['h_g_val']
                (self.f_evol['h_g']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=h_g)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            MBA += (self.f_evol['h_g'][boundary]['exp']
                    * drho_adPg(Pg_n, Pc_n, T_n)
                    * (Pg - Pg_n) * v_1 * ds(marker))
            MBA += (self.f_evol['h_g'][boundary]['exp']
                    * drho_adPc(Pg_n, Pc_n, T_n)
                    * (Pc - Pc_n) * v_1 * ds(marker))
            MBA += (self.f_evol['h_g'][boundary]['exp']
                    * drho_adT(Pg_n, Pc_n, T_n)
                    * (T - T_n) * v_1 * ds(marker))
            MBA += (self.f_evol['h_g'][boundary]['exp']
                    * (rho_a(Pg_n, Pc_n, T_n)
                       - (self.f_evol['rho_a_inf']
                                     [boundary]
                                     ['exp'])) * v_1 * ds(marker))

        # Add Neumman or Robin BC for each boundary on the configuration dict
        for boundary in self.Pg_BC_cfg[STAGE]:
            entry = self.Pg_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'neumann':
                Pg_neumann_BC_parser(boundary, entry)
            elif entry['condition'] == 'robin':
                Pg_robin_BC_parser(boundary, entry)

        # "Water Equation"
        MBH = C_cc(Pg_n, Pc_n, T_n, G_n) * ((Pc - Pc_n) / dt) * v_2 * dx
        # MBH += C_cg(Pg_n, Pc_n, T_n, G_n) * ((Pg - Pg_n) / dt) * v_2 * dx
        MBH += C_ct(Pg_n, Pc_n, T_n, G_n) * ((T - T_n) / dt) * v_2 * dx
        MBH += inner(K_cc(Pg_n, Pc_n, T_n, G_n) * grad(Pc), grad(v_2)) * dx
        MBH += inner(K_cg(Pg_n, Pc_n, T_n, G_n) * grad(Pg), grad(v_2)) * dx
        MBH += inner(K_ct(Pg_n, Pc_n, T_n, G_n) * grad(T), grad(v_2)) * dx
        MBH += - F_c(Pg_n, Pc_n, T_n, G_n) * dGamma_tilde_dt * v_2 * dx

        def Pc_neumann_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pc
                Neumann BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal MBH, DEBUG, v_2, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pc Neumann Condition on the '
                       f'{boundary.capitalize()}'))

            # The value of fixed liquid and vapor flux applied at this boundary
            q_bar_l = entry['q_bar_l_val']
            q_bar_v = entry['q_bar_v_val']

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            MBH += - (q_bar_l + q_bar_v) * v_2 * ds(marker)

        def Pc_robin_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pc
                Robin BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal MBH, DEBUG, Pg_n, Pc_n, T_n, v_2, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pc Convection on the '
                       f'{boundary.capitalize()}'))

            # The vapor density defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['rho_v_inf_val']):
                self.f_evol['rho_v_inf'][boundary] = {}

                (self.f_evol['rho_v_inf']
                            [boundary]
                            ['evol']) = entry['rho_v_inf_val']
                (self.f_evol['rho_v_inf']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=entry['rho_v_inf_val'](self.t))

            else:
                self.f_evol['rho_v_inf'][boundary] = {}
                (self.f_evol['rho_v_inf']
                            [boundary]
                            ['val']) = entry['rho_v_inf_val']
                (self.f_evol['rho_v_inf']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=entry['rho_v_inf_val'])

            # The mass transfer coefficient defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['h_g_val']):
                h_g = entry['h_g_val']
                self.f_evol['h_g'][boundary] = {}
                (self.f_evol['h_g']
                            [boundary]
                            ['evol']) = h_g
                (self.f_evol['h_g']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=h_g(self.t))

            else:
                h_g = entry['h_g_val']
                self.f_evol['h_g'][boundary] = {}
                (self.f_evol['h_g']
                            [boundary]
                            ['val']) = entry['h_g_val']
                (self.f_evol['h_g']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=h_g)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            MBH += (self.f_evol['h_g'][boundary]['exp']
                    * drho_vdPg(Pg_n, Pc_n, T_n)
                    * (Pg - Pg_n) * v_2 * ds(marker))
            MBH += (self.f_evol['h_g'][boundary]['exp']
                    * drho_vdPc(Pg_n, Pc_n, T_n)
                    * (Pc - Pc_n) * v_2 * ds(marker))
            MBH += (self.f_evol['h_g'][boundary]['exp']
                    * drho_vdT(Pg_n, Pc_n, T_n)
                    * (T - T_n) * v_2 * ds(marker))
            MBH += (self.f_evol['h_g'][boundary]['exp']
                    * (rho_v(Pg_n, Pc_n, T_n)
                       - (self.f_evol['rho_v_inf']
                                     [boundary]
                                     ['exp'])) * v_2 * ds(marker))

        # Add Neumman or Robin BC for each boundary on the configuration dict
        for boundary in self.Pc_BC_cfg[STAGE]:
            entry = self.Pc_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'neumann':
                Pc_neumann_BC_parser(boundary, entry)
            elif entry['condition'] == 'robin':
                Pc_robin_BC_parser(boundary, entry)

        # Energy conservation equation
        ECE = C_tt(Pg_n, Pc_n, T_n, G_n) * ((T - T_n) / dt) * v_3 * dx
        ECE += C_tc(Pg_n, Pc_n, T_n, G_n) * ((Pc - Pc_n) / dt) * v_3 * dx
        # ECE += C_tg = 0
        ECE += inner(K_tt(Pg_n, Pc_n, T_n, G_n) * grad(T), grad(v_3)) * dx
        ECE += inner(K_tc(Pg_n, Pc_n, T_n, G_n) * grad(Pc), grad(v_3)) * dx
        ECE += inner(K_tg(Pg_n, Pc_n, T_n, G_n) * grad(Pg), grad(v_3)) * dx
        ECE += - F_t(Pg_n, Pc_n, T_n, G_n) * dGamma_tilde_dt * v_3 * dx

        def T_neumann_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for T
                Neumann BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal ECE, DEBUG, v_3, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding T Neumann Condition on the '
                       f'{boundary.capitalize()}'))

            # The value of fixed heat flux applied at this boundary
            q_bar_T = entry['q_bar_T_val']

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            ECE += - q_bar_T * v_3 * ds(marker)

        def T_robin_conv_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for T
                Robin Convection BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal ECE, DEBUG, Pg_n, Pc_n, T_n, v_3, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding T Convection on the '
                       f'{boundary.capitalize()}'))

            # The temperature at the far field defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['T_inf_val']):
                T_inf = entry['T_inf_val']
                self.f_evol['T_conv_inf'][boundary] = {}
                (self.f_evol['T_conv_inf']
                            [boundary]
                            ['evol']) = T_inf
                (self.f_evol['T_conv_inf']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=T_inf(self.t))

            else:
                T_inf = entry['T_inf_val']
                self.f_evol['T_conv_inf'][boundary] = {}
                (self.f_evol['T_conv_inf']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=T_inf)

            # The heat transfer coefficient defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['h_T_val']):
                h_T = entry['h_T_val']
                self.f_evol['h_T'][boundary] = {}
                (self.f_evol['h_T']
                            [boundary]
                            ['evol']) = h_T
                (self.f_evol['h_T']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=h_T(self.t))

            else:
                h_T = entry['h_T_val']
                self.f_evol['h_T'][boundary] = {}
                (self.f_evol['h_T']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=h_T)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            ECE += ((self.f_evol['h_T'][boundary]['exp']
                     * (T - self.f_evol['T_conv_inf'][boundary]['exp']))
                    * v_3 * ds(marker))

        def T_robin_rad_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pg
                Robin BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal ECE, DEBUG, Pg_n, Pc_n, T_n, v_3, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding T Radiation on the '
                       f'{boundary.capitalize()}'))

            # The temperature at the far field defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['T_inf_val']):
                T_inf = entry['T_inf_val']
                self.f_evol['T_rad_inf'][boundary] = {}
                (self.f_evol['T_rad_inf']
                            [boundary]
                            ['evol']) = T_inf
                (self.f_evol['T_rad_inf']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=T_inf(self.t))

            else:
                T_inf = entry['T_inf_val']
                self.f_evol['T_rad_inf'][boundary] = {}
                (self.f_evol['T_rad_inf']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=T_inf)

            # Thermal total emissivity of the boundary
            epsilon = entry['epsilon_val']

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            ECE += ((epsilon * sigma_SB
                     * (T_n**3 * T
                        - self.f_evol['T_rad_inf'][boundary]['exp']**4))
                    * v_3 * ds(marker))

        # Add Neumman or Robin Convection, or Robin Radiation BC
        # for each boundary on the configuration dict
        for boundary in self.T_BC_cfg[STAGE]:
            entry = self.T_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'neumann':
                T_neumann_BC_parser(boundary, entry)
            if 'conv' in entry['condition']:
                T_robin_conv_BC_parser(boundary, entry)
            if 'rad' in entry['condition']:
                T_robin_rad_BC_parser(boundary, entry)

        # Hydration Degree Equation
        GE = (- dGammadt(Pg_n, Pc_n, T_n, G_n) + ((G - G_n) / dt)) * v_4 * dx
        Res = MBH + MBA + ECE + GE
        self.Res = Res

    def generate_BCS(self, STAGE=None, DEBUG=True):
        '''
        Create the boundary conditions for the current case.

        Parameters
        ----------
        STAGE : str
            The current stage of the simulation. If None, starts with
            the first stage.

        DEBUG : bool
            If True, print information about where each boundary
            condition is added.

        Returns
        -------
        None.
        '''

        def Pg_dirichlet_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pg
                Dirichlet BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal dirichlet_bcs, DEBUG

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pg Dirichlet Boundary'
                       f' Condition on {boundary.capitalize()}'))

            # The fixed Gas Pressure defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['Pg_fix_val']):
                Pg_fix = entry['Pg_fix_val']
                self.f_evol['Pg_fix'][boundary] = {}
                (self.f_evol['Pg_fix']
                            [boundary]
                            ['evol']) = Pg_fix
                (self.f_evol['Pg_fix']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=Pg_fix(self.t))

            else:
                Pg_fix = entry['Pg_fix_val']
                self.f_evol['Pg_fix'][boundary] = {}
                (self.f_evol['Pg_fix']
                            [boundary]
                            ['exp']) = Expression('val ', degree=2,
                                                  val=Pg_fix)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Dirchlet BC FEniCS object
            bc_d = DirichletBC(self.V.sub(0),
                               self.f_evol['Pg_fix'][boundary]['exp'],
                               self.boundaries, marker)

            # Update the list of Dirchlet BC
            dirichlet_bcs += [bc_d]

        def Pc_dirichlet_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pc
                Dirichlet BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal dirichlet_bcs, DEBUG

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pc Dirichlet Boundary'
                       f' Condition on {boundary.capitalize()}'))

            # The fixed Capillary Pressure defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['Pc_fix_val']):
                Pc_fix = entry['Pc_fix_val']
                self.f_evol['Pc_fix'][boundary] = {}
                (self.f_evol['Pc_fix']
                            [boundary]
                            ['evol']) = Pc_fix
                (self.f_evol['Pc_fix']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=Pc_fix(self.t))

            else:
                Pc_fix = entry['Pc_fix_val']
                self.f_evol['Pc_fix'][boundary] = {}
                (self.f_evol['Pc_fix']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=Pc_fix)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Dirchlet BC FEniCS object
            bc_d = DirichletBC(self.V.sub(1),
                               self.f_evol['Pc_fix'][boundary]['exp'],
                               self.boundaries, marker)

            # Update the list of Dirchlet BC
            dirichlet_bcs += [bc_d]

        def T_dirichlet_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for T
                Dirichlet BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal dirichlet_bcs, DEBUG

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding T Dirichlet Boundary'
                       f' Condition on {boundary.capitalize()}'))

            # The fixed Temperature defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['T_fix_val']):
                T_fix = entry['T_fix_val']
                self.f_evol['T_fix'][boundary] = {}
                (self.f_evol['T_fix']
                            [boundary]
                            ['evol']) = T_fix
                (self.f_evol['T_fix']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=T_fix(self.t))

            else:
                T_fix = entry['T_fix_val']
                self.f_evol['T_fix'][boundary] = {}
                (self.f_evol['T_fix']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=T_fix)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Dirchlet BC FEniCS object
            bc_d = DirichletBC(self.V.sub(2),
                               self.f_evol['T_fix'][boundary]['exp'],
                               self.boundaries, marker)

            # Update the list of Dirchlet BC
            dirichlet_bcs += [bc_d]

        if STAGE is None:
            STAGE = self.stages_cfg['stage_1']['case']
        dirichlet_bcs = []

        for boundary in self.Pg_BC_cfg[STAGE]:
            entry = self.Pg_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'dirichlet':
                Pg_dirichlet_BC_parser(boundary, entry)

        for boundary in self.Pc_BC_cfg[STAGE]:
            entry = self.Pc_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'dirichlet':
                Pc_dirichlet_BC_parser(boundary, entry)

        for boundary in self.T_BC_cfg[STAGE]:
            entry = self.T_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'dirichlet':
                T_dirichlet_BC_parser(boundary, entry)

        self.dirichlet_bcs = dirichlet_bcs

    def create_variational_problem_and_solver(self):
        '''
        Creates the variational problem and a newton solver with
        the standard parameters.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        parameters["form_compiler"]["cpp_optimize"] = True
        self.ffc_options = {"quadrature_degree": 6, "optimize": True}
        if has_linear_algebra_backend("Epetra"):
            parameters["linear_algebra_backend"] = "Epetra"

        Jac = derivative(self.Res, self.u)
        self.problem = NonlinearVariationalProblem(self.Res, self.u,
                                                   self.dirichlet_bcs, Jac,
                                                   self.ffc_options)
        self.solver = NonlinearVariationalSolver(self.problem)
        self.solver.parameters["newton_solver"]["absolute_tolerance"] = 1E-8
        self.solver.parameters["newton_solver"]["relative_tolerance"] = 1E-14
        self.solver.parameters["newton_solver"]["maximum_iterations"] = 10
        (self.solver.parameters['newton_solver']
                               ['error_on_nonconvergence']) = True

    def prepare_outputs(self):
        '''
        Create the directories and the files for writing the results.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        dir_output = thc_model_dir + '/results' + self.dir_output
        dir_backup = self.dir_backup
        if not path.exists(dir_output):
            mkdir(dir_output)
        if not path.exists(dir_output + dir_backup):
            mkdir(dir_output + dir_backup)

        shutil.copy(__file__,
                    dir_output + dir_backup + '/model.py')
        shutil.copy(self.case_file,
                    dir_output + dir_backup + '/case_input.py')
        shutil.copy(thermo_dir + '/materials/materials_constitutive_laws.py',
                    dir_output + dir_backup
                    + '/materials_constitutive_laws.py')
        shutil.copy(thermo_dir + '/core/CK_blocks.py',
                    dir_output + dir_backup
                    + '/CK_blocks.py')

        # Files for saving the fields
        self.filex_mesh = XDMFFile(dir_output
                                   + f'/Mesh_Field_{self.dir_output[1:]}.xdmf')

        self.filex_bdry = XDMFFile(dir_output
                                   + f'/Boundaries_{self.dir_output[1:]}.xdmf')
        self.filex_mesh.write(self.mesh)
        self.filex_bdry.write(self.boundaries)

        self.filex_Pg = XDMFFile(dir_output
                                 + f'/Pg_Field_{self.dir_output[1:]}.xdmf')
        self.filex_Pg.parameters['functions_share_mesh'] = True
        self.filex_Pg.parameters['rewrite_function_mesh'] = False
        self.filex_Pg.parameters["flush_output"] = True

        self.filex_Pc = XDMFFile(dir_output
                                 + f'/Pc_Field_{self.dir_output[1:]}.xdmf')
        self.filex_Pc.parameters['functions_share_mesh'] = True
        self.filex_Pc.parameters['rewrite_function_mesh'] = False
        self.filex_Pc.parameters["flush_output"] = True

        self.filex_T = XDMFFile(dir_output
                                + f'/T_Field_{self.dir_output[1:]}.xdmf')
        self.filex_T.parameters['functions_share_mesh'] = True
        self.filex_T.parameters['rewrite_function_mesh'] = False
        self.filex_T.parameters["flush_output"] = True

        self.filex_G = XDMFFile(dir_output
                                + f'/G_Field_{self.dir_output[1:]}.xdmf')
        self.filex_G.parameters['functions_share_mesh'] = True
        self.filex_G.parameters['rewrite_function_mesh'] = False
        self.filex_G.parameters["flush_output"] = True

    def time_step_transition(self, nt, dt_1, dt_2, nt_tr, rate=None):
        '''
        Function to perform a logistic timestep transition.
        Reference: https://en.wikipedia.org/wiki/Logistic_function

        Parameters
        ----------
        nt : array_like
            Time step increment value.
        dt_1 : float
            Initial time step.
        dt_2 : float
            Final time step.
        nt_0: float
            The time step increment in which the dt is 50% of
            the final time step.
        rate: float, optional
            How steep is the time step variation.
            By default it is the half of the number of time step variation.

        Returns
        -------
        dt_val : ndarray or float
            Current time step.
        '''

        delta_dt = dt_2 - dt_1
        if not rate:
            rate = nt_tr / nt_tr
        nt_0 = nt_tr / 2
        dt_val = np.around(dt_1 + delta_dt
                           / (1 + np.exp(- rate * (nt - nt_0))))
        return dt_val

    def setup_stages(self):
        '''
        Prepare the stages related variables, specially the time step
        transition between different time steps and the current step
        size selector name.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        total_duration = 0
        all_stages = sorted(self.stages_cfg.keys())
        for s, STAGE in enumerate(all_stages):
            total_duration += self.stages_cfg[STAGE]['duration']
            if self.stages_cfg[STAGE]['stepsizeselector'] == 'manual':
                self.dts_tr = self.time_step_transition(np.arange(1, self.stages_cfg[STAGE]['tau']),
                                                        self.stages_cfg[STAGE]['dt'],
                                                        self.stages_cfg[all_stages[s + 1]]['dt'],
                                                        self.stages_cfg[STAGE]['tau'],
                                                        rate=self.stages_cfg[STAGE]['rate'])
                self.stages_cfg[STAGE]['dts_tr'] = self.dts_tr
                self.stages_cfg[STAGE]['t_tr'] = self.dts_tr.sum()
                self.stages_cfg[STAGE]['time_step_transition_remainder_tr'] = self.dts_tr.sum() % self.stages_cfg[STAGE]['dt']
                self.stages_cfg[STAGE]['nt_tr'] = 1
            self.stages_cfg[STAGE]['total_duration'] = total_duration


    def update_current_stage(self):
        '''
        Updates the current stage.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        for s, STAGE in enumerate(self.stages_cfg.keys()):
            if self.t <= self.stages_cfg[STAGE]['total_duration']:
                self.PROBLEM = f'{self.stages_cfg[STAGE]["name"]}'
                if self.CASE != self.stages_cfg[STAGE]['case']:
                    self.CASE = self.stages_cfg[STAGE]['case']
                    self.generate_BCS(STAGE=self.CASE)
                    self.define_total_residual(STAGE=self.CASE)
                    self.create_variational_problem_and_solver()
                return

    def dtManual(self):
        '''
        A manual step size selector.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        for s, STAGE in enumerate(self.stages_cfg.keys()):
            if self.t < self.stages_cfg[STAGE]['total_duration'] - self.stages_cfg[STAGE]['t_tr'] - self.stages_cfg[STAGE]['time_step_transition_remainder_tr']:
                self.PROBLEM = f'{self.stages_cfg[STAGE]["name"]}'
                self.dt.assign(self.stages_cfg[STAGE]['dt'])
                self.update_time_dependent_functions(self.t)
                return

            elif self.t < self.stages_cfg[STAGE]['total_duration'] - self.stages_cfg[STAGE]['time_step_transition_remainder_tr']:
                self.PROBLEM = f'{self.stages_cfg[STAGE]["name"]} - transition'
                self.dt.assign((self.stages_cfg[STAGE]['dts_tr'])[self.stages_cfg[STAGE]['nt_tr'] - 1])
                if self.stages_cfg[STAGE]['nt_tr'] < self.stages_cfg[STAGE]['tau'] - 1:
                    self.stages_cfg[STAGE]['nt_tr'] += 1
                self.update_time_dependent_functions(self.t)
                return

            elif self.t < self.stages_cfg[STAGE]['total_duration']:
                self.PROBLEM = (f'{(self.stages_cfg[STAGE]["name"])}'
                                f' - enforcing')
                self.dt.assign((self.stages_cfg[STAGE]
                                               ['total_duration']) - self.t)
                self.update_time_dependent_functions(self.t)
                return
            

    def verifyStepsize(self):
        '''
        Step size can not be:
          - greater than remaining time domain

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        if self.t + float(self.dt) >= self.t_total:
            self.dt.assign(self.t_total - self.t)
            self.breakTimeLoop = True

    def solve(self):
        '''
        Function that solves the residual with the Newton Nonlinear solver.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        try:
            # Number of newton iterations and convergence flag
            self.n_newton, self.converged = self.solver.solve()
        except RuntimeError:
            # Set the convergence flag to False
            self.converged = False

    def simulation(self):
        '''
        Function for the transient simulation.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        self.nt = 0                   # Current number of the time step
        self.dts = []                 # Lists for time step size
        self.STAGES_EVOL = []         # Lists for current stage

        # Setup the stages of the simulation
        self.setup_stages()

        # Start the timer of the total simulation
        startTime = datetime.now()

        # Time stepping loop
        while True:
            # 1. Update time dependent functions used in the current solution
            self.update_time_dependent_functions(self.t)

            # 2. Set the dt
            self.dtManual()

            # 3. Update current stage
            self.update_current_stage()

            # 4. Solve non-linear problem with current time step
            self.solve()

            # 5.1. Update solution with last computed value
            self.update_vars_n(self.u)

            # 5.2. List the current stage
            self.STAGES_EVOL.append(self.PROBLEM)

            # 5.3. List the time steps
            if not self.breakTimeLoop:
                self.dts.append(float(self.dt))

            # 5.4. Printing informations on time step
            self.prt_step_log()

            # 5.5. Writing the results respecting the freq. of output
            self.write_step()

            # 5.6. Increase time step and current time
            self.nt += 1
            self.t += float(self.dt)



            # 6. Break the time stepping loop if this is final time step
            if self.breakTimeLoop:
                if self.terminateStatus != 'FAIL':
                    self.terminateReason = "SUCCESS: End of time stepping loop"
                    self.prt_sep(char='|')
                    self.prt_center_line(self.terminateReason)
                    self.prt_sep(char='|')

                else:
                    self.prt_sep(char='|')
                    self.prt_center_line(self.terminateReason)
                    self.prt_sep(char='|')
                # 10.1 Write the solution of the final step
                # (i.e.: by pass n_freq)
                self.write_step(final_step=True)
                break

            # 7. Verify if new step size complies with the step size rules
            self.verifyStepsize()

        # Calculation of the CPU time
        time_delta = datetime.now() - startTime

        # Print the simulation time
        msg_sim_time = f'Simulation time: {str(time_delta)}'
        self.prt_center_line(msg_sim_time)
        if self.terminateStatus != 'FAIL':
            self.prt_end_stats()

    def update_time_dependent_functions(self, t_val):
        '''
        Update all the time dependent functions.

        Parameters
        ----------
        t_val : float
            Time value to update the functions with.

        Returns
        -------
        None.
        '''
        for f in self.f_evol:
            for boundary in self.f_evol[f]:
                if 'evol' in self.f_evol[f][boundary].keys():
                    current_val = self.f_evol[f][boundary]['evol'](t_val)
                    self.f_evol[f][boundary]['exp'].val = current_val

    def write_step(self, final_step=False):
        '''
        Write the results of the time step.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''
        if (self.nt % self.freq_out == 0) or final_step:
            self.filex_Pg.write(self.Pg_n, self.t)
            self.filex_Pc.write(self.Pc_n, self.t)
            self.filex_T.write(self.T_n, self.t)
            self.filex_G.write(self.G_n, self.t)


    def update_vars_n(self, u):
        '''
        Updates the variables of the previous time step.

        Parameters
        ----------
        u : Function
            Function to update the previous time step with.

        Returns
        -------
        None.
        '''

        (Pg, Pc, T, G) = u.split(deepcopy=True)
        self.Pg_n.vector()[:] = Pg.vector()
        self.Pc_n.vector()[:] = Pc.vector()
        self.T_n.vector()[:] = T.vector()
        self.G_n.vector()[:] = G.vector()
        self.T_max.vector()[:] = self.max_values(self.T_max, self.T_n)

    def max_values(self, x, y):
        max_vals = np.max([x.vector()[:], y.vector()[:]], axis=0)
        return max_vals

    def update_vars(self, u):
        '''
        Updates the primary variables the current time step.

        Parameters
        ----------
        u : Function
            Function to update the current time step with.

        Returns
        -------
        None.
        '''

        (self.Pg, self.Pc, self.T, self.G) = u.split(True)

    def write_output(self):
        '''
        Converting into numpy arrays to save the files as .npz files.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        # Cast lists into numpy arrays
        dts = array(self.dts)
        STAGES_EVOL = array(self.STAGES_EVOL)

        # Round the current dt to use as a case identifier
        dt = round(float(self.dt), 2)

        # Creating the .npz files
        dir_output = thc_model_dir + '/results' + self.dir_output
        savez(path.join(dir_output, 'time_s_dt_' + str(dt)),
              dts=dts)
        savez(path.join(dir_output, 'stages_dt_' + str(dt)),
              stages=STAGES_EVOL)

    def run(self):
        '''
        Function that gathers the functions for running the simulation.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        self.initiate_mesh()

        self.initiate_boundary_markers()

        self.generate_function_spaces()

        self.generate_functions()

        self.generate_IC_functions()

        # Start with first stage problem and case
        self.PROBLEM = self.stages_cfg['stage_1']['name']
        self.CASE = self.stages_cfg['stage_1']['case']

        print('|--- Generating Boundary Conditions:')
        self.define_total_residual(STAGE=self.CASE)
        self.generate_BCS(STAGE=self.CASE)

        print('|--- Defining Variational Problem:')
        self.create_variational_problem_and_solver()

        print('|--- Preparing Outputs:')
        self.prepare_outputs()

        signal(SIGINT, self.userExit)

        # Wait two seconds for user to see print output        
        sleep(2)

        print('|--- Running Simulation:')
        self.simulation()

        self.write_output()

    def prt_sep(self, char):
        '''
        Prints a separation line.

        Parameters
        ----------
        char : string
            String to be used as delimiter.

        Returns
        -------
        None.
        '''

        sep_line = 53 * char
        print(f'        | {sep_line:^54}|')

    def prt_center_line(self, msg):
        '''
        Prints the message in the center of the line.

        Parameters
        ----------
        msg : string
            Message to be printed.

        Returns
        -------
        None.
        '''

        print(f'        | {msg:^54}|')

    def prt_three_col_line(self, val1_name, val2_name, val3_name,
                           val1, val2, val3, unit1, unit2, unit3,
                           fmt1='.2f', fmt2='.2f', fmt3='.2f'):
        '''
        Prints the information of 3 values in 3 columns
        with different units.

        Parameters
        ----------
        val1_name : string
            Name of the first value to be printed.

        val2_name : string
            Name of the second value to be printed.

        val3_name : string
            Name of the third value to be printed.

        val1 : float, int, string
            First value to be printed.

        val2 : float, int, string
            Second value to be printed.

        val3 : float, int, string
            Third value to be printed.

        unit1 : string
            Unit of the first value to be printed.

        unit2 : string
            Unit of the second value to be printed.

        unit3 : string
            Unit of the third value to be printed.

        Returns
        -------
        None.
        '''

        line_len = 64
        pre_amb_len = 10
        val1_name_len = len(val1_name) + 1
        val2_name_len = len(val2_name) + 1
        val3_name_len = len(val2_name) + 1
        val1_len = 8
        val2_len = 6
        val3_len = 6

        unit1_len = len(unit1)
        unit2_len = len(unit2)
        unit3_len = len(unit3)

        sep_1_len = 2
        sep_2_len = 2
        sep_1_2_len = 1
        sep_1_2 = ' ' * sep_1_2_len
        sep_2_3_len = 1
        sep_2_3 = ' ' * sep_2_3_len
        end_len = line_len - sum([pre_amb_len, val1_name_len, val2_name_len,
                                  val3_name_len, val1_len, val2_len, val3_len,
                                  unit1_len, unit2_len, unit3_len, sep_1_len,
                                  sep_1_2_len, sep_2_len, sep_2_3_len]) - 9
        end = ' ' * end_len
        print((f'        | '
               f'{val1_name}: {val1:{val1_len}{fmt1}} {unit1}{sep_1_2}| '
               f' {val2_name}: {val2:{val2_len}{fmt2}} {unit2}{sep_2_3}| '
               f'{val3_name}: {val3:{val3_len}{fmt3}} {unit3} {end}|'))

    def prt_two_col_line(self, val1_name, val2_name, val1, val2, unit,
                         val1_len=13, val2_len=13, half_line_len=36,
                         fmt1='.2f', fmt2='.2f', end_space=2):
        '''
        Prints the information of values 1 and 2 in 2 columns
        with same unit.

        Parameters
        ----------
        val1_name : string
            Name of the first value to be printed.

        val2_name : string
            Name of the second value to be printed.

        val1 : float, int, string
            First value to be printed.

        val2 : float, int, string
            Second value to be printed.

        unit : string
            Unit of values to be printed.

        val1_len : int
            Lenght of the first value to be printed.

        val2_len : int
            Lenght of the second value to be printed.

        half_line_len : int
            Unit of values to be printed.

        fmt1 : str, optional
            Format specifier of the first value.

        fmt2 : str, optional
            Format specifier of the second value.

        Returns
        -------
        None.
        '''

        line_len = 64
        pre_amb_len = 10
        val1_name_len = len(val1_name) + 1
        val2_name_len = len(val2_name) + 1
        unit_len = len(unit)
        sep_1_len = 2
        sep_2_len = 2
        sep_unit_len = 1
        sep_1_2_len = half_line_len - sum([pre_amb_len, val1_name_len,
                                           val1_len, sep_1_len, unit_len])
        sep_1_2 = ' ' * sep_1_2_len
        end_len = line_len - sum([pre_amb_len, val1_name_len, val2_name_len,
                                  val1_len, val2_len, 2 * unit_len,
                                  sep_1_len, sep_1_2_len, sep_2_len,
                                  sep_unit_len]) - end_space
        end = ' ' * end_len
        print((f'        | '
               f'{val1_name}: {val1:{val1_len}{fmt1}} {unit}{sep_1_2}| '
               f'{val2_name}: {val2:{val2_len}{fmt2}} {unit} {end}|'))

    def prt_step_log(self, char='-'):
        '''
        Prints the information on the current time step.

        Parameters
        ----------
        char : string, optional
            String to be used as delimiter.

        Returns
        -------
        None.
        '''

        # Calculation of min and max values of primary vars
        Pg_max = round(max(self.Pg_n.vector()[:]), 3)
        Pg_min = round(min(self.Pg_n.vector()[:]), 3)

        Pc_max = round(max(self.Pc_n.vector()[:]) / 1e6, 3)
        Pc_min = round(min(self.Pc_n.vector()[:]) / 1e6, 3)

        T_max = round(max(self.T_n.vector()[:]), 3)
        T_min = round(min(self.T_n.vector()[:]), 3)

        G_max = round(max(self.G_n.vector()[:]), 3)
        G_min = round(min(self.G_n.vector()[:]), 3)

        progress = round(self.t / self.t_total * 100, 2)
        msg_stage = 'STAGE: \"' + self.PROBLEM + '\"'

        self.prt_sep(char)
        self.prt_three_col_line('Progress', 't', 'nt', progress,
                                float(self.t / 3600), self.nt, '%', 'h', '[-]',
                                fmt3='d')
        self.prt_center_line(msg_stage)
        self.prt_two_col_line('T_min', 'T_max', T_min, T_max, 'K')
        self.prt_two_col_line('G_min', 'G_max', G_min, G_max, '[-]')
        self.prt_two_col_line('Pg_min', 'Pg_max', Pg_min, Pg_max, 'Pa')
        self.prt_two_col_line('Pc_min', 'Pc_max', Pc_min, Pc_max, 'MPa')
        msg_dt = f'Δt: {float(self.dt):6.2f}'
        self.prt_center_line(msg_dt)
        self.prt_sep(char)

    def prt_end_stats(self, char='='):
        terminateReason = "SUCCESS: End of time stepping loop"
        self.prt_sep(char='¨')
        self.prt_center_line(terminateReason)
        self.prt_sep(char='¨')

    def userExit(self, signal_received, frame):
        '''
        Saves the current data if user cacelled the simulation run.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        '''

        sep_line = 53 * ':'
        print(f'      | {sep_line:^54}|')
        self.prt_sep(char=':')
        self.prt_sep(char=':')
        print(('        | << HALTED: User Cancelled'
               ' Simulation Run (Ctrl+C) >>  |'))
        self.prt_sep(char=':')
        self.prt_sep(char=':')
        self.prt_sep(char=':')
        self.write_step(final_step=True)
        self.write_output()
        print('\n')

        sys.exit(0)


# Class for creating log files
class Logger(object):
    def __init__(self, dir_log):
        self.terminal = sys.stdout
        self.log = open(dir_log, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class thc_model_axisymmetric(thc_model_core):
    def define_total_residual(self, STAGE=None, DEBUG=True):
        '''
        Create the total residual for the current stage.

        Parameters
        ----------
        STAGE : str
            The current stage of the simulation. If None, starts with
            the first stage.

        DEBUG : bool
            If True, print information about where each boundary
            condition is added.

        Returns
        -------
        None.
        '''

        if STAGE is None:
            STAGE = self.stages_cfg['stage_1']['case']

        # Substitute the ds measure with a new measure w. the marked boundaries
        ds = Measure("ds", domain=self.mesh, subdomain_data=self.boundaries)
        self.ds = ds
        r = Expression('x[0]', degree=1)
        # Test functions of V
        v_1, v_2, v_3, v_4 = TestFunctions(self.V)
        # Create local variables just for readability's sake
        (Pg, Pc, T, G) = (self.Pg, self.Pc, self.T, self.G)
        (Pg_n, Pc_n, T_n, G_n) = (self.Pg_n, self.Pc_n, self.T_n, self.G_n)
        T_max = self.T_max
        dt = self.dt

        def dFdTppo(T_n, T):
            dFdTppo_vals = conditional(lt(T, T_n), 0, dFdT(T))
            return dFdTppo_vals

        dGamma_tilde_dt = ((1 - F(T)) * dGammadt(Pg_n, Pc_n, T_n, G_n)
                           - G_n * dFdTppo(T_n, T) * ((T - T_n) / dt))
        dt = self.dt


        # "Dry Air Equation"
        MBA = C_gg(Pg_n, Pc_n, T_n, G_n) * ((Pg - Pg_n) / dt) * 2 * np.pi * r * v_1 * dx
        MBA += C_gc(Pg_n, Pc_n, T_n, G_n) * ((Pc - Pc_n) / dt) * 2 * np.pi * r * v_1 * dx
        MBA += C_gt(Pg_n, Pc_n, T_n, G_n) * ((T - T_n) / dt) * 2 * np.pi * r * v_1 * dx
        MBA += inner(K_gg(Pg_n, Pc_n, T_n, G_n) * grad(Pg), grad(v_1)) * 2 * np.pi * r * dx
        MBA += inner(K_gc(Pg_n, Pc_n, T_n, G_n) * grad(Pc), grad(v_1)) * 2 * np.pi * r * dx
        MBA += inner(K_gt(Pg_n, Pc_n, T_n, G_n) * grad(T), grad(v_1)) * 2 * np.pi * r * dx
        MBA += - F_g(Pg_n, Pc_n, T_n, G_n) * 2 * np.pi * r * dGamma_tilde_dt * v_1 * dx

        def Pg_neumann_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pg
                Neumann BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal MBA, DEBUG, v_1, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pg Neumann Condition on the '
                       f'{boundary.capitalize()}'))

            # The value of fixed air flux applied at this boundary
            q_bar_a = entry['q_bar_a_val']

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            MBA += - q_bar_a * v_1 * ds(marker)

        def Pg_robin_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pg
                Robin BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal MBA, DEBUG, Pg_n, Pc_n, T_n, v_1, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pg Convection on the '
                       f'{boundary.capitalize()}'))

            # The air density defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['rho_a_inf_val']):
                self.f_evol['rho_a_inf'][boundary] = {}
                (self.f_evol['rho_a_inf']
                            [boundary]
                            ['evol']) = entry['rho_a_inf_val']
                (self.f_evol['rho_a_inf']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=entry['rho_a_inf_val'](self.t))

            else:
                rho_a_val = entry['rho_a_inf_val']
                self.f_evol['rho_a_inf'][boundary] = {}
                (self.f_evol['rho_a_inf']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=rho_a_val)

            # The mass transfer coefficient defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['h_g_val']):
                self.f_evol['h_g'][boundary] = {}
                h_g = entry['h_g_val']
                (self.f_evol['h_g']
                            [boundary]
                            ['evol']) = h_g
                (self.f_evol['h_g']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=h_g(self.t))

            else:
                self.f_evol['h_g'][boundary] = {}
                h_g = entry['h_g_val']
                (self.f_evol['h_g']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=h_g)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            MBA += (self.f_evol['h_g'][boundary]['exp']
                    * drho_adPg(Pg_n, Pc_n, T_n)
                    * (Pg - Pg_n) * v_1 * ds(marker))
            MBA += (self.f_evol['h_g'][boundary]['exp']
                    * drho_adPc(Pg_n, Pc_n, T_n)
                    * (Pc - Pc_n) * v_1 * ds(marker))
            MBA += (self.f_evol['h_g'][boundary]['exp']
                    * drho_adT(Pg_n, Pc_n, T_n)
                    * (T - T_n) * v_1 * ds(marker))
            MBA += (self.f_evol['h_g'][boundary]['exp']
                    * (rho_a(Pg_n, Pc_n, T_n)
                       - (self.f_evol['rho_a_inf']
                                     [boundary]
                                     ['exp'])) * v_1 * ds(marker))

        # Add Neumman or Robin BC for each boundary on the configuration dict
        for boundary in self.Pg_BC_cfg[STAGE]:
            entry = self.Pg_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'neumann':
                Pg_neumann_BC_parser(boundary, entry)
            elif entry['condition'] == 'robin':
                Pg_robin_BC_parser(boundary, entry)

        # "Water Equation"
        MBH = C_cc(Pg_n, Pc_n, T_n, G_n) * ((Pc - Pc_n) / dt)  * 2 * np.pi * r * v_2 * dx
        # MBH += C_cg(Pg_n, Pc_n, T_n, G_n) * ((Pg - Pg_n) / dt)  * 2 * np.pi * r * v_2 * dx
        MBH += C_ct(Pg_n, Pc_n, T_n, G_n) * ((T - T_n) / dt)  * 2 * np.pi * r * v_2 * dx
        MBH += inner(K_cc(Pg_n, Pc_n, T_n, G_n) * grad(Pc), grad(v_2))  * 2 * np.pi * r * dx
        MBH += inner(K_cg(Pg_n, Pc_n, T_n, G_n) * grad(Pg), grad(v_2))  * 2 * np.pi * r * dx
        MBH += inner(K_ct(Pg_n, Pc_n, T_n, G_n) * grad(T), grad(v_2))  * 2 * np.pi * r * dx
        MBH += - F_c(Pg_n, Pc_n, T_n, G_n) * dGamma_tilde_dt  * 2 * np.pi * r * v_2 * dx

        def Pc_neumann_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pc
                Neumann BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal MBH, DEBUG, v_2, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pc Neumann Condition on the '
                       f'{boundary.capitalize()}'))

            # The value of fixed liquid and vapor flux applied at this boundary
            q_bar_l = entry['q_bar_l_val']
            q_bar_v = entry['q_bar_v_val']

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            MBH += - (q_bar_l + q_bar_v) * v_2 * ds(marker)

        def Pc_robin_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pc
                Robin BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal MBH, DEBUG, Pg_n, Pc_n, T_n, v_2, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding Pc Convection on the '
                       f'{boundary.capitalize()}'))

            # The vapor density defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['rho_v_inf_val']):
                self.f_evol['rho_v_inf'][boundary] = {}

                (self.f_evol['rho_v_inf']
                            [boundary]
                            ['evol']) = entry['rho_v_inf_val']
                (self.f_evol['rho_v_inf']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=entry['rho_v_inf_val'](self.t))

            else:
                self.f_evol['rho_v_inf'][boundary] = {}
                (self.f_evol['rho_v_inf']
                            [boundary]
                            ['val']) = entry['rho_v_inf_val']
                (self.f_evol['rho_v_inf']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=entry['rho_v_inf_val'])

            # The mass transfer coefficient defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['h_g_val']):
                h_g = entry['h_g_val']
                self.f_evol['h_g'][boundary] = {}
                (self.f_evol['h_g']
                            [boundary]
                            ['evol']) = h_g
                (self.f_evol['h_g']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=h_g(self.t))

            else:
                h_g = entry['h_g_val']
                self.f_evol['h_g'][boundary] = {}
                (self.f_evol['h_g']
                            [boundary]
                            ['val']) = entry['h_g_val']
                (self.f_evol['h_g']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=h_g)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            MBH += (self.f_evol['h_g'][boundary]['exp']
                    * drho_vdPg(Pg_n, Pc_n, T_n)
                    * (Pg - Pg_n) * v_2 * ds(marker))
            MBH += (self.f_evol['h_g'][boundary]['exp']
                    * drho_vdPc(Pg_n, Pc_n, T_n)
                    * (Pc - Pc_n) * v_2 * ds(marker))
            MBH += (self.f_evol['h_g'][boundary]['exp']
                    * drho_vdT(Pg_n, Pc_n, T_n)
                    * (T - T_n) * v_2 * ds(marker))
            MBH += (self.f_evol['h_g'][boundary]['exp']
                    * (rho_v(Pg_n, Pc_n, T_n)
                       - (self.f_evol['rho_v_inf']
                                     [boundary]
                                     ['exp'])) * v_2 * ds(marker))

        # Add Neumman or Robin BC for each boundary on the configuration dict
        for boundary in self.Pc_BC_cfg[STAGE]:
            entry = self.Pc_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'neumann':
                Pc_neumann_BC_parser(boundary, entry)
            elif entry['condition'] == 'robin':
                Pc_robin_BC_parser(boundary, entry)

        # Energy conservation equation
        ECE = C_tt(Pg_n, Pc_n, T_n, G_n) * ((T - T_n) / dt)  * 2 * np.pi * r * v_3 * dx
        ECE += C_tc(Pg_n, Pc_n, T_n, G_n) * ((Pc - Pc_n) / dt)  * 2 * np.pi * r * v_3 * dx
        # ECE += C_tg = 0
        ECE += inner(K_tt(Pg_n, Pc_n, T_n, G_n) * grad(T), grad(v_3))  * 2 * np.pi * r * dx
        # ECE += K_tt_gradPg(Pg_n, Pc_n, T_n, G_n)  * 2 * np.pi * r * v_3 * dx
        # ECE += K_tt_gradPc(Pg_n, Pc_n, T_n, G_n)  * 2 * np.pi * r * v_3 * dx
        ECE += inner(K_tc(Pg_n, Pc_n, T_n, G_n) * grad(Pc), grad(v_3))  * 2 * np.pi * r * dx
        ECE += inner(K_tg(Pg_n, Pc_n, T_n, G_n) * grad(Pg), grad(v_3))  * 2 * np.pi * r * dx
        ECE += - F_t(Pg_n, Pc_n, T_n, G_n) * dGamma_tilde_dt  * 2 * np.pi * r * v_3 * dx

        def T_neumann_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for T
                Neumann BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal ECE, DEBUG, v_3, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding T Neumann Condition on the '
                       f'{boundary.capitalize()}'))

            # The value of fixed heat flux applied at this boundary
            q_bar_T = entry['q_bar_T_val']

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            ECE += - q_bar_T * v_3 * ds(marker)

        def T_robin_conv_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for T
                Robin Convection BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal ECE, DEBUG, Pg_n, Pc_n, T_n, v_3, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding T Convection on the '
                       f'{boundary.capitalize()}'))

            # The temperature at the far field defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['T_inf_val']):
                T_inf = entry['T_inf_val']
                self.f_evol['T_conv_inf'][boundary] = {}
                (self.f_evol['T_conv_inf']
                            [boundary]
                            ['evol']) = T_inf
                (self.f_evol['T_conv_inf']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=T_inf(self.t))

            else:
                T_inf = entry['T_inf_val']
                self.f_evol['T_conv_inf'][boundary] = {}
                (self.f_evol['T_conv_inf']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=T_inf)

            # The heat transfer coefficient defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['h_T_val']):
                h_T = entry['h_T_val']
                self.f_evol['h_T'][boundary] = {}
                (self.f_evol['h_T']
                            [boundary]
                            ['evol']) = h_T
                (self.f_evol['h_T']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=h_T(self.t))

            else:
                h_T = entry['h_T_val']
                self.f_evol['h_T'][boundary] = {}
                (self.f_evol['h_T']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=h_T)

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            ECE += ((self.f_evol['h_T'][boundary]['exp']
                     * (T - self.f_evol['T_conv_inf'][boundary]['exp']))
                    * v_3 * ds(marker))

        def T_robin_rad_BC_parser(boundary, entry):
            '''
            Create the total residual for the current stage.

            Parameters
            ----------
            boundary : str
                The boundary to parse the BC information from.

            entry : dict
                The dictionary holding the BC information for Pg
                Robin BC.

            Returns
            -------
            None.
            '''

            # Load variables from the `define_total_residual` function
            nonlocal ECE, DEBUG, Pg_n, Pc_n, T_n, v_3, ds

            # If debug in mode print information of BC setting
            if DEBUG:
                print((f'   |--- Adding T Radiation on the '
                       f'{boundary.capitalize()}'))

            # The temperature at the far field defined at the current boundary
            # If it is a function, store it in the time dependent
            # function dictionary (`f_evol`) and create an `Expression`
            # also stored at this dictionary
            if callable(entry['T_inf_val']):
                T_inf = entry['T_inf_val']
                self.f_evol['T_rad_inf'][boundary] = {}
                (self.f_evol['T_rad_inf']
                            [boundary]
                            ['evol']) = T_inf
                (self.f_evol['T_rad_inf']
                            [boundary]
                            ['exp']) = Expression('val',
                                                  degree=2,
                                                  val=T_inf(self.t))

            else:
                T_inf = entry['T_inf_val']
                self.f_evol['T_rad_inf'][boundary] = {}
                (self.f_evol['T_rad_inf']
                            [boundary]
                            ['exp']) = Expression('val', degree=2,
                                                  val=T_inf)

            # Thermal total emissivity of the boundary
            epsilon = entry['epsilon_val']

            # FEniCS Marker identifying such boundary
            marker = entry['marker']

            # Updating residual
            ECE += ((epsilon * sigma_SB
                     * (T_n**3 * T
                        - self.f_evol['T_rad_inf'][boundary]['exp']**4))
                    * v_3 * ds(marker))

        # Add Neumman or Robin Convection, or Robin Radiation BC
        # for each boundary on the configuration dict
        for boundary in self.T_BC_cfg[STAGE]:
            entry = self.T_BC_cfg[STAGE][boundary]
            if entry['condition'] == 'neumann':
                T_neumann_BC_parser(boundary, entry)
            if 'conv' in entry['condition']:
                T_robin_conv_BC_parser(boundary, entry)
            if 'rad' in entry['condition']:
                T_robin_rad_BC_parser(boundary, entry)

        # Hydration Degree Equation
        GE = (- dGammadt(Pg_n, Pc_n, T_n, G_n) + ((G - G_n) / dt))  * 2 * np.pi * r * v_4 * dx
        Res = (MBH + MBA + ECE + GE)
        self.Res = Res

