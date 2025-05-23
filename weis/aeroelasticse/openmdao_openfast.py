import numpy as np
import pandas as pd
import os
import shutil
import sys
import copy
import glob
import logging
import pickle
from pathlib import Path
from scipy.interpolate                      import PchipInterpolator
from openmdao.api                           import ExplicitComponent
from openmdao.utils.mpi import MPI
from wisdem.commonse import NFREQ
from wisdem.commonse.cylinder_member import get_nfull
import wisdem.commonse.utilities              as util
from wisdem.rotorse.rotor_power             import eval_unsteady
from openfast_io.FAST_writer         import InputWriter_OpenFAST
from openfast_io.FAST_reader         import InputReader_OpenFAST
import weis.aeroelasticse.runFAST_pywrapper as fastwrap
from openfast_io.FAST_post         import FAST_IO_timeseries
from wisdem.floatingse.floating_frame import NULL, NNODES_MAX, NELEM_MAX
from weis.dlc_driver.dlc_generator    import DLCGenerator
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from functools import partial
from weis.aeroelasticse.LinearFAST import LinearFAST
from weis.control.LinearModel import LinearTurbineModel, LinearControlModel
from openfast_io import FileTools
from openfast_io.turbsim_file   import TurbSimFile
from weis.aeroelasticse.utils import OLAFParams, generate_wind_files
from rosco.toolbox import control_interface as ROSCO_ci
from pCrunch import AeroelasticOutput, FatigueParams
from weis.control.dtqp_wrapper          import dtqp_wrapper
from openfast_io.StC_defaults        import default_StC_vt
from weis.aeroelasticse.CaseGen_General import case_naming
from wisdem.inputs import load_yaml, write_yaml

logger = logging.getLogger("wisdem/weis")

weis_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def make_coarse_grid(s_grid, diam):

    s_coarse = [s_grid[0]]
    slope = np.diff(diam) / np.diff(s_grid)
    for k in range(slope.size-1):
        if np.abs(slope[k]-slope[k+1]) > 1e-2:
            s_coarse.append(s_grid[k+1])
    s_coarse.append(s_grid[-1])
    return np.array(s_coarse)

    
class FASTLoadCases(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')
        self.options.declare('opt_options')

    def setup(self):
        modopt = self.options['modeling_options']
        rotorse_options  = modopt['WISDEM']['RotorSE']
        mat_init_options = modopt['materials']

        self.modopt_dir = os.path.dirname(self.options['modeling_options']['fname_input_modeling'])

        self.n_blades      = modopt['assembly']['number_of_blades']
        self.n_span        = n_span    = rotorse_options['n_span']
        self.n_pc          = n_pc      = rotorse_options['n_pc']

        # Environmental Conditions needed regardless of where model comes from
        self.add_input('V_cutin',     val=0.0, units='m/s',      desc='Minimum wind speed where turbine operates (cut-in)')
        self.add_input('V_cutout',    val=0.0, units='m/s',      desc='Maximum wind speed where turbine operates (cut-out)')
        self.add_input('Vrated',      val=0.0, units='m/s',      desc='rated wind speed')
        self.add_input('hub_height',                val=0.0, units='m', desc='hub height')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('turbine_class',    val='I', desc='IEC turbine class')
        self.add_input('Rtip',              val=0.0, units='m', desc='dimensional radius of tip')
        self.add_input('shearExp',    val=0.0,                   desc='shear exponent')
        self.add_input('lifetime', val=25.0, units='yr', desc='Turbine design lifetime')
        self.add_input(
            "water_depth", val=0.0, units="m", desc="Water depth for analysis.  Values > 0 mean offshore"
            )

        if not self.options['modeling_options']['OpenFAST']['from_openfast']:
            self.n_pitch       = n_pitch   = rotorse_options['n_pitch_perf_surfaces']
            self.n_tsr         = n_tsr     = rotorse_options['n_tsr_perf_surfaces']
            self.n_U           = n_U       = rotorse_options['n_U_perf_surfaces']
            self.n_mat         = n_mat    = mat_init_options['n_mat']
            self.n_layers      = n_layers = rotorse_options['n_layers']

            self.n_xy          = n_xy      = rotorse_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
            self.n_aoa         = n_aoa     = rotorse_options['n_aoa']# Number of angle of attacks
            self.n_Re          = n_Re      = rotorse_options['n_Re'] # Number of Reynolds, so far hard set at 1
            self.n_tab         = n_tab     = rotorse_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1
            
            self.te_ss_var       = rotorse_options['te_ss']
            self.te_ps_var       = rotorse_options['te_ps']
            self.spar_cap_ss_var = rotorse_options['spar_cap_ss']
            self.spar_cap_ps_var = rotorse_options['spar_cap_ps']

            n_freq_blade = int(rotorse_options['n_freq']/2)
            n_pc         = int(rotorse_options['n_pc'])

            self.n_xy          = n_xy      = rotorse_options['n_xy'] # Number of coordinate points to describe the airfoil geometry
            self.n_aoa         = n_aoa     = rotorse_options['n_aoa']# Number of angle of attacks
            self.n_Re          = n_Re      = rotorse_options['n_Re'] # Number of Reynolds, so far hard set at 1
            self.n_tab         = n_tab     = rotorse_options['n_tab']# Number of tabulated data. For distributed aerodynamic control this could be > 1

            self.te_ss_var       = rotorse_options['te_ss']
            self.te_ps_var       = rotorse_options['te_ps']
            self.spar_cap_ss_var = rotorse_options['spar_cap_ss']
            self.spar_cap_ps_var = rotorse_options['spar_cap_ps']

            # ElastoDyn and BeamDyn Inputs
            self.add_input(
                'r', 
                val=np.zeros(n_span), 
                units='m', 
                desc='radial positions. r[0] should be the hub location \
                while r[-1] should be the blade tip. Any number \
                of locations can be specified between these in ascending order.',
            )
            self.add_input(
                'le_location', 
                val=np.zeros(n_span), 
                desc='Leading-edge positions from a reference blade axis \
                usually blade pitch axis). Locations are normalized by the \
                local chord length. Positive in -x direction for airfoil-aligned coordinate system',
            )
            self.add_input(
                "blade:EIxx",
                val=np.zeros(n_span),
                units="N*m**2",
                desc="Section lag (edgewise) bending stiffness about the XE axis, using the convention of WISDEM solver PreComp.",
            )
            self.add_input("blade:EIyy",
                val=np.zeros(n_span),
                units="N*m**2",
                desc="Section flap bending stiffness about the YE axis, using the convention of WISDEM solver PreComp.",
            )
            self.add_input(
                "blade:rhoA", 
                val=np.zeros(n_span), 
                units="kg/m", 
                desc="Section mass per unit length, using the convention of WISDEM solver PreComp.",
            )
            self.add_input(
                "blade:K",
                val=np.zeros((n_span,6,6)), 
                desc="Stiffness matrix at the center of the windIO reference axes."
            )
            self.add_input(
                "blade:I",
                val=np.zeros((n_span,6,6)), 
                desc="Inertia matrix at the center of the windIO reference axes."
            )
            self.add_input(
                'blade:flap_mode_shapes', 
                val=np.zeros((n_freq_blade,5)), 
                desc='6-degree polynomial coefficients of mode shapes in the flap direction (x^2..x^6, no linear or constant term)',
            )
            self.add_input(
                'blade:edge_mode_shapes', 
                val=np.zeros((n_freq_blade,5)), 
                desc='6-degree polynomial coefficients of mode shapes in the edge direction (x^2..x^6, no linear or constant term)',
            )
            
            self.add_input('gearbox_efficiency', val=1.0, desc='Gearbox efficiency')
            self.add_input('gearbox_ratio', val=1.0, desc='Gearbox ratio')
            self.add_input('platform_displacement', val=1.0, desc='Volumetric platform displacement', units='m**3')

            # ServoDyn Inputs
            self.add_input('generator_efficiency',   val=1.0,              desc='Generator efficiency')
            self.add_input('max_pitch_rate',         val=0.0,        units='deg/s',          desc='Maximum allowed blade pitch rate')

            # StC or TMD inputs; structural control and tuned mass dampers

            # tower properties
            n_height_tow = modopt['WISDEM']['TowerSE']['n_height']
            n_full_tow   = get_nfull(n_height_tow, nref=modopt['WISDEM']['TowerSE']['n_refine'])
            n_freq_tower = int(NFREQ/2)
            self.add_input('fore_aft_modes',   val=np.zeros((n_freq_tower,5)),               desc='6-degree polynomial coefficients of mode shapes in the flap direction (x^2..x^6, no linear or constant term)')
            self.add_input('side_side_modes',  val=np.zeros((n_freq_tower,5)),               desc='6-degree polynomial coefficients of mode shapes in the edge direction (x^2..x^6, no linear or constant term)')
            self.add_input('mass_den',         val=np.zeros(n_height_tow-1),         units='kg/m',   desc='sectional mass per unit length')
            self.add_input('foreaft_stff',     val=np.zeros(n_height_tow-1),         units='N*m**2', desc='sectional fore-aft bending stiffness per unit length about the Y_E elastic axis')
            self.add_input('sideside_stff',    val=np.zeros(n_height_tow-1),         units='N*m**2', desc='sectional side-side bending stiffness per unit length about the Y_E elastic axis')
            self.add_input('tor_stff',    val=np.zeros(n_height_tow-1),         units='N*m**2', desc='torsional stiffness per unit length about the Y_E elastic axis')
            self.add_input('tor_freq',    val=0.0,         units='Hz', desc='First tower torsional frequency')
            self.add_input('tower_outer_diameter', val=np.zeros(n_height_tow),   units='m',      desc='cylinder diameter at corresponding locations')
            self.add_input('tower_z', val=np.zeros(n_height_tow),   units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
            self.add_input('tower_z_full', val=np.zeros(n_full_tow),   units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
            self.add_input('tower_height',              val=0.0, units='m', desc='tower height from the tower base')
            self.add_input('tower_base_height',         val=0.0, units='m', desc='tower base height from the ground or mean sea level')
            self.add_input('tower_cd',         val=np.zeros(n_height_tow),                   desc='drag coefficients along tower height at corresponding locations')
            self.add_input("tower_I_base", np.zeros(6), units="kg*m**2", desc="tower moments of inertia at the tower base")

            # These next ones are needed for SubDyn
            n_height_mon = n_full_mon = 0
            if modopt['flags']['offshore']:
                self.add_input('transition_piece_mass', val=0.0, units='kg')
                self.add_input('transition_piece_I', val=np.zeros(3), units='kg*m**2')
                
                if modopt['flags']['monopile']:
                    n_height_mon = modopt['WISDEM']['FixedBottomSE']['n_height']
                    n_full_mon   = get_nfull(n_height_mon, nref=modopt['WISDEM']['FixedBottomSE']['n_refine'])
                    self.add_input('monopile_z', val=np.zeros(n_height_mon),   units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
                    self.add_input('monopile_z_full', val=np.zeros(n_full_mon),   units='m',      desc='z-coordinates of tower and monopile used in TowerSE')
                    self.add_input('monopile_outer_diameter', val=np.zeros(n_height_mon),   units='m',      desc='cylinder diameter at corresponding locations')
                    self.add_input('monopile_wall_thickness', val=np.zeros(n_height_mon-1), units='m')
                    self.add_input('monopile_E', val=np.zeros(n_height_mon-1), units='Pa')
                    self.add_input('monopile_G', val=np.zeros(n_height_mon-1), units='Pa')
                    self.add_input('monopile_rho', val=np.zeros(n_height_mon-1), units='kg/m**3')
                    self.add_input('gravity_foundation_mass', val=0.0, units='kg')
                    self.add_input('gravity_foundation_I', val=np.zeros(3), units='kg*m**2')
            monlen = max(0, n_height_mon-1)
            monlen_full = max(0, n_full_mon-1)

            # DriveSE quantities
            self.add_input('hub_system_cm',   val=np.zeros(3),             units='m',  desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
            self.add_input('hub_system_I',    val=np.zeros(6),             units='kg*m**2', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
            self.add_input('hub_system_mass', val=0.0,                     units='kg', desc='mass of hub system')
            self.add_input('above_yaw_mass',  val=0.0, units='kg', desc='Mass of the nacelle above the yaw system')
            self.add_input('yaw_mass',        val=0.0, units='kg', desc='Mass of yaw system')
            self.add_input('rna_I_TT',       val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the rna [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about the tower top')
            self.add_input('nacelle_cm',      val=np.zeros(3), units='m', desc='Center of mass of the component in [x,y,z] for an arbitrary coordinate system')
            self.add_input('nacelle_I_TT',       val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the nacelle [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] about the tower top')
            self.add_input('distance_tt_hub', val=0.0,         units='m',   desc='Vertical distance from tower top plane to hub flange')
            self.add_input('twr2shaft',       val=0.0,         units='m',   desc='Vertical distance from tower top plane to shaft start')
            self.add_input('GenIner',         val=0.0,         units='kg*m**2',   desc='Moments of inertia for the generator about high speed shaft')
            self.add_input('drivetrain_spring_constant',         val=0.0,         units='N*m/rad',   desc='Moments of inertia for the generator about high speed shaft')
            self.add_input("drivetrain_damping_coefficient", 0.0, units="N*m*s/rad", desc='Equivalent damping coefficient for the drivetrain system')

            # AeroDyn Inputs
            self.add_input('ref_axis_blade',    val=np.zeros((n_span,3)),units='m',   desc='2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.')
            self.add_input('chord',             val=np.zeros(n_span), units='m', desc='chord at airfoil locations')
            self.add_input('theta',             val=np.zeros(n_span), units='deg', desc='twist at airfoil locations')
            self.add_input('rthick',            val=np.zeros(n_span), desc='relative thickness of airfoil distribution')
            self.add_input('ac',                val=np.zeros(n_span), desc='aerodynamic center of airfoil distribution')
            self.add_input('pitch_axis',        val=np.zeros(n_span), desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
            self.add_input('Rhub',              val=0.0, units='m', desc='dimensional radius of hub')
            self.add_input('airfoils_cl',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='lift coefficients, spanwise')
            self.add_input('airfoils_cd',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='drag coefficients, spanwise')
            self.add_input('airfoils_cm',       val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc='moment coefficients, spanwise')
            self.add_input('airfoils_aoa',      val=np.zeros((n_aoa)), units='deg', desc='angle of attack grid for polars')
            self.add_input('airfoils_Re',       val=np.zeros((n_Re)), desc='Reynolds numbers of polars')
            self.add_input('airfoils_UserProp',     val=np.zeros((n_span, n_Re, n_tab)), units='deg',desc='Airfoil control paremeter (i.e. flap angle)')

            # Airfoil coordinates
            self.add_input('coord_xy_interp',   val=np.zeros((n_span, n_xy, 2)),              desc='3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.')

            # Floating platform inputs
            self.add_input("transition_node", np.zeros(3), units="m")
            self.add_input("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
            self.add_input("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
            self.add_input("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
            self.add_input("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m")
            self.add_input("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
            self.add_input("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
            self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
            self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
            self.add_input("platform_total_center_of_mass", np.zeros(3), units="m")
            self.add_input("platform_mass", 0.0, units="kg")
            self.add_input("platform_I_total", np.zeros(6), units="kg*m**2")

            if modopt['flags']["floating"]:
                n_member = modopt["floating"]["members"]["n_members"]
                for k in range(n_member):
                    n_height_mem = modopt["floating"]["members"]["n_height"][k]
                    self.add_input(f"member{k}:joint1", np.zeros(3), units="m")
                    self.add_input(f"member{k}:joint2", np.zeros(3), units="m")
                    self.add_input(f"member{k}:s", np.zeros(n_height_mem))
                    self.add_input(f"member{k}:s_ghost1", 0.0)
                    self.add_input(f"member{k}:s_ghost2", 0.0)
                    self.add_input(f"member{k}:outer_diameter", np.zeros(n_height_mem), units="m")
                    self.add_input(f"member{k}:wall_thickness", np.zeros(n_height_mem-1), units="m")

            # Turbine level inputs
            self.add_discrete_input('rotor_orientation',val='upwind', desc='Rotor orientation, either upwind or downwind.')
            self.add_input('control_ratedPower',        val=0.,  units='W',    desc='machine power rating')
            self.add_input('control_maxOmega',          val=0.0, units='rpm',  desc='maximum allowed rotor rotation speed')
            self.add_input('control_maxTS',             val=0.0, units='m/s',  desc='maximum allowed blade tip speed')
            self.add_input('cone',             val=0.0, units='deg',   desc='Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.')
            self.add_input('tilt',             val=0.0, units='deg',   desc='Nacelle uptilt angle. A standard machine has positive values.')
            self.add_input('overhang',         val=0.0, units='m',     desc='Horizontal distance from tower top to hub center.')

            # Initial conditions
            self.add_input('U', val=np.zeros(n_pc), units='m/s', desc='wind speeds')
            self.add_input('Omega', val=np.zeros(n_pc), units='rpm', desc='rotation speeds to run')
            self.add_input('pitch', val=np.zeros(n_pc), units='deg', desc='pitch angles to run')
            self.add_input("Ct_aero", val=np.zeros(n_pc), desc="rotor aerodynamic thrust coefficient")

            # Cp-Ct-Cq surfaces
            self.add_input('Cp_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero power coefficient')
            self.add_input('Ct_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero thrust coefficient')
            self.add_input('Cq_aero_table', val=np.zeros((n_tsr, n_pitch, n_U)), desc='Table of aero torque coefficient')
            self.add_input('pitch_vector',  val=np.zeros(n_pitch), units='deg',  desc='Pitch vector used')
            self.add_input('tsr_vector',    val=np.zeros(n_tsr),                 desc='TSR vector used')
            self.add_input('U_vector',      val=np.zeros(n_U),     units='m/s',  desc='Wind speed vector used')

            # Environmental conditions
            self.add_input('V_R25',       val=0.0, units='m/s',      desc='region 2.5 transition wind speed')
            self.add_input('Vgust',       val=0.0, units='m/s',      desc='gust wind speed')
            self.add_input('V_extreme1',  val=0.0, units='m/s',      desc='IEC extreme wind speed at hub height for a 1-year retunr period')
            self.add_input('V_extreme50', val=0.0, units='m/s',      desc='IEC extreme wind speed at hub height for a 50-year retunr period')
            self.add_input('V_mean_iec',  val=0.0, units='m/s',      desc='IEC mean wind for turbulence class')
            
            self.add_input('rho',         val=0.0, units='kg/m**3',  desc='density of air')
            self.add_input('mu',          val=0.0, units='kg/(m*s)', desc='dynamic viscosity of air')
            self.add_input('speed_sound_air',  val=340.,    units='m/s',        desc='Speed of sound in air.')
            self.add_input('rho_water',   val=0.0, units='kg/m**3',  desc='density of water')
            self.add_input('mu_water',    val=0.0, units='kg/(m*s)', desc='dynamic viscosity of water')
            self.add_input('beta_wave',    val=0.0, units='deg', desc='Incident wave propagation heading direction')
            self.add_input('Hsig_wave',    val=0.0, units='m', desc='Significant wave height of incident waves')
            self.add_input('Tsig_wave',    val=0.0, units='s', desc='Peak-spectral period of incident waves')

            # Blade composite material properties (used for fatigue analysis)
            self.add_input('gamma_f',      val=1.35,                             desc='safety factor on loads')
            self.add_input('gamma_m',      val=1.1,                              desc='safety factor on materials')
            self.add_input('E',            val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
            self.add_input('Xt',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.')
            self.add_input('Xc',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.')
            self.add_input('m',            val=np.zeros([n_mat]),                desc='2D array of the S-N fatigue slope exponent for the materials')

            # Blade composit layup info (used for fatigue analysis)
            self.add_input('sc_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
            self.add_input('sc_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
            self.add_input('te_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
            self.add_input('te_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
            self.add_discrete_input('definition_layer', val=np.zeros(n_layers),  desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')
            # self.add_discrete_input('layer_name',       val=n_layers * [''],     desc='1D array of the names of the layers modeled in the blade structure.')
            # self.add_discrete_input('layer_web',        val=n_layers * [''],     desc='1D array of the names of the webs the layer is associated to. If the layer is on the outer profile this entry can simply stay empty.')
            # self.add_discrete_input('layer_mat',        val=n_layers * [''],     desc='1D array of the names of the materials of each layer modeled in the blade structure.')
            self.layer_name = rotorse_options['layer_name']

            # MoorDyn inputs
            mooropt = modopt["mooring"]
            if self.options["modeling_options"]["flags"]["mooring"]:
                n_nodes = mooropt["n_nodes"]
                n_lines = mooropt["n_lines"]
                self.add_input("line_diameter", val=np.zeros(n_lines), units="m")
                self.add_input("line_mass_density", val=np.zeros(n_lines), units="kg/m")
                self.add_input("line_stiffness", val=np.zeros(n_lines), units="N")
                self.add_input("line_transverse_added_mass", val=np.zeros(n_lines), units="kg/m")
                self.add_input("line_tangential_added_mass", val=np.zeros(n_lines), units="kg/m")
                self.add_input("line_transverse_drag", val=np.zeros(n_lines))
                self.add_input("line_tangential_drag", val=np.zeros(n_lines))
                self.add_input("nodes_location_full", val=np.zeros((n_nodes, 3)), units="m")
                self.add_input("nodes_mass", val=np.zeros(n_nodes), units="kg")
                self.add_input("nodes_volume", val=np.zeros(n_nodes), units="m**3")
                self.add_input("nodes_added_mass", val=np.zeros(n_nodes))
                self.add_input("nodes_drag_area", val=np.zeros(n_nodes), units="m**2")
                self.add_input("unstretched_length", val=np.zeros(n_lines), units="m")
                self.add_discrete_input("node_names", val=[""] * n_nodes)

            # Inputs required for fatigue processing
            self.add_input('blade_sparU_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparU_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparU_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
            self.add_input('blade_sparL_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparL_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_sparL_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
            self.add_input('blade_teU_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teU_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teU_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
            self.add_input('blade_teL_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teL_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('blade_teL_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
            self.add_input('blade_root_sparU_load2stress',   val=np.ones(6), units="m**2",  desc='Blade root upper spar cap coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_root_sparL_load2stress',   val=np.ones(6), units="m**2",  desc='Blade root lower spar cap coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_maxc_teU_load2stress',   val=np.ones(6), units="m**2",  desc='Blade max chord upper trailing edge coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('blade_maxc_teL_load2stress',   val=np.ones(6), units="m**2",  desc='Blade max chord lower trailing edge coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('lss_wohlerexp',   val=1.0,   desc='Low speed shaft Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('lss_wohlerA',     val=1.0,   desc='Low speed shaft parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('lss_ultstress',   val=1.0, units="Pa",   desc='Low speed shaft Ultimate stress for material')
            self.add_input('lss_axial_load2stress',   val=np.ones(6), units="m**2",  desc='Low speed shaft coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('lss_shear_load2stress',   val=np.ones(6), units="m**2",  desc='Low speed shaft coefficient between shear load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('tower_wohlerexp',   val=np.ones(n_height_tow-1),   desc='Tower Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('tower_wohlerA',     val=np.ones(n_height_tow-1),   desc='Tower parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('tower_ultstress',   val=np.ones(n_height_tow-1), units="Pa",   desc='Tower ultimate stress for material')
            self.add_input('tower_axial_load2stress',   val=np.ones([n_height_tow-1,6]), units="m**2",  desc='Tower coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('tower_shear_load2stress',   val=np.ones([n_height_tow-1,6]), units="m**2",  desc='Tower coefficient between shear load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('monopile_wohlerexp',   val=np.ones(monlen),   desc='Tower Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
            self.add_input('monopile_wohlerA',     val=np.ones(monlen),   desc='Tower parameter, A, in S/N curve S=A*N^-(1/m)')
            self.add_input('monopile_ultstress',   val=np.ones(monlen), units="Pa",   desc='Tower ultimate stress for material')
            self.add_input('monopile_axial_load2stress',   val=np.ones([monlen,6]), units="m**2",  desc='Tower coefficient between axial load and stress S=C^T [Fx-z;Mx-z]')
            self.add_input('monopile_shear_load2stress',   val=np.ones([monlen,6]), units="m**2",  desc='Tower coefficient between shear load and stress S=C^T [Fx-z;Mx-z]')
        

        # TMD params
        if self.options['modeling_options']['flags']['TMDs']:
            n_TMDs = self.options['modeling_options']['TMDs']['n_TMDs']
            self.add_input('TMD_mass',         val=np.zeros(n_TMDs), units='kg',         desc='TMD Mass')
            self.add_input('TMD_stiffness',    val=np.zeros(n_TMDs), units='N/m',        desc='TMD Stiffnes')
            self.add_input('TMD_damping',      val=np.zeros(n_TMDs), units='N/(m/s)',    desc='TMD Damping')

        # DLC options
        n_ws_aep = np.max([1,modopt['DLC_driver']['n_ws_aep']])

        # OpenFAST options
        OFmgmt = modopt['General']['openfast_configuration']
        self.model_only = OFmgmt['model_only']
        FAST_directory_base = OFmgmt['OF_run_dir']
        # If the path is relative, make it an absolute path to modeling options file
        if not os.path.isabs(FAST_directory_base):
            FAST_directory_base = os.path.join(os.path.dirname(modopt['fname_input_modeling']), FAST_directory_base)
        # Flag to clear OpenFAST run folder. Use it only if disk space is an issue
        self.clean_FAST_directory = False
        self.FAST_InputFile = OFmgmt['OF_run_fst']
        # File naming changes whether in MPI or not
        if MPI:
            rank    = MPI.COMM_WORLD.Get_rank()
            self.FAST_runDirectory = os.path.join(FAST_directory_base,'rank_%000d'%int(rank))
            self.FAST_namingOut = self.FAST_InputFile+'_%000d'%int(rank)
        else:
            self.FAST_runDirectory = FAST_directory_base
            self.FAST_namingOut = self.FAST_InputFile
        self.wind_directory = os.path.join(self.FAST_runDirectory, 'wind')
        if not os.path.exists(self.FAST_runDirectory):
            os.makedirs(self.FAST_runDirectory, exist_ok=True)
        if not os.path.exists(self.wind_directory):
            os.makedirs(self.wind_directory, exist_ok=True)
        # Number of cores used outside of MPI. If larger than 1, the multiprocessing module is called
        self.cores = OFmgmt['cores']
        self.case = {}
        self.channels = {}
        self.mpi_run = False
        if 'mpi_run' in OFmgmt.keys():
            self.mpi_run         = OFmgmt['mpi_run']
            if self.mpi_run:
                self.mpi_comm_map_down   = OFmgmt['mpi_comm_map_down']

        # User-defined FAST library/executable
        if OFmgmt['FAST_exe'] != 'none':
            if os.path.isabs(OFmgmt['FAST_exe']):
                self.FAST_exe_user = OFmgmt['FAST_exe']
            else:
                self.FAST_exe_user = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                             OFmgmt['FAST_exe'])
        else:
            self.FAST_exe_user = None

        if OFmgmt['FAST_lib'] != 'none':
            if os.path.isabs(OFmgmt['FAST_lib']):
                self.FAST_lib_user = OFmgmt['FAST_lib']
            else:
                self.FAST_lib_user = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                             OFmgmt['FAST_lib'])
        else:
            self.FAST_lib_user = None

        if OFmgmt['turbsim_exe'] != 'none':
            if os.path.isabs(OFmgmt['turbsim_exe']):
                self.turbsim_exe = OFmgmt['turbsim_exe']
            else:
                self.turbsim_exe = os.path.join(os.path.dirname(self.options['modeling_options']['fname_input_modeling']),
                                             OFmgmt['turbsim_exe'])
        else:
            self.turbsim_exe = shutil.which('turbsim')
        

        # Rotor power outputs
        self.add_output('V_out', val=np.zeros(n_ws_aep), units='m/s', desc='wind speed vector from the OF simulations')
        self.add_output('P_out', val=np.zeros(n_ws_aep), units='W', desc='rotor electrical power')
        self.add_output('Cp_out', val=np.zeros(n_ws_aep), desc='rotor aero power coefficient')
        self.add_output('Ct_out', val=np.zeros(n_ws_aep), desc='rotor aero thrust coefficient')
        self.add_output('Omega_out', val=np.zeros(n_ws_aep), units='rpm', desc='rotation speeds to run')
        self.add_output('pitch_out', val=np.zeros(n_ws_aep), units='deg', desc='pitch angles to run')
        self.add_output('AEP', val=0.0, units='kW*h', desc='annual energy production reconstructed from the openfast simulations')

        self.add_output('My_std',      val=0.0,            units='N*m',  desc='standard deviation of blade root flap bending moment in out-of-plane direction')
        self.add_output('flp1_std',    val=0.0,            units='deg',  desc='standard deviation of trailing-edge flap angle')

        self.add_output('rated_V',     val=0.0,            units='m/s',  desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,            units='rpm',  desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,            units='deg',  desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,            units='N',    desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,            units='N*m',  desc='rotor aerodynamic torque at rated')

        self.add_output('loads_r',      val=np.zeros(n_span), units='m', desc='radial positions along blade going toward tip')
        self.add_output('loads_Px',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_output('loads_Py',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_output('loads_Pz',     val=np.zeros(n_span), units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_output('loads_Omega',  val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_output('loads_pitch',  val=0.0, units='deg', desc='pitch angle')
        self.add_output('loads_azimuth', val=0.0, units='deg', desc='azimuthal angle')

        # Control outputs
        self.add_output('rotor_overspeed',  val=0.0, desc='Maximum percent overspeed of the rotor during all OpenFAST simulations')  # is this over a set of sims?
        self.add_output('max_nac_accel',    val=0.0, units='m/s**2', desc='Maximum nacelle acceleration magnitude all OpenFAST simulations')  # is this over a set of sims?
        self.add_output('avg_pitch_travel',    val=0.0, units='deg/s', desc='Average pitch travel')  # is this over a set of sims?
        self.add_output('pitch_duty_cycle',    val=0.0, units='deg/s', desc='Number of pitch direction changes')  # is this over a set of sims?
        self.add_output('max_pitch_rate_sim',    val=0.0, units='deg/s', desc='Maximum pitch command rate over all simulations')  # is this over a set of sims?

        # Blade outputs
        self.add_output('max_TipDxc', val=0.0, units='m', desc='Maximum of channel TipDxc, i.e. out of plane tip deflection. For upwind rotors, the max value is tower the tower')
        self.add_output('max_RootMyb', val=0.0, units='kN*m', desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root flapwise moment')
        self.add_output('max_RootMyc', val=0.0, units='kN*m', desc='Maximum of the signals RootMyb1, RootMyb2, ... across all n blades representing the maximum blade root out of plane moment')
        self.add_output('max_RootMzb', val=0.0, units='kN*m', desc='Maximum of the signals RootMzb1, RootMzb2, ... across all n blades representing the maximum blade root torsional moment')
        self.add_output('DEL_RootMyb', val=0.0, units='kN*m', desc='damage equivalent load of blade root flap bending moment in out-of-plane direction')
        self.add_output('max_aoa', val=np.zeros(n_span), units='deg', desc='maxima of the angles of attack distributed along blade span')
        self.add_output('std_aoa', val=np.zeros(n_span), units='deg', desc='standard deviation of the angles of attack distributed along blade span')
        self.add_output('mean_aoa', val=np.zeros(n_span), units='deg', desc='mean of the angles of attack distributed along blade span')
        # Blade loads corresponding to maximum blade tip deflection
        self.add_output('blade_maxTD_Mx', val=np.zeros(n_span), units='kN*m', desc='distributed moment around blade-aligned x-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_My', val=np.zeros(n_span), units='kN*m', desc='distributed moment around blade-aligned y-axis corresponding to maximum blade tip deflection')
        self.add_output('blade_maxTD_Fz', val=np.zeros(n_span), units='kN', desc='distributed force in blade-aligned z-direction corresponding to maximum blade tip deflection')

        # Hub outputs
        self.add_output('hub_Fxyz', val=np.zeros(3), units='kN', desc = 'Maximum hub forces in the non rotating frame')
        self.add_output('hub_Mxyz', val=np.zeros(3), units='kN*m', desc = 'Maximum hub moments in the non rotating frame')
        self.add_output('hub_Fxyz_aero', val=np.zeros(3), units='N', desc = 'Aero-only maximum hub forces in the non rotating frame')
        self.add_output('hub_Mxyz_aero', val=np.zeros(3), units='N*m', desc = 'Aero-only maximum hub moments in the non rotating frame')

        self.add_output('max_TwrBsMyt',val=0.0, units='kN*m', desc='maximum of tower base bending moment in fore-aft direction')
        self.add_output('max_TwrBsMyt_ratio',val=0.0,  desc='ratio of maximum of tower base bending moment in fore-aft direction to maximum allowable bending moment')
        self.add_output('DEL_TwrBsMyt',val=0.0, units='kN*m', desc='damage equivalent load of tower base bending moment in fore-aft direction')
        self.add_output('DEL_TwrBsMyt_ratio',val=0.0, desc='ratio of damage equivalent load of tower base bending moment in fore-aft direction to maximum allowable bending moment')
        
        # Tower outputs
        if not self.options['modeling_options']['OpenFAST']['from_openfast']:
            self.add_output('tower_maxMy_Fx', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned x-direction corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_Fy', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned y-direction corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_Fz', val=np.zeros(n_full_tow-1), units='kN', desc='distributed force in tower-aligned z-direction corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_Mx', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_My', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')
            self.add_output('tower_maxMy_Mz', val=np.zeros(n_full_tow-1), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to maximum fore-aft moment at tower base')

            # Monopile outputs
            self.add_output('max_M1N1MKye',val=0.0, units='kN*m', desc='maximum of My moment of member 1 at node 1 (base of the monopile)')
            self.add_output('monopile_maxMy_Fx', val=np.zeros(monlen_full), units='kN', desc='distributed force in monopile-aligned x-direction corresponding to max_M1N1MKye')
            self.add_output('monopile_maxMy_Fy', val=np.zeros(monlen_full), units='kN', desc='distributed force in monopile-aligned y-direction corresponding to max_M1N1MKye')
            self.add_output('monopile_maxMy_Fz', val=np.zeros(monlen_full), units='kN', desc='distributed force in monopile-aligned z-direction corresponding to max_M1N1MKye')
            self.add_output('monopile_maxMy_Mx', val=np.zeros(monlen_full), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
            self.add_output('monopile_maxMy_My', val=np.zeros(monlen_full), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')
            self.add_output('monopile_maxMy_Mz', val=np.zeros(monlen_full), units='kN*m', desc='distributed moment around tower-aligned x-axis corresponding to max_M1N1MKye')

        # Floating outputs
        self.add_output('Max_PtfmPitch', val=0.0, desc='Maximum platform pitch angle over a set of OpenFAST simulations')
        self.add_output('Std_PtfmPitch', val=0.0, units='deg', desc='standard deviation of platform pitch angle')
        self.add_output('Max_Offset', val=0.0, units='m', desc='Maximum distance in surge/sway direction')

        # Fatigue output
        self.add_output('damage_blade_root_sparU', val=0.0, desc="Miner's rule cumulative damage to upper spar cap at blade root")
        self.add_output('damage_blade_root_sparL', val=0.0, desc="Miner's rule cumulative damage to lower spar cap at blade root")
        self.add_output('damage_blade_maxc_teU', val=0.0, desc="Miner's rule cumulative damage to upper trailing edge at blade max chord")
        self.add_output('damage_blade_maxc_teL', val=0.0, desc="Miner's rule cumulative damage to lower trailing edge at blade max chord")
        self.add_output('damage_lss', val=0.0, desc="Miner's rule cumulative damage to low speed shaft at hub attachment")
        self.add_output('damage_tower_base', val=0.0, desc="Miner's rule cumulative damage at tower base")
        self.add_output('damage_monopile_base', val=0.0, desc="Miner's rule cumulative damage at monopile base")

        # Simulation output
        self.add_output('openfast_failed', val=0.0, desc="Numerical value for whether any openfast runs failed. 0 if false, 2 if true")
        
        # Open loop to closed loop error
        if self.options['modeling_options']['OL2CL']['flag']:
            self.add_output('OL2CL_pitch', val=0.0, desc="Open loop to closed loop avarege error")

        self.add_discrete_output('fst_vt_out', val={})
        self.add_discrete_output('ts_out_dir', val={})

        # Iteration counter for openfast calls. Initialize at -1 so 0 after first call
        self.of_inumber = -1
        self.sim_idx = -1

        if modopt['OpenFAST_Linear']['flag']:
            if MPI:
                rank = MPI.COMM_WORLD.Get_rank()
                lin_pkl_dir = os.path.join(self.options['opt_options']['general']['folder_output'], 'lin', 'rank_{}'.format(rank))
                if not os.path.exists(lin_pkl_dir):
                    os.makedirs(lin_pkl_dir, exist_ok=True)
                self.lin_pkl_file_name = os.path.join(lin_pkl_dir, 'ABCD_matrices.pkl')
            else:
                lin_pkl_dir = os.path.join(self.options['opt_options']['general']['folder_output'], 'lin')
                self.lin_pkl_file_name = os.path.join(lin_pkl_dir, 'ABCD_matrices.pkl')

            self.ABCD_list = []
            path = '.'.join(self.lin_pkl_file_name.split('.')[:-1])
            os.makedirs(path, exist_ok=True)

            with open(self.lin_pkl_file_name, 'wb') as handle:
                pickle.dump(self.ABCD_list, handle)
            
            self.lin_idx = 0

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        modopt = self.options['modeling_options']
        sys.stdout.flush()

        if modopt['OpenFAST_Linear']['flag']:
            self.sim_idx += 1
            ABCD = {
                'sim_idx' : self.sim_idx,
                'A' : None,
                'B' : None,
                'C' : None,
                'D' : None,
                'x_ops':None,
                'u_ops':None,
                'y_ops':None,
                'u_h':None,
                'omega_rpm' : None,
                'DescCntrlInpt' : None,
                'DescStates' : None,
                'DescOutput' : None,
                'StateDerivOrder' : None,
                'ind_fast_inps' : None,
                'ind_fast_outs' : None,
                'AEP':None
                }
            with open(self.lin_pkl_file_name, 'rb') as handle:
                ABCD_list = pickle.load(handle)

            ABCD_list.append(ABCD)
            self.ABCD_list = ABCD_list

            with open(self.lin_pkl_file_name, 'wb') as handle:
                pickle.dump(ABCD_list, handle)

        fst_vt = self.init_FAST_model()

        if not modopt['OpenFAST']['from_openfast']:
            fst_vt = self.update_FAST_model(fst_vt, inputs, discrete_inputs)
        else:
            fast_reader = InputReader_OpenFAST()
            fast_reader.FAST_InputFile  = modopt['OpenFAST']['openfast_file']   # FAST input file (ext=.fst)
            fast_reader.FAST_directory  = os.path.join(self.modopt_dir,modopt['OpenFAST']['openfast_dir'])   # Path to fst directory files
            fast_reader.path2dll            = modopt['General']['openfast_configuration']['path2dll']   # Path to dll file
            fast_reader.execute()
            fst_vt = fast_reader.fst_vt

            # Fix TwrTI: WEIS modeling options have it as a single value...
            if not isinstance(fst_vt['AeroDyn']['TwrTI'],list):
                fst_vt['AeroDyn']['TwrTI'] = [fst_vt['AeroDyn']['TwrTI']] * len(fst_vt['AeroDyn']['TwrElev'])
            if not isinstance(fst_vt['AeroDyn']['TwrCb'],list):
                fst_vt['AeroDyn']['TwrCb'] = [fst_vt['AeroDyn']['TwrCb']] * len(fst_vt['AeroDyn']['TwrElev'])

            # Fix AddF0: Should be a n x 1 array (list of lists):
            if fst_vt['HydroDyn'] and fst_vt['HydroDyn']['NBodyMod'] == 1:
                fst_vt['HydroDyn']['AddF0'] = [[F0] for F0 in fst_vt['HydroDyn']['AddF0']]

            if modopt['ROSCO']['flag']:
                fst_vt['DISCON_in'] = modopt['General']['openfast_configuration']['fst_vt']['DISCON_in']
                
        #  Allow user-defined OpenFAST options to override WISDEM-generated ones
        #  Re-load modeling options without defaults to learn only what needs to change, has already been validated when first loaded
        modopts_no_defaults = load_yaml(self.options['modeling_options']['fname_input_modeling'])

        # Backwards compatibility with Level3
        if 'Level3' in modopts_no_defaults:
            if 'OpenFAST' not in modopts_no_defaults:
                modopts_no_defaults['OpenFAST'] = {}
            modopts_no_defaults['OpenFAST'].update(modopts_no_defaults['Level3'])

        fst_vt = self.load_FAST_model_opts(fst_vt,modopts_no_defaults)
                
        if self.model_only == True:
            # Write input OF files, but do not run OF
            fst_vt['Fst']['TMax'] = 10.
            fst_vt['Fst']['TStart'] = 0.
            self.write_FAST(fst_vt)
        else:
            # Write OF model and run
            case_list, case_name, dlc_generator  = self.run_FAST(inputs, discrete_inputs, fst_vt)

            # Set up linear turbine model
            if modopt['OpenFAST_Linear']['flag']:
                try: 
                    LinearTurbine = LinearTurbineModel(
                    self.FAST_runDirectory,
                    self.lin_case_name,
                    nlin=modopt['OpenFAST_Linear']['linearization']['NLinTimes'],
                    reduceControls=True
                    )
                except FileNotFoundError as e:
                    logger.warning('FileNotFoundError: {} {}'.format(e.strerror, e.filename))
                    return

                # Save linearizations
                logger.warning('Saving ABCD matrices!')
                ABCD = {
                    'sim_idx' : self.sim_idx,
                    'A' : LinearTurbine.A_ops,
                    'B' : LinearTurbine.B_ops,
                    'C' : LinearTurbine.C_ops,
                    'D' : LinearTurbine.D_ops,
                    'x_ops':LinearTurbine.x_ops,
                    'u_ops':LinearTurbine.u_ops,
                    'y_ops':LinearTurbine.y_ops,
                    'u_h':LinearTurbine.u_h,
                    'omega_rpm' : LinearTurbine.omega_rpm,
                    'DescCntrlInpt' : LinearTurbine.DescCntrlInpt,
                    'DescStates' : LinearTurbine.DescStates,
                    'DescOutput' : LinearTurbine.DescOutput,
                    'StateDerivOrder' : LinearTurbine.StateDerivOrder,
                    'ind_fast_inps' : LinearTurbine.ind_fast_inps,
                    'ind_fast_outs' : LinearTurbine.ind_fast_outs,
                    }
                with open(self.lin_pkl_file_name, 'rb') as handle:
                    ABCD_list = pickle.load(handle)

                ABCD_list[self.sim_idx] = ABCD

                with open(self.lin_pkl_file_name, 'wb') as handle:
                    pickle.dump(ABCD_list, handle)
                    
                lin_files = glob.glob(os.path.join(self.FAST_runDirectory, '*.lin'))
                
                dest = os.path.join(self.FAST_runDirectory, f'copied_lin_files_{self.lin_idx}')
                Path(dest).mkdir(parents=True, exist_ok=True)
                for file in lin_files:
                    shutil.copy2(file, dest)
                self.lin_idx += 1

                # Shorten output names from linearization output to one like OpenFAST openfast output
                # This depends on how openfast sets up the linearization output names and may break if that is changed
                OutList     = [out_name.split()[1][:-1] for out_name in LinearTurbine.DescOutput]
                OutOps      = {}
                for i_out, out in enumerate(OutList):
                    OutOps[out] = LinearTurbine.y_ops[i_out,:]

                # save to yaml, might want in analysis outputs
                FileTools.save_yaml(
                    self.FAST_runDirectory,
                    'OutOps.yaml',OutOps)

                # Set up Level 2 disturbance (simulation or DTQP)
                if modopt['OpenFAST_Linear']['simulation']['flag'] or modopt['OpenFAST_Linear']['DTQP']['flag']:
                    # Extract disturbance(s)
                    level2_disturbance = []
                    for case in case_list:
                        ts_file     = TurbSimFile(case[('InflowWind','FileName_BTS')])
                        ts_file.compute_rot_avg(fst_vt['ElastoDyn']['TipRad'])
                        u_h         = ts_file['rot_avg'][0,:]
                        tt          = ts_file['t']
                        level2_disturbance.append({'Time':tt, 'Wind': u_h})

                # Run linear simulation:

                # Get case list, wind inputs should have already been generated
                if modopt['OpenFAST_Linear']['simulation']['flag']:
            
                    if modopt['OpenFAST_Linear']['DTQP']['flag']:
                        raise Exception('Only DTQP or simulation flag can be set to true in OpenFAST_Linear modeling options')

                    # This is going to use the last discon_in file of the linearization set as the simulation file
                    # Currently fine because openfast is executed (or not executed if overwrite=False) after the file writing
                    if 'DLL_InFile' in self.fst_vt['ServoDyn']:     # if using file inputs
                        discon_in_file = os.path.join(self.FAST_runDirectory, self.fst_vt['ServoDyn']['DLL_InFile'])
                    else:       # if using fst_vt inputs from openfast_openmdao
                        discon_in_file = os.path.join(self.FAST_runDirectory, self.lin_case_name[0] + '_DISCON.IN')

                    lib_name = modopt['General']['openfast_configuration']['path2dll']

                    ct = []
                    for i_dist, dist in enumerate(level2_disturbance):
                        sim_name = 'l2_sim_{}'.format(i_dist)
                        controller_int = ROSCO_ci.ControllerInterface(
                            lib_name,
                            param_filename=discon_in_file,
                            DT=1/80,        # modelling input?
                            sim_name = os.path.join(self.FAST_runDirectory,sim_name)
                            )

                        l2_out, _, P_op = LinearTurbine.solve(dist,Plot=False,controller=controller_int)

                        output = AeroelasticOutput(l2_out, dlc=sim_name, magnitude_channels=self.magnitude_channels)
                        self.cruncher.add_output(output)
                        ct.append(l2_out)

                        output.to_df().to_pickle(os.path.join(self.FAST_runDirectory,sim_name+'.p'))

                elif modopt['OpenFAST_Linear']['DTQP']['flag']:

                    dtqp_wrapper(
                        LinearTurbine, 
                        level2_disturbance, 
                        self.options['opt_options'], 
                        self.options['modeling_options'], 
                        self.fst_vt, 
                        self.cruncher, 
                        self.magnitude_channels, 
                        self.FAST_runDirectory
                    )

            # Post process regardless of level
            self.post_process(case_list, dlc_generator, inputs, discrete_inputs, outputs, discrete_outputs)
            
            # Save AEP value to linear pickle file
            if modopt['OpenFAST_Linear']['flag']:
                with open(self.lin_pkl_file_name, 'rb') as handle:
                        ABCD_list = pickle.load(handle)

                ABCD_list[self.sim_idx]['AEP'] = outputs['AEP']

                with open(self.lin_pkl_file_name, 'wb') as handle:
                    pickle.dump(ABCD_list, handle)
        
        # delete run directory. not recommended for most cases, use for large parallelization problems where disk storage will otherwise fill up
        if self.clean_FAST_directory:
            try:
                shutil.rmtree(self.FAST_runDirectory)
            except:
                logger.warning('Failed to delete directory: %s'%self.FAST_runDirectory)

    def init_FAST_model(self):

        modopt = self.options['modeling_options']
        fst_vt = modopt['General']['openfast_configuration']['fst_vt']

        # Main .fst file`
        fst_vt['Fst'] = {}
        fst_vt['ElastoDyn'] = {}
        fst_vt['ElastoDynBlade'] = {}
        fst_vt['ElastoDynTower'] = {}
        fst_vt['AeroDyn'] = {}
        fst_vt['AeroDynBlade'] = {}
        fst_vt['ServoDyn'] = {}
        fst_vt['InflowWind'] = {}
        fst_vt['SubDyn'] = {}
        fst_vt['SeaState'] = {}
        fst_vt['HydroDyn'] = {}
        fst_vt['MoorDyn'] = {}
        fst_vt['MAP'] = {}
        fst_vt['BeamDyn'] = {}
        # fst_vt['BeamDynBlade'] = {}
        
        # List of structural controllers
        fst_vt['TStC'] = {}; fst_vt['TStC'] = []
        fst_vt['SStC'] = {}; fst_vt['SStC'] = []
        fst_vt['NStC'] = {}; fst_vt['NStC'] = []
        fst_vt['BStC'] = {}; fst_vt['BStC'] = []

        fst_vt = self.load_FAST_model_opts(fst_vt)

        return fst_vt

    def load_FAST_model_opts(self,fst_vt,modeling_options={}):
        # Can provide own modeling options, used when we don't want to use default OpenFAST options
        ititializing = False
        if not modeling_options:
            modeling_options = self.options['modeling_options']
            ititializing = True

        if 'simulation' in modeling_options['OpenFAST']:
            for key in modeling_options['OpenFAST']['simulation']:
                fst_vt['Fst'][key] = modeling_options['OpenFAST']['simulation'][key]

        if 'ElastoDyn' in modeling_options['OpenFAST']:
            for key in modeling_options['OpenFAST']['ElastoDyn']:
                fst_vt['ElastoDyn'][key] = modeling_options['OpenFAST']['ElastoDyn'][key]
        
        if 'ElastoDynBlade' in modeling_options['OpenFAST']:
            for key in modeling_options['OpenFAST']['ElastoDynBlade']:
                fst_vt['ElastoDynBlade'][key] = modeling_options['OpenFAST']['ElastoDynBlade'][key]
        
        # if 'BeamDyn' in modeling_options['OpenFAST']:
        #     for key in modeling_options['OpenFAST']['BeamDyn']:
        #         fst_vt['BeamDyn'][key] = modeling_options['OpenFAST']['BeamDyn'][key]
        
        # if 'BeamDynBlade' in modeling_options['OpenFAST']:
        #     for key in modeling_options['OpenFAST']['BeamDynBlade']:
        #         fst_vt['BeamDynBlade'][key] = modeling_options['OpenFAST']['BeamDynBlade'][key]
        
        if 'ElastoDynTower' in modeling_options['OpenFAST']:   
            for key in modeling_options['OpenFAST']['ElastoDynTower']:
                fst_vt['ElastoDynTower'][key] = modeling_options['OpenFAST']['ElastoDynTower'][key]

        if 'AeroDyn' in modeling_options['OpenFAST']: 
            for key in modeling_options['OpenFAST']['AeroDyn']:
                if key == 'OLAF':
                    if 'OLAF' not in fst_vt['AeroDyn']:
                        fst_vt['AeroDyn']['OLAF'] = {}
                    for o_key in modeling_options['OpenFAST']['AeroDyn']['OLAF']:
                        fst_vt['AeroDyn']['OLAF'][o_key] = modeling_options['OpenFAST']['AeroDyn']['OLAF'][o_key]
                    if not ititializing:
                        continue

                fst_vt['AeroDyn'][key] = copy.copy(modeling_options['OpenFAST']['AeroDyn'][key])

        if 'InflowWind' in modeling_options['OpenFAST']:    
            for key in modeling_options['OpenFAST']['InflowWind']:
                fst_vt['InflowWind'][key] = modeling_options['OpenFAST']['InflowWind'][key]
            
        if 'ServoDyn' in modeling_options['OpenFAST']:    
            for key in modeling_options['OpenFAST']['ServoDyn']:
                fst_vt['ServoDyn'][key] = modeling_options['OpenFAST']['ServoDyn'][key]

        if 'SubDyn' in modeling_options['OpenFAST']:    
            for key in modeling_options['OpenFAST']['SubDyn']:
                fst_vt['SubDyn'][key] = modeling_options['OpenFAST']['SubDyn'][key]

        if 'SeaState' in modeling_options['OpenFAST']:    
            for key in modeling_options['OpenFAST']['SeaState']:
                fst_vt['SeaState'][key] = modeling_options['OpenFAST']['SeaState'][key]

        if 'HydroDyn' in modeling_options['OpenFAST']:    
            for key in modeling_options['OpenFAST']['HydroDyn']:
                # PotFile is an edge case where it is either specified by the user, and has already been made into an absolute path in gc_LoadInputs, 
                # then when we update it again here, it's made relative because that's how it's spec'd in the input
                if key == 'PotFile':
                    # When first setting up fst_vt, we need to transfer PotFile from modopts to fst_vt
                    if ititializing:
                        pass
                    # When updating PotFile, we don't want exact (relative) value from modeling option inputs
                    else:
                        continue
                fst_vt['HydroDyn'][key] = modeling_options['OpenFAST']['HydroDyn'][key]

        if 'MoorDyn' in modeling_options['OpenFAST']:    
            for key in modeling_options['OpenFAST']['MoorDyn']:
                fst_vt['MoorDyn'][key] = modeling_options['OpenFAST']['MoorDyn'][key]
        
        if 'outlist' in modeling_options['OpenFAST']:
            for key1 in modeling_options['OpenFAST']['outlist']:
                    for key2 in modeling_options['OpenFAST']['outlist'][key1]:
                        fst_vt['outlist'][key1][key2] = modeling_options['OpenFAST']['outlist'][key1][key2]
        
        if ('openfast_configuration' in modeling_options['General']) and ('path2dll' in modeling_options['General']['openfast_configuration']):
            fst_vt['ServoDyn']['DLL_FileName'] = modeling_options['General']['openfast_configuration']['path2dll']

        if fst_vt['AeroDyn']['IndToler'] == 0.:
            fst_vt['AeroDyn']['IndToler'] = 'Default'
        if fst_vt['AeroDyn']['DTAero'] == 0.:
            fst_vt['AeroDyn']['DTAero'] = 'Default'
        if 'OLAF' in fst_vt['AeroDyn'] and 'DTfvw' in fst_vt['AeroDyn']['OLAF']:
            if fst_vt['AeroDyn']['OLAF']['DTfvw'] == 0.:
                fst_vt['AeroDyn']['OLAF']['DTfvw'] = 'Default'
        else:
            fst_vt['AeroDyn']['OLAF'] = {}
        if fst_vt['ElastoDyn']['DT'] == 0.:
            fst_vt['ElastoDyn']['DT'] = 'Default'
        if fst_vt['Fst']['DT_Out'] == 0.:
            fst_vt['Fst']['DT_Out'] = 'Default'

        return fst_vt

    def update_FAST_model(self, fst_vt, inputs, discrete_inputs):

        modopt = self.options['modeling_options']

        modopts_no_defaults = load_yaml(self.options['modeling_options']['fname_input_modeling'])

        # Update fst_vt nested dictionary with data coming from WISDEM

        # Comp flags in main input
        if modopt['flags']['offshore']:
            fst_vt['Fst']['CompHydro'] = 1  # Use HydroDyn if not set in modeling inputs 
            fst_vt['Fst']['CompSeaSt'] = 1  # Use SeaDyn if not set in modeling inputs

        # If there's mooring and  CompMooring not set in modeling inputs
        if modopt["flags"]["mooring"] and not fst_vt['Fst']['CompMooring']:
            fst_vt['Fst']['CompMooring'] = 3  # Use MoorDyn    

        # Set MHK flag
        if modopt['flags']['marine_hydro']:
            if modopt['flags']['floating']:
                fst_vt['Fst']['MHK'] = 2
            else:
                fst_vt['Fst']['MHK'] = 1
        else:
            fst_vt['Fst']['MHK'] = 0
        
        # Overwrite with user input
        if 'MHK' in modopts_no_defaults['OpenFAST']['simulation']:
            fst_vt['Fst']['MHK'] = modopts_no_defaults['OpenFAST']['simulation']['MHK']

        # Update ElastoDyn
        fst_vt['ElastoDyn']['NumBl']  = self.n_blades
        fst_vt['ElastoDyn']['TipRad'] = inputs['Rtip'][0]
        fst_vt['ElastoDyn']['HubRad'] = inputs['Rhub'][0]
        if discrete_inputs['rotor_orientation'] == 'upwind':
            k = -1.
        else:
            k = 1
        fst_vt['ElastoDyn']['PreCone(1)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['PreCone(2)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['PreCone(3)'] = k*inputs['cone'][0]
        fst_vt['ElastoDyn']['ShftTilt']   = k*inputs['tilt'][0]
        fst_vt['ElastoDyn']['OverHang']   = k*inputs['overhang'][0] / np.cos(np.deg2rad(inputs['tilt'][0])) # OpenFAST defines the overhang tilted (!)
        fst_vt['ElastoDyn']['GBoxEff']    = inputs['gearbox_efficiency'][0] * 100.
        fst_vt['ElastoDyn']['GBRatio']    = inputs['gearbox_ratio'][0]

        # Update ServoDyn
        fst_vt['ServoDyn']['GenEff']       = float(inputs['generator_efficiency']/inputs['gearbox_efficiency']) * 100.
        fst_vt['ServoDyn']['PitManRat(1)'] = float(inputs['max_pitch_rate'])
        fst_vt['ServoDyn']['PitManRat(2)'] = float(inputs['max_pitch_rate'])
        fst_vt['ServoDyn']['PitManRat(3)'] = float(inputs['max_pitch_rate'])
        

        # Update ServoDyn
        fst_vt['ServoDyn']['GenEff']       = float(inputs['generator_efficiency']/inputs['gearbox_efficiency']) * 100.
        fst_vt['ServoDyn']['PitManRat(1)'] = float(inputs['max_pitch_rate'])
        fst_vt['ServoDyn']['PitManRat(2)'] = float(inputs['max_pitch_rate'])
        fst_vt['ServoDyn']['PitManRat(3)'] = float(inputs['max_pitch_rate'])
        

        # Masses and inertias from DriveSE
        fst_vt['ElastoDyn']['HubMass']   = inputs['hub_system_mass'][0]
        fst_vt['ElastoDyn']['HubIner']   = inputs['hub_system_I'][0]
        fst_vt['ElastoDyn']['HubCM']     = inputs['hub_system_cm'][0] # k*inputs['overhang'][0] - inputs['hub_system_cm'][0], but we need to solve the circular dependency in DriveSE first
        fst_vt['ElastoDyn']['NacMass']   = inputs['above_yaw_mass'][0]
        fst_vt['ElastoDyn']['YawBrMass'] = inputs['yaw_mass'][0]
        # Advice from R. Bergua, add 1/3 the tower yaw inertia here because it is activated as a lumped modal mass
        fst_vt['ElastoDyn']['NacYIner']  = inputs['nacelle_I_TT'][2] + inputs['tower_I_base'][2]/3.0
        fst_vt['ElastoDyn']['NacCMxn']   = -k*inputs['nacelle_cm'][0]
        fst_vt['ElastoDyn']['NacCMyn']   = inputs['nacelle_cm'][1]
        fst_vt['ElastoDyn']['NacCMzn']   = inputs['nacelle_cm'][2]
        tower_top_height = float(inputs['hub_height']) - float(inputs['distance_tt_hub']) # Height of tower above ground level [onshore] or MSL [offshore] (meters)
        # The Twr2Shft is just the difference between hub height, tower top height, and sin(tilt)*overhang
        fst_vt['ElastoDyn']['Twr2Shft']  = float(inputs['hub_height']) - tower_top_height - abs(fst_vt['ElastoDyn']['OverHang'])*np.sin(np.deg2rad(inputs['tilt'][0]))
        # Flip Twr2Shft if floating MHK
        if modopt['flags']['marine_hydro'] and modopt['flags']['floating']:
            fst_vt['ElastoDyn']['Twr2Shft'] *= -1
        fst_vt['ElastoDyn']['GenIner']   = float(inputs['GenIner'])

        # Mass and inertia inputs
        fst_vt['ElastoDyn']['TipMass(1)'] = 0.
        fst_vt['ElastoDyn']['TipMass(2)'] = 0.
        fst_vt['ElastoDyn']['TipMass(3)'] = 0.

        tower_base_height = max(float(inputs['tower_base_height']), float(inputs["platform_total_center_of_mass"][2]))
        fst_vt['ElastoDyn']['TowerBsHt'] = tower_base_height # Height of tower base above ground level [onshore] or MSL [offshore] (meters)
        fst_vt['ElastoDyn']['TowerHt']   = tower_top_height

        # TODO: There is some confusion on PtfmRefzt
        # DZ: based on the openfast r-tests:
        #   if this is floating, the z ref. point is 0.  Is this the reference that platform_total_center_of_mass is relative to?
        #   if fixed bottom, it's the tower base height.
        if modopt['flags']['floating']:
            fst_vt['ElastoDyn']['PtfmMass'] = float(inputs["platform_mass"])
            fst_vt['ElastoDyn']['PtfmRIner'] = float(inputs["platform_I_total"][0])
            fst_vt['ElastoDyn']['PtfmPIner'] = float(inputs["platform_I_total"][1])
            fst_vt['ElastoDyn']['PtfmYIner'] = float(inputs["platform_I_total"][2])

            fst_vt['ElastoDyn']['PtfmXYIner'] = float(inputs["platform_I_total"][3])
            fst_vt['ElastoDyn']['PtfmYZIner'] = float(inputs["platform_I_total"][5])
            fst_vt['ElastoDyn']['PtfmXZIner'] = float(inputs["platform_I_total"][4])

            fst_vt['ElastoDyn']['PtfmCMxt'] = float(inputs["platform_total_center_of_mass"][0])
            fst_vt['ElastoDyn']['PtfmCMyt'] = float(inputs["platform_total_center_of_mass"][1])
            fst_vt['ElastoDyn']['PtfmCMzt'] = float(inputs["platform_total_center_of_mass"][2])
            fst_vt['ElastoDyn']['PtfmRefzt'] = 0. # Vertical distance from the ground level [onshore] or MSL [offshore] to the platform reference point (meters)

        else:
            # Ptfm* can capture the transition piece for fixed-bottom, but we are doing that in subdyn, so only worry about getting height right
            fst_vt['ElastoDyn']['PtfmMass'] = 0.
            fst_vt['ElastoDyn']['PtfmRIner'] = 0.
            fst_vt['ElastoDyn']['PtfmPIner'] = 0.

            fst_vt['ElastoDyn']['PtfmXYIner'] = 0.
            fst_vt['ElastoDyn']['PtfmYZIner'] = 0.
            fst_vt['ElastoDyn']['PtfmXZIner'] = 0.
            # Advice from R. Bergua- Use a dummy quantity (at least 1e4) here when have fixed-bottom support
            fst_vt['ElastoDyn']['PtfmYIner'] = 1e8 if modopt['flags']['offshore'] else 0.0
            fst_vt['ElastoDyn']['PtfmCMxt'] = 0.
            fst_vt['ElastoDyn']['PtfmCMyt'] = 0.
            fst_vt['ElastoDyn']['PtfmCMzt'] = float(inputs['tower_base_height'])
            fst_vt['ElastoDyn']['PtfmRefzt'] = tower_base_height # Vertical distance from the ground level [onshore] or MSL [offshore] to the platform reference point (meters)





        # Drivetrain inputs
        fst_vt['ElastoDyn']['DTTorSpr'] = float(inputs['drivetrain_spring_constant'])
        fst_vt['ElastoDyn']['DTTorDmp'] = float(inputs['drivetrain_damping_coefficient'])

        # Update Inflowwind
        fst_vt['InflowWind']['RefHt'] = float(inputs['hub_height'])
        fst_vt['InflowWind']['RefHt_Uni'] = float(inputs['hub_height'])
        fst_vt['InflowWind']['PLexp'] = float(inputs['shearExp'])
        if fst_vt['InflowWind']['NWindVel'] == 1:
            fst_vt['InflowWind']['WindVxiList'] = ['0.']
            fst_vt['InflowWind']['WindVyiList'] = ['0.']
            fst_vt['InflowWind']['WindVziList'] = [str(float(inputs['hub_height']))]
        else:
            raise Exception('The code only supports InflowWind NWindVel == 1')

        # Update AeroDyn Tower Input File starting one station above ground to avoid error because the wind grid hits the ground        
        twr_elev  = inputs['tower_z']
        if modopt['flags']['marine_hydro']:
            twr_elev = np.flip(twr_elev)    # WISDEM flips the tower indices, flip them back
        twr_d     = inputs['tower_outer_diameter']
        twr_index = np.argmin(abs(twr_elev - np.maximum(1.0, tower_base_height)))
        cd_index  = 0
        if twr_elev[twr_index] <= 1. and not modopt['flags']['marine_hydro']:
            twr_index += 1
            cd_index  += 1

        twr_ids = np.arange(twr_index,len(twr_elev))

        fst_vt['AeroDyn']['NumTwrNds'] = len(twr_elev[twr_ids])
        fst_vt['AeroDyn']['TwrElev']   = twr_elev[twr_ids]
        fst_vt['AeroDyn']['TwrDiam']   = twr_d[twr_ids]
        fst_vt['AeroDyn']['TwrCd']     = inputs['tower_cd'][cd_index:]
        fst_vt['AeroDyn']['TwrTI']     = np.ones(len(twr_elev[twr_index:])) * fst_vt['AeroDyn']['TwrTI']
        fst_vt['AeroDyn']['TwrCb']     = np.ones(len(twr_elev[twr_index:])) * fst_vt['AeroDyn']['TwrCb']

        z_tow = twr_elev
        z_sec, _ = util.nodal2sectional(z_tow)
        sec_loc = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])
        fst_vt['ElastoDynTower']['NTwInpSt'] = len(sec_loc)
        fst_vt['ElastoDynTower']['HtFract']  = sec_loc
        fst_vt['ElastoDynTower']['TMassDen'] = inputs['mass_den']
        fst_vt['ElastoDynTower']['TwFAStif'] = inputs['foreaft_stff']
        fst_vt['ElastoDynTower']['TwSSStif'] = inputs['sideside_stff']
        fst_vt['ElastoDynTower']['TwFAM1Sh'] = inputs['fore_aft_modes'][0, :]  / sum(inputs['fore_aft_modes'][0, :])
        fst_vt['ElastoDynTower']['TwFAM2Sh'] = inputs['fore_aft_modes'][1, :]  / sum(inputs['fore_aft_modes'][1, :])
        fst_vt['ElastoDynTower']['TwSSM1Sh'] = inputs['side_side_modes'][0, :] / sum(inputs['side_side_modes'][0, :])
        fst_vt['ElastoDynTower']['TwSSM2Sh'] = inputs['side_side_modes'][1, :] / sum(inputs['side_side_modes'][1, :])

        # Catch cases where modes not calculated
        if all(np.isnan(fst_vt['ElastoDynTower']['TwFAM1Sh'])):
            fst_vt['ElastoDyn']['TwFADOF1'] = False
            fst_vt['ElastoDynTower']['TwFAM1Sh'] = [1,0,0,0,0]  # placeholder is required in OpenFAST
            print("WEIS Warning: Setting TwFADOF1 to False in ElastoDyn because mode shapes were not calculated")

        if all(np.isnan(fst_vt['ElastoDynTower']['TwFAM2Sh'])):
            fst_vt['ElastoDyn']['TwFADOF2'] = False
            fst_vt['ElastoDynTower']['TwFAM2Sh'] = [1,0,0,0,0]  # placeholder is required in OpenFAST
            print("WEIS Warning: Setting TwFADOF2 to False in ElastoDyn because mode shapes were not calculated")

        if all(np.isnan(fst_vt['ElastoDynTower']['TwSSM1Sh'])):
            fst_vt['ElastoDyn']['TwSSDOF1'] = False
            fst_vt['ElastoDynTower']['TwSSM1Sh'] = [1,0,0,0,0]  # placeholder is required in OpenFAST
            print("WEIS Warning: Setting TwSSDOF1 to False in ElastoDyn because mode shapes were not calculated")

        if all(np.isnan(fst_vt['ElastoDynTower']['TwSSM2Sh'])):
            fst_vt['ElastoDyn']['TwSSDOF2'] = False
            fst_vt['ElastoDynTower']['TwSSM2Sh'] = [1,0,0,0,0]  # placeholder is required in OpenFAST
            print("WEIS Warning: Setting TwSSDOF2 to False in ElastoDyn because mode shapes were not calculated")
        
        # Calculate yaw stiffness of tower (springs in series) and use in servodyn as yaw spring constant
        k_tow_tor = inputs['tor_stff'] / np.diff(inputs['tower_z'])
        k_tow_tor = 1.0/np.sum(1.0/k_tow_tor)
        # R. Bergua's suggestion to set the stiffness to the tower torsional stiffness and the damping to the frequency of the first tower torsional mode- easier than getting the yaw inertia right
        damp_ratio = 0.01
        f_torsion = float(inputs['tor_freq'])
        fst_vt['ServoDyn']['YawSpr'] = k_tow_tor
        if f_torsion > 0.0:
            fst_vt['ServoDyn']['YawDamp'] = damp_ratio * k_tow_tor / np.pi / f_torsion
        else:
            fst_vt['ServoDyn']['YawDamp'] = 2 * damp_ratio * np.sqrt(k_tow_tor * inputs['rna_I_TT'][2])

        # Update ElastoDyn Blade Input File
        fst_vt['ElastoDyn']['BldFile1'] = ''
        fst_vt['ElastoDyn']['BldFile2'] = ''
        fst_vt['ElastoDyn']['BldFile3'] = ''
        fst_vt['ElastoDynBlade']['NBlInpSt']   = len(inputs['r'])
        fst_vt['ElastoDynBlade']['BlFract']    = (inputs['r']-inputs['Rhub'])/(inputs['Rtip']-inputs['Rhub'])
        fst_vt['ElastoDynBlade']['BlFract'][0] = 0.
        fst_vt['ElastoDynBlade']['BlFract'][-1]= 1.
        fst_vt['ElastoDynBlade']['PitchAxis']  = inputs['le_location']
        fst_vt['ElastoDynBlade']['StrcTwst']   = inputs['theta'] # to do: structural twist is not nessessarily (nor likely to be) the same as aero twist
        fst_vt['ElastoDynBlade']['BMassDen']   = inputs['blade:rhoA']
        fst_vt['ElastoDynBlade']['FlpStff']    = inputs['blade:EIyy']
        fst_vt['ElastoDynBlade']['EdgStff']    = inputs['blade:EIxx']
        fst_vt['ElastoDynBlade']['BldFl1Sh']   = np.zeros(5)
        fst_vt['ElastoDynBlade']['BldFl2Sh']   = np.zeros(5)
        fst_vt['ElastoDynBlade']['BldEdgSh']   = np.zeros(5)
        for i in range(5):
            fst_vt['ElastoDynBlade']['BldFl1Sh'][i] = inputs['blade:flap_mode_shapes'][0,i] / sum(inputs['blade:flap_mode_shapes'][0,:])
            fst_vt['ElastoDynBlade']['BldFl2Sh'][i] = inputs['blade:flap_mode_shapes'][1,i] / sum(inputs['blade:flap_mode_shapes'][1,:])
            fst_vt['ElastoDynBlade']['BldEdgSh'][i] = inputs['blade:edge_mode_shapes'][0,i] / sum(inputs['blade:edge_mode_shapes'][0,:])

        # if flap/other mode shapes are all 0s, then the DOF should be disabled because they were not calculated
        if all(np.isnan(fst_vt['ElastoDynBlade']['BldFl1Sh'])):
            fst_vt['ElastoDyn']['FlapDOF1'] = False
            fst_vt['ElastoDynBlade']['BldFl1Sh'] = [1,0,0,0,0]
            print("WEIS Warning: Setting FlapDOF1 to False in ElastoDyn because mode shapes were not calculated")

        if all(np.isnan(fst_vt['ElastoDynBlade']['BldFl2Sh'])):
            fst_vt['ElastoDyn']['FlapDOF2'] = False
            fst_vt['ElastoDynBlade']['BldFl2Sh'] = [1,0,0,0,0]
            print("WEIS Warning: Setting FlapDOF2 to False in ElastoDyn because mode shapes were not calculated")

        if all(np.isnan(fst_vt['ElastoDynBlade']['BldEdgSh'])):
            fst_vt['ElastoDyn']['EdgeDOF'] = False
            fst_vt['ElastoDynBlade']['BldEdgSh'] = [1,0,0,0,0]
            print("WEIS Warning: Setting EdgeDOF1 to False in ElastoDyn because mode shapes were not calculated")
              
        # Update AeroDyn, Fst inputs for environment
        fst_vt['Fst']['AirDens']   = float(inputs['rho'])
        fst_vt['Fst']['SpdSound']  = float(inputs['speed_sound_air'])
        if modopt['flags']['marine_hydro']:
            fst_vt['Fst']['KinVisc']   = inputs['mu_water'][0] / inputs['rho_water'][0]
        else:
            fst_vt['Fst']['KinVisc']   = inputs['mu'][0] / inputs['rho'][0]

        # Use defaults in AeroDyn
        fst_vt['AeroDyn']['AirDens']  = 'default'
        fst_vt['AeroDyn']['KinVisc']  = 'default'
        fst_vt['AeroDyn']['SpdSound'] = 'default'
        fst_vt['AeroDyn']['Patm']     = 'default'
        fst_vt['AeroDyn']['Pvap']     = 'default'

        if modopt['flags']['marine_hydro']:
            fst_vt['AeroDyn']['Buoyancy'] = True
            # fst_vt['AeroDyn']['CavitCheck'] = True   # Disabling since WISDEM doesn't keep track of Cpmin yet

        # Update AeroDyn Blade Input File
        r = (inputs['r']-inputs['Rhub'])
        r[0]  = 0.
        r[-1] = inputs['Rtip']-inputs['Rhub']
        fst_vt['AeroDynBlade']['NumBlNds'] = self.n_span
        fst_vt['AeroDynBlade']['BlSpn']    = r
        BlCrvAC, BlSwpAC = self.get_ac_axis(inputs)
        fst_vt['AeroDynBlade']['BlCrvAC']  = BlCrvAC
        fst_vt['AeroDynBlade']['BlSwpAC']  = BlSwpAC
        fst_vt['AeroDynBlade']['BlCrvAng'] = np.degrees(np.arcsin(np.gradient(BlCrvAC)/np.gradient(r)))
        fst_vt['AeroDynBlade']['BlTwist']  = inputs['theta']
        fst_vt['AeroDynBlade']['BlChord']  = inputs['chord']
        fst_vt['AeroDynBlade']['BlAFID']   = np.asarray(range(1,self.n_span+1))

        # TODO: Check these additional values required for MHK, setting them to zero for now
        fst_vt['AeroDynBlade']['BlCb'] = np.zeros(self.n_span)
        fst_vt['AeroDynBlade']['BlCenBn'] = np.zeros(self.n_span)
        fst_vt['AeroDynBlade']['BlCenBt'] = np.zeros(self.n_span)

        # Update AeroDyn Airfoile Input Files
        # airfoils = inputs['airfoils']
        fst_vt['AeroDyn']['NumAFfiles'] = self.n_span
        # fst_vt['AeroDyn']['af_data'] = [{}]*len(airfoils)
        fst_vt['AeroDyn']['af_data'] = []

        # Set the AD15 flag AFTabMod, deciding whether we use more Re per airfoil or user-defined tables (used for example in distributed aerodynamic control)
        if fst_vt['AeroDyn']['AFTabMod'] == 1:
            # If AFTabMod is the default coming form the schema, check the value from WISDEM, which might be set to 2 if more Re per airfoil are defined in the geometry yaml
            fst_vt['AeroDyn']['AFTabMod'] = modopt["WISDEM"]["RotorSE"]["AFTabMod"]
        if self.n_tab > 1 and fst_vt['AeroDyn']['AFTabMod'] == 1:
            fst_vt['AeroDyn']['AFTabMod'] = 3
        elif self.n_tab > 1 and fst_vt['AeroDyn']['AFTabMod'] == 2:
            raise Exception('OpenFAST does not support both multiple Re and multiple user defined tabs. Please remove DAC devices or Re polars')

        for i in range(self.n_span): # No of blade radial stations

            fst_vt['AeroDyn']['af_data'].append([])

            if fst_vt['AeroDyn']['AFTabMod'] == 1:
                loop_index = 1
            elif fst_vt['AeroDyn']['AFTabMod'] == 2:
                loop_index = self.n_Re
            else:
                loop_index = self.n_tab

            for j in range(loop_index): # Number of tabs or Re
                if fst_vt['AeroDyn']['AFTabMod'] == 1:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,0], inputs['airfoils_cd'][i,:,0,0], inputs['airfoils_cm'][i,:,0,0])
                elif fst_vt['AeroDyn']['AFTabMod'] == 2:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,j,0], inputs['airfoils_cd'][i,:,j,0], inputs['airfoils_cm'][i,:,j,0])
                else:
                    unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][i,:,0,j], inputs['airfoils_cd'][i,:,0,j], inputs['airfoils_cm'][i,:,0,j])

                fst_vt['AeroDyn']['af_data'][i].append({})


                fst_vt['AeroDyn']['af_data'][i][j]['InterpOrd'] = "DEFAULT"
                fst_vt['AeroDyn']['af_data'][i][j]['NonDimArea']= 1
                if modopt['General']['openfast_configuration']['generate_af_coords']:
                    fst_vt['AeroDyn']['af_data'][i][j]['NumCoords'] = '@"AF{:02d}_Coords.txt"'.format(i)
                else:
                    fst_vt['AeroDyn']['af_data'][i][j]['NumCoords'] = '0'

                fst_vt['AeroDyn']['af_data'][i][j]['NumTabs']   = loop_index
                if fst_vt['AeroDyn']['AFTabMod'] == 3:
                    fst_vt['AeroDyn']['af_data'][i][j]['UserPropProp'] = inputs['airfoils_UserProp'][i,0,j]  # unsteady['UserProp'] # added to unsteady function for variable flap controls at airfoils
                    fst_vt['AeroDyn']['af_data'][i][j]['Re']   = inputs['airfoils_Re'][0] # If AFTabMod==3 the Re is neglected, but it still must be the same across tables
                else:
                    fst_vt['AeroDyn']['af_data'][i][j]['Re']   = inputs['airfoils_Re'][j]
                    fst_vt['AeroDyn']['af_data'][i][j]['UserProp'] = 0.
                fst_vt['AeroDyn']['af_data'][i][j]['InclUAdata']= "True"
                fst_vt['AeroDyn']['af_data'][i][j]['alpha0']    = unsteady['alpha0']
                fst_vt['AeroDyn']['af_data'][i][j]['alpha1']    = max(unsteady['alpha0'], unsteady['alpha1'])
                fst_vt['AeroDyn']['af_data'][i][j]['alpha2']    = min(unsteady['alpha0'], unsteady['alpha2'])
                fst_vt['AeroDyn']['af_data'][i][j]['eta_e']     = unsteady['eta_e']
                fst_vt['AeroDyn']['af_data'][i][j]['C_nalpha']  = unsteady['C_nalpha']
                fst_vt['AeroDyn']['af_data'][i][j]['T_f0']      = unsteady['T_f0']
                fst_vt['AeroDyn']['af_data'][i][j]['T_V0']      = unsteady['T_V0']
                fst_vt['AeroDyn']['af_data'][i][j]['T_p']       = unsteady['T_p']
                fst_vt['AeroDyn']['af_data'][i][j]['T_VL']      = unsteady['T_VL']
                fst_vt['AeroDyn']['af_data'][i][j]['b1']        = unsteady['b1']
                fst_vt['AeroDyn']['af_data'][i][j]['b2']        = unsteady['b2']
                fst_vt['AeroDyn']['af_data'][i][j]['b5']        = unsteady['b5']
                fst_vt['AeroDyn']['af_data'][i][j]['A1']        = unsteady['A1']
                fst_vt['AeroDyn']['af_data'][i][j]['A2']        = unsteady['A2']
                fst_vt['AeroDyn']['af_data'][i][j]['A5']        = unsteady['A5']
                fst_vt['AeroDyn']['af_data'][i][j]['S1']        = unsteady['S1']
                fst_vt['AeroDyn']['af_data'][i][j]['S2']        = unsteady['S2']
                fst_vt['AeroDyn']['af_data'][i][j]['S3']        = unsteady['S3']
                fst_vt['AeroDyn']['af_data'][i][j]['S4']        = unsteady['S4']
                fst_vt['AeroDyn']['af_data'][i][j]['Cn1']       = unsteady['Cn1']
                fst_vt['AeroDyn']['af_data'][i][j]['Cn2']       = unsteady['Cn2']
                fst_vt['AeroDyn']['af_data'][i][j]['St_sh']     = unsteady['St_sh']
                fst_vt['AeroDyn']['af_data'][i][j]['Cd0']       = unsteady['Cd0']
                fst_vt['AeroDyn']['af_data'][i][j]['Cm0']       = unsteady['Cm0']
                fst_vt['AeroDyn']['af_data'][i][j]['k0']        = unsteady['k0']
                fst_vt['AeroDyn']['af_data'][i][j]['k1']        = unsteady['k1']
                fst_vt['AeroDyn']['af_data'][i][j]['k2']        = unsteady['k2']
                fst_vt['AeroDyn']['af_data'][i][j]['k3']        = unsteady['k3']
                fst_vt['AeroDyn']['af_data'][i][j]['k1_hat']    = unsteady['k1_hat']
                fst_vt['AeroDyn']['af_data'][i][j]['x_cp_bar']  = unsteady['x_cp_bar']
                fst_vt['AeroDyn']['af_data'][i][j]['UACutout']  = unsteady['UACutout']
                fst_vt['AeroDyn']['af_data'][i][j]['filtCutOff']= unsteady['filtCutOff']
                fst_vt['AeroDyn']['af_data'][i][j]['NumAlf']    = len(unsteady['Alpha'])
                fst_vt['AeroDyn']['af_data'][i][j]['Alpha']     = np.array(unsteady['Alpha'])
                fst_vt['AeroDyn']['af_data'][i][j]['Cl']        = np.array(unsteady['Cl'])
                fst_vt['AeroDyn']['af_data'][i][j]['Cd']        = np.array(unsteady['Cd'])
                fst_vt['AeroDyn']['af_data'][i][j]['Cm']        = np.array(unsteady['Cm'])
                fst_vt['AeroDyn']['af_data'][i][j]['Cpmin']     = np.zeros_like(unsteady['Cm'])

        fst_vt['AeroDyn']['af_coord'] = []
        fst_vt['AeroDyn']['rthick']   = np.zeros(self.n_span)
        fst_vt['AeroDyn']['ac']   = np.zeros(self.n_span)
        for i in range(self.n_span):
            fst_vt['AeroDyn']['af_coord'].append({})
            fst_vt['AeroDyn']['af_coord'][i]['x']  = inputs['coord_xy_interp'][i,:,0]
            fst_vt['AeroDyn']['af_coord'][i]['y']  = inputs['coord_xy_interp'][i,:,1]
            fst_vt['AeroDyn']['rthick'][i]         = inputs['rthick'][i]
            fst_vt['AeroDyn']['ac'][i]             = inputs['ac'][i]

        # # AeroDyn blade spanwise output positions
        r_out_target  = [0.1, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        r = r/r[-1]
        idx_out       = [np.argmin(abs(r-ri)) for ri in r_out_target]
        self.R_out_AD = [fst_vt['AeroDynBlade']['BlSpn'][i] for i in idx_out]
        if len(self.R_out_AD) != len(np.unique(self.R_out_AD)):
            raise Exception('ERROR: the spanwise resolution is too coarse and does not support 9 channels along blade span. Please increase it in the modeling_options.yaml.')
        fst_vt['AeroDyn']['BlOutNd']  = [str(idx+1) for idx in idx_out]
        fst_vt['AeroDyn']['NBlOuts']  = len(idx_out)

        # ElastoDyn blade spanwise output positions
        nBldNodes     = fst_vt['ElastoDyn']['BldNodes']
        bld_fract     = np.arange(1./nBldNodes/2., 1, 1./nBldNodes)
        idx_out       = [np.argmin(abs(bld_fract-ri)) for ri in r_out_target]
        r_nodes       = bld_fract*(fst_vt['ElastoDyn']['TipRad']-fst_vt['ElastoDyn']['HubRad']) + fst_vt['ElastoDyn']['HubRad']
        self.R_out_ED_bl = np.hstack((fst_vt['ElastoDyn']['HubRad'], [r_nodes[i] for i in idx_out]))
        if len(self.R_out_ED_bl) != len(np.unique(self.R_out_ED_bl)):
            raise Exception('ERROR: the spanwise resolution is too coarse and does not support 9 channels along blade span. Please increase it in the modeling_options.yaml.')
        fst_vt['ElastoDyn']['BldGagNd'] = [idx+1 for idx in idx_out]
        fst_vt['ElastoDyn']['NBlGages'] = len(idx_out)

        # ElastoDyn tower output positions along height
        fst_vt['ElastoDyn']['NTwGages'] = 9
        nTwrNodes = fst_vt['ElastoDyn']['TwrNodes']
        twr_fract = np.arange(1./nTwrNodes/2., 1, 1./nTwrNodes)
        idx_out = [np.argmin(abs(twr_fract-ri)) for ri in r_out_target]
        fst_vt['ElastoDyn']['TwrGagNd'] = [idx+1 for idx in idx_out]
        fst_vt['AeroDyn']['NTwOuts'] = 0
        fst_vt['AeroDyn']['TwOutNd'] = ['0']
        self.Z_out_ED_twr = np.hstack((0., [twr_fract[i] for i in idx_out], 1.))
        if len(np.unique(self.Z_out_ED_twr)) < len(self.Z_out_ED_twr):
            raise Exception('The minimum number of tower nodes for WEIS to compute forces along the tower height is 11.')

        # SubDyn inputs- monopile and floating
        if modopt['OpenFAST']['simulation']['CompSub']:
            if modopt['flags']['monopile']:
                mono_d = inputs['monopile_outer_diameter']
                mono_t = inputs['monopile_wall_thickness']
                mono_elev = inputs['monopile_z']
                n_joints = len(mono_d[1:]) # Omit submerged pile
                n_members = n_joints - 1
                itrans = n_joints - 1
                fst_vt['SubDyn']['JointXss'] = np.zeros( n_joints )
                fst_vt['SubDyn']['JointYss'] = np.zeros( n_joints )
                fst_vt['SubDyn']['JointZss'] = mono_elev[1:]
                fst_vt['SubDyn']['NReact'] = 1
                fst_vt['SubDyn']['RJointID'] = [1]
                fst_vt['SubDyn']['RctTDXss'] = fst_vt['SubDyn']['RctTDYss'] = fst_vt['SubDyn']['RctTDZss'] = [1]
                fst_vt['SubDyn']['RctRDXss'] = fst_vt['SubDyn']['RctRDYss'] = fst_vt['SubDyn']['RctRDZss'] = [1]
                fst_vt['SubDyn']['NInterf'] = 1
                fst_vt['SubDyn']['IJointID'] = [n_joints]
                fst_vt['SubDyn']['MJointID1'] = np.arange( n_members, dtype=np.int_ ) + 1
                fst_vt['SubDyn']['MJointID2'] = np.arange( n_members, dtype=np.int_ ) + 2
                fst_vt['SubDyn']['YoungE1'] = inputs['monopile_E'][1:]
                fst_vt['SubDyn']['ShearG1'] = inputs['monopile_G'][1:]
                fst_vt['SubDyn']['MatDens1'] = inputs['monopile_rho'][1:]
                fst_vt['SubDyn']['XsecD'] = util.nodal2sectional(mono_d[1:])[0] # Don't need deriv
                fst_vt['SubDyn']['XsecT'] = mono_t[1:]
                
                # Find the members where the 9 channels of SubDyn should be placed
                grid_joints_monopile = (fst_vt['SubDyn']['JointZss'] - fst_vt['SubDyn']['JointZss'][0]) / (fst_vt['SubDyn']['JointZss'][-1] - fst_vt['SubDyn']['JointZss'][0])
                n_channels = 9
                grid_target = np.linspace(0., 0.999999999, n_channels)
                idx_out = [np.where(grid_i >= grid_joints_monopile)[0][-1] for grid_i in grid_target]
                idx_out = np.unique(idx_out)
                fst_vt['SubDyn']['NMOutputs'] = len(idx_out)
                fst_vt['SubDyn']['MemberID_out'] = [idx+1 for idx in idx_out]
                fst_vt['SubDyn']['NOutCnt'] = np.ones_like(fst_vt['SubDyn']['MemberID_out'])
                fst_vt['SubDyn']['NodeCnt'] = np.ones_like(fst_vt['SubDyn']['MemberID_out'])
                fst_vt['SubDyn']['NodeCnt'][-1] = 2
                self.Z_out_SD_mpl = [grid_joints_monopile[i] for i in idx_out]

                # Add SubDyn output channels for monopile
                for i in range(fst_vt['SubDyn']['NMOutputs']):
                    for j in fst_vt['SubDyn']['NodeCnt'][i]:
                        fst_vt['outlist']['SubDyn'][f'M{i+1}N{j}FKxe'] = True
                        fst_vt['outlist']['SubDyn'][f'M{i+1}N{j}FKye'] = True
                        fst_vt['outlist']['SubDyn'][f'M{i+1}N{j}FKze'] = True
                        fst_vt['outlist']['SubDyn'][f'M{i+1}N{j}MKxe'] = True
                        fst_vt['outlist']['SubDyn'][f'M{i+1}N{j}MKye'] = True
                        fst_vt['outlist']['SubDyn'][f'M{i+1}N{j}MKze'] = True

            elif modopt['flags']['floating']:
                joints_xyz = inputs["platform_nodes"]
                n_joints = np.where(joints_xyz[:, 0] == NULL)[0][0]
                joints_xyz = joints_xyz[:n_joints, :]
                itrans = util.closest_node(joints_xyz, inputs["transition_node"])

                N1 = np.int_(inputs["platform_elem_n1"])
                n_members = np.where(N1 == NULL)[0][0]
                N1 = N1[:n_members]
                N2 = np.int_(inputs["platform_elem_n2"][:n_members])

                fst_vt['SubDyn']['JointXss'] = joints_xyz[:,0]
                fst_vt['SubDyn']['JointYss'] = joints_xyz[:,1]
                fst_vt['SubDyn']['JointZss'] = joints_xyz[:,2]
                fst_vt['SubDyn']['NReact'] = 0
                fst_vt['SubDyn']['RJointID'] = []
                fst_vt['SubDyn']['RctTDXss'] = fst_vt['SubDyn']['RctTDYss'] = fst_vt['SubDyn']['RctTDZss'] = []
                fst_vt['SubDyn']['RctRDXss'] = fst_vt['SubDyn']['RctRDYss'] = fst_vt['SubDyn']['RctRDZss'] = []
                if modopt['floating']['transition_joint'] is None:
                    fst_vt['SubDyn']['NInterf'] = 0
                    fst_vt['SubDyn']['IJointID'] = []
                else:
                    fst_vt['SubDyn']['NInterf'] = 1
                    fst_vt['SubDyn']['IJointID'] = [itrans+1]
                fst_vt['SubDyn']['MJointID1'] = N1+1
                fst_vt['SubDyn']['MJointID2'] = N2+1

                fst_vt['SubDyn']['YoungE1'] = inputs["platform_elem_E"][:n_members]
                fst_vt['SubDyn']['ShearG1'] = inputs["platform_elem_G"][:n_members]
                fst_vt['SubDyn']['MatDens1'] = inputs["platform_elem_rho"][:n_members]
                fst_vt['SubDyn']['XsecD'] = inputs["platform_elem_D"][:n_members]
                fst_vt['SubDyn']['XsecT'] = inputs["platform_elem_t"][:n_members]

            # SubDyn inputs- offshore generic
            if modopt['flags']['offshore']:
                mgrav = 0.0 if not modopt['flags']['monopile'] else float(inputs['gravity_foundation_mass'])
                if fst_vt['SubDyn']['SDdeltaT']<=-999.0: fst_vt['SubDyn']['SDdeltaT'] = "DEFAULT"
                fst_vt['SubDyn']['GuyanDamp'] = np.vstack( tuple([fst_vt['SubDyn']['GuyanDamp'+str(m+1)] for m in range(6)]) )
                fst_vt['SubDyn']['Rct_SoilFile'] = [""]*fst_vt['SubDyn']['NReact']
                fst_vt['SubDyn']['NJoints'] = n_joints
                fst_vt['SubDyn']['JointID'] = np.arange( n_joints, dtype=np.int_) + 1
                fst_vt['SubDyn']['JointType'] = np.ones( n_joints, dtype=np.int_)
                fst_vt['SubDyn']['JointDirX'] = fst_vt['SubDyn']['JointDirY'] = fst_vt['SubDyn']['JointDirZ'] = np.zeros( n_joints )
                fst_vt['SubDyn']['JointStiff'] = np.zeros( n_joints )
                fst_vt['SubDyn']['ItfTDXss'] = fst_vt['SubDyn']['ItfTDYss'] = fst_vt['SubDyn']['ItfTDZss'] = [1]
                fst_vt['SubDyn']['ItfRDXss'] = fst_vt['SubDyn']['ItfRDYss'] = fst_vt['SubDyn']['ItfRDZss'] = [1]
                fst_vt['SubDyn']['NMembers'] = n_members
                fst_vt['SubDyn']['MemberID'] = np.arange( n_members, dtype=np.int_ ) + 1
                fst_vt['SubDyn']['MPropSetID1'] = fst_vt['SubDyn']['MPropSetID2'] = np.arange( n_members, dtype=np.int_ ) + 1
                fst_vt['SubDyn']['MType'] = np.ones( n_members, dtype=np.int_ )
                fst_vt['SubDyn']['M_COSMID'] = np.ones( n_members, dtype=np.int_ ) * -1 #  TODO: verify based on https://openfast.readthedocs.io/en/dev/source/user/subdyn/input_files.html#members
                fst_vt['SubDyn']['NPropSets'] = n_members
                fst_vt['SubDyn']['PropSetID1'] = np.arange( n_members, dtype=np.int_ ) + 1
                fst_vt['SubDyn']['NCablePropSets'] = 0
                fst_vt['SubDyn']['NRigidPropSets'] = 0
                fst_vt['SubDyn']['NSpringPropSets'] = 0
                fst_vt['SubDyn']['NCOSMs'] = 0
                fst_vt['SubDyn']['NXPropSets'] = 0
                fst_vt['SubDyn']['NCmass'] = 2 if mgrav > 0.0 else 1
                fst_vt['SubDyn']['CMJointID'] = [itrans+1]
                fst_vt['SubDyn']['JMass'] = [float(inputs['transition_piece_mass'])]
                fst_vt['SubDyn']['JMXX'] = [inputs['transition_piece_I'][0]]
                fst_vt['SubDyn']['JMYY'] = [inputs['transition_piece_I'][1]]
                fst_vt['SubDyn']['JMZZ'] = [inputs['transition_piece_I'][2]]
                fst_vt['SubDyn']['JMXY'] = fst_vt['SubDyn']['JMXZ'] = fst_vt['SubDyn']['JMYZ'] = [0.0]
                fst_vt['SubDyn']['MCGX'] = fst_vt['SubDyn']['MCGY'] = fst_vt['SubDyn']['MCGZ'] = [0.0]
                if mgrav > 0.0:
                    fst_vt['SubDyn']['CMJointID'] += [1]
                    fst_vt['SubDyn']['JMass'] += [mgrav]
                    fst_vt['SubDyn']['JMXX'] += [inputs['gravity_foundation_I'][0]]
                    fst_vt['SubDyn']['JMYY'] += [inputs['gravity_foundation_I'][1]]
                    fst_vt['SubDyn']['JMZZ'] += [inputs['gravity_foundation_I'][2]]
                    fst_vt['SubDyn']['JMXY'] += [0.0]
                    fst_vt['SubDyn']['JMXZ'] += [0.0]
                    fst_vt['SubDyn']['JMYZ'] += [0.0]
                    fst_vt['SubDyn']['MCGX'] += [0.0]
                    fst_vt['SubDyn']['MCGY'] += [0.0]
                    fst_vt['SubDyn']['MCGZ'] += [0.0]


        # HydroDyn inputs
        if fst_vt['Fst']['CompHydro']:
            if modopt['flags']['monopile']:
                z_coarse = make_coarse_grid(mono_elev[1:], mono_d[1:])
                # Don't want any nodes near zero for annoying hydrodyn errors
                idx0 = np.intersect1d(np.where(z_coarse>-0.5), np.where(z_coarse<0.5))
                z_coarse = np.delete(z_coarse, idx0) 
                n_joints = len(z_coarse)
                n_members = n_joints - 1
                joints_xyz = np.c_[np.zeros((n_joints,2)), z_coarse]
                d_coarse = np.interp(z_coarse, mono_elev[1:], mono_d[1:])
                t_coarse = util.sectional_interp(z_coarse, mono_elev[1:], mono_t[1:])
                N1 = np.arange( n_members, dtype=np.int_ ) + 1
                N2 = np.arange( n_members, dtype=np.int_ ) + 2
                
            elif modopt['flags']['floating']:
                joints_xyz = np.empty((0, 3))
                axial_coeffs = np.empty((0, 3))
                N1 = np.array([], dtype=np.int_)
                N2 = np.array([], dtype=np.int_)
                d_coarse = np.array([])
                t_coarse = np.array([])
                
                # Look over members and grab all nodes and internal connections
                n_member = modopt["floating"]["members"]["n_members"]
                for k in range(n_member):
                    s_grid = inputs[f"member{k}:s"]
                    idiam = inputs[f"member{k}:outer_diameter"]
                    s_coarse = make_coarse_grid(s_grid, idiam)
                    s_coarse = np.unique( np.minimum( np.maximum(s_coarse, inputs[f"member{k}:s_ghost1"]), inputs[f"member{k}:s_ghost2"]) )
                    id_coarse = np.interp(s_coarse, s_grid, idiam)
                    it_coarse = util.sectional_interp(s_coarse, s_grid, inputs[f"member{k}:wall_thickness"])
                    xyz0 = inputs[f"member{k}:joint1"]
                    xyz1 = inputs[f"member{k}:joint2"]
                    dxyz = xyz1 - xyz0
                    inode_xyz = np.outer(s_coarse, dxyz) + xyz0[np.newaxis, :]
                    inode_range = np.arange(inode_xyz.shape[0] - 1)

                    nk = joints_xyz.shape[0]
                    N1 = np.append(N1, nk + inode_range + 1)
                    N2 = np.append(N2, nk + inode_range + 2)
                    d_coarse = np.append(d_coarse, id_coarse)  
                    t_coarse = np.append(t_coarse, it_coarse)  
                    joints_xyz = np.append(joints_xyz, inode_xyz, axis=0)

                    # Axial coefficients
                    joint_1_orig_index = modopt['floating']['joints']['name2idx'][modopt['floating']['members']['joint1'][k]]
                    joint_2_orig_index = modopt['floating']['joints']['name2idx'][modopt['floating']['members']['joint2'][k]]
                    
                    # may need to check if joint is in original list, axial joints will not be
                    if modopt['floating']['members']['joint1'][k] in modopt['floating']['joints']['name'] and \
                        modopt['floating']['members']['joint2'][k] in modopt['floating']['joints']['name'] :

                        i_axial_coeff_1 = [
                            modopt['floating']['joints']['axial_coeffs'][joint_1_orig_index]['Cd'],
                            modopt['floating']['joints']['axial_coeffs'][joint_1_orig_index]['Ca'],
                            modopt['floating']['joints']['axial_coeffs'][joint_1_orig_index]['Cp']
                        ]

                        i_axial_coeff_2 = [
                            modopt['floating']['joints']['axial_coeffs'][joint_2_orig_index]['Cd'],
                            modopt['floating']['joints']['axial_coeffs'][joint_2_orig_index]['Ca'],
                            modopt['floating']['joints']['axial_coeffs'][joint_2_orig_index]['Cp']
                        ]
                    else:
                        # not originally defined
                        i_axial_coeff_1 = np.zeros(3)
                        i_axial_coeff_2 = np.zeros(3)


                    i_axial_coeffs = np.r_[[i_axial_coeff_1],[i_axial_coeff_2]]
                    axial_coeffs = np.append(axial_coeffs,i_axial_coeffs, axis = 0)
                

                    
            if modopt['flags']['offshore']:
                fst_vt['SeaState']['WtrDens'] = float(inputs['rho_water'])
                fst_vt['SeaState']['WtrDpth'] = float(inputs['water_depth'])
                fst_vt['SeaState']['MSL2SWL'] = 0.0
                fst_vt['SeaState']['WaveHs'] = float(inputs['Hsig_wave'])
                fst_vt['SeaState']['WaveTp'] = float(inputs['Tsig_wave'])
                fst_vt['SeaState']['WaveDir'] = float(inputs['beta_wave'])
                fst_vt['SeaState']['WaveDirRange'] = fst_vt['SeaState']['WaveDirRange'] / np.rad2deg(1)
                fst_vt['SeaState']['WaveElevxi'] = [str(m) for m in fst_vt['SeaState']['WaveElevxi']]
                fst_vt['SeaState']['WaveElevyi'] = [str(m) for m in fst_vt['SeaState']['WaveElevyi']]
                fst_vt['SeaState']['CurrSSDir'] = "DEFAULT" if fst_vt['SeaState']['CurrSSDir']<=-999.0 else np.rad2deg(fst_vt['SeaState']['CurrSSDir'])
                fst_vt['HydroDyn']['AddF0'] = np.array( fst_vt['HydroDyn']['AddF0'] ).reshape(-1,1)
                fst_vt['HydroDyn']['AddCLin'] = np.vstack( tuple([fst_vt['HydroDyn']['AddCLin'+str(m+1)] for m in range(6)]) )
                fst_vt['HydroDyn']['AddBLin'] = np.vstack( tuple([fst_vt['HydroDyn']['AddBLin'+str(m+1)] for m in range(6)]) )
                BQuad = np.vstack( tuple([fst_vt['HydroDyn']['AddBQuad'+str(m+1)] for m in range(6)]) )
                if np.any(BQuad):
                    logger.warning('WARNING: You are adding in additional drag terms that may double count strip theory estimated viscous drag terms.  Please zero out the BQuad entries or use modeling options SimplCd/a/p and/or potential_model_override and/or potential_bem_members to suppress strip theory for the members')
                fst_vt['HydroDyn']['AddBQuad'] = BQuad
                if modopt['flags']['floating']:
                    fst_vt['HydroDyn']['NAxCoef'] = axial_coeffs.shape[0]
                    fst_vt['HydroDyn']['AxCoefID'] = 1 + np.arange( fst_vt['HydroDyn']['NAxCoef'], dtype=np.int_)
                    fst_vt['HydroDyn']['AxCd'] = axial_coeffs[:,0]
                    fst_vt['HydroDyn']['AxCa'] = axial_coeffs[:,1]
                    fst_vt['HydroDyn']['AxCp'] = axial_coeffs[:,2]
                else:
                    fst_vt['HydroDyn']['NAxCoef'] = 1
                    fst_vt['HydroDyn']['AxCoefID'] = 1 + np.arange( fst_vt['HydroDyn']['NAxCoef'], dtype=np.int_)
                    fst_vt['HydroDyn']['AxCd'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
                    fst_vt['HydroDyn']['AxCa'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
                    fst_vt['HydroDyn']['AxCp'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
                
                # Other axial coefficients (zeros for now)
                fst_vt['HydroDyn']['AxFDMod'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
                fst_vt['HydroDyn']['AxVnCOff'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )
                fst_vt['HydroDyn']['AxFDLoFSc'] = np.zeros( fst_vt['HydroDyn']['NAxCoef'] )


            # Simplify members if using potential model only
            if modopt["RAFT"]["potential_model_override"] == 2:
                joints_xyz = np.array([[0,0,0],[0,0,-1]])
                N1 = np.array([N1[0]])
                N2 = np.array([N2[0]])
                
            # Tweak z-position
            idx = np.where(joints_xyz[:,2]==-fst_vt['SeaState']['WtrDpth'])[0]
            if len(idx) > 0:
                joints_xyz[idx,2] += 1e-2
            # Store data
            n_joints = joints_xyz.shape[0]
            n_members = N1.shape[0]
            ijoints = np.arange( n_joints, dtype=np.int_ ) + 1
            imembers = np.arange( n_members, dtype=np.int_ ) + 1
            fst_vt['HydroDyn']['NJoints'] = n_joints
            fst_vt['HydroDyn']['JointID'] = ijoints
            fst_vt['HydroDyn']['Jointxi'] = joints_xyz[:,0]
            fst_vt['HydroDyn']['Jointyi'] = joints_xyz[:,1]
            fst_vt['HydroDyn']['Jointzi'] = joints_xyz[:,2]
            fst_vt['HydroDyn']['NPropSets'] = n_joints      # each joint has a cross section
            fst_vt['HydroDyn']['PropSetID'] = ijoints
            fst_vt['HydroDyn']['PropD'] = d_coarse
            fst_vt['HydroDyn']['PropThck'] = t_coarse
            fst_vt['HydroDyn']['NMembers'] = n_members
            fst_vt['HydroDyn']['MemberID'] = imembers
            fst_vt['HydroDyn']['MJointID1'] = fst_vt['HydroDyn']['MPropSetID1'] = N1
            fst_vt['HydroDyn']['MJointID2'] = fst_vt['HydroDyn']['MPropSetID2'] = N2
            fst_vt['HydroDyn']['MDivSize'] = 0.5*np.ones( fst_vt['HydroDyn']['NMembers'] )
            fst_vt['HydroDyn']['MCoefMod'] = np.ones( fst_vt['HydroDyn']['NMembers'], dtype=np.int_)
            fst_vt['HydroDyn']['MHstLMod'] = np.ones( fst_vt['HydroDyn']['NMembers'], dtype=np.int_)
            fst_vt['HydroDyn']['JointAxID'] = np.ones( fst_vt['HydroDyn']['NJoints'], dtype=np.int_)
            fst_vt['HydroDyn']['JointOvrlp'] = np.zeros( fst_vt['HydroDyn']['NJoints'], dtype=np.int_)
            fst_vt['HydroDyn']['NCoefDpth'] = 0
            fst_vt['HydroDyn']['NCoefMembers'] = 0
            fst_vt['HydroDyn']['NFillGroups'] = 0
            fst_vt['HydroDyn']['NMGDepths'] = 0

            if modopt["RAFT"]["potential_model_override"] == 1:
                # Strip theory only, no BEM
                fst_vt['HydroDyn']['PropPot'] = [False] * fst_vt['HydroDyn']['NMembers']
            elif modopt["RAFT"]["potential_model_override"] == 2:
                # BEM only, no strip theory
                fst_vt['HydroDyn']['SimplCd'] = fst_vt['HydroDyn']['SimplCdMG'] = 0.0
                fst_vt['HydroDyn']['SimplCa'] = fst_vt['HydroDyn']['SimplCaMG'] = 0.0
                fst_vt['HydroDyn']['SimplCp'] = fst_vt['HydroDyn']['SimplCpMG'] = 0.0
                fst_vt['HydroDyn']['SimplAxCd'] = fst_vt['HydroDyn']['SimplAxCdMG'] = 0.0
                fst_vt['HydroDyn']['SimplAxCa'] = fst_vt['HydroDyn']['SimplAxCaMG'] = 0.0
                fst_vt['HydroDyn']['SimplAxCp'] = fst_vt['HydroDyn']['SimplAxCpMG'] = 0.0
                fst_vt['HydroDyn']['SimplCb'] = fst_vt['HydroDyn']['SimplCbMG'] = 0.0
                fst_vt['HydroDyn']['PropPot'] = [True] * fst_vt['HydroDyn']['NMembers']
            else:
                PropPotBool = [False] * fst_vt['HydroDyn']['NMembers']
                for k in range(fst_vt['HydroDyn']['NMembers']):
                    # Potential modeling of fixed substructres not supported
                    if modopt['flags']['floating']:
                        idx = modopt['floating']['members']['platform_elem_memid'][k]
                        PropPotBool[k] = modopt["RAFT"]["model_potential"][idx]    
                fst_vt['HydroDyn']['PropPot'] = PropPotBool

                if fst_vt['HydroDyn']['NBody'] > 1:
                    raise Exception('Multiple HydroDyn bodies (NBody > 1) is currently not supported in WEIS')

            # Offset of body reference point
            fst_vt['HydroDyn']['PtfmRefxt']     = [0]
            fst_vt['HydroDyn']['PtfmRefyt']     = [0]
            fst_vt['HydroDyn']['PtfmRefzt']     = [0]
            fst_vt['HydroDyn']['PtfmRefztRot']  = [0]

            # If we're using the potential model, need these settings that aren't default
            if fst_vt['HydroDyn']['PotMod'] == 1:
                fst_vt['HydroDyn']['ExctnMod'] = 1
                fst_vt['HydroDyn']['RdtnMod'] = 1
                fst_vt['HydroDyn']['RdtnDT'] = "DEFAULT"

            # Displaced volume for WAMIT models
            fst_vt['HydroDyn']['PtfmVol0'] = [float(inputs['platform_displacement'])] 

            if fst_vt['HydroDyn']['PotMod'] == 1 and modopt['OpenFAST_Linear']['flag'] and modopt['RAFT']['runPyHAMS']:
                fst_vt['HydroDyn']['ExctnMod'] = 1
                fst_vt['HydroDyn']['RdtnMod'] = 1
                fst_vt['HydroDyn']['RdtnDT'] = "DEFAULT"

                from weis.ss_fitting.SS_FitTools import SSFit_Excitation, FDI_Fitting
                logger.warning('Writing .ss and .ssexctn models to: {}'.format(fst_vt['HydroDyn']['PotFile']))
                exctn_fit = SSFit_Excitation(HydroFile=fst_vt['HydroDyn']['PotFile'])
                rad_fit = FDI_Fitting(HydroFile=fst_vt['HydroDyn']['PotFile'])
                exctn_fit.writeMats()
                rad_fit.fit()
                rad_fit.outputMats()
                if True:
                    fig_list = rad_fit.visualizeFits()
                    
                    os.makedirs(os.path.join(os.path.dirname(fst_vt['HydroDyn']['PotFile']),'rad_fit'), exist_ok=True)

                for i_fig, fig in enumerate(fig_list):
                    fig.savefig(os.path.join(os.path.dirname(fst_vt['HydroDyn']['PotFile']),'rad_fit',f'rad_fit_{i_fig}.png'))
            


        # Moordyn inputs
        if modopt["flags"]["mooring"]:
            mooropt = modopt["mooring"]
            
            # Line types
            # Creating a line type for each line, regardless of whether it is unique or not
            n_lines = mooropt["n_lines"]
            line_names = ['line'+str(m) for m in range(n_lines)]
            fst_vt['MoorDyn']['NTypes'] = n_lines
            fst_vt['MoorDyn']['Name'] = fst_vt['MAP']['LineType'] = line_names
            fst_vt['MoorDyn']['Diam'] = fst_vt['MAP']['Diam'] = inputs["line_diameter"]
            fst_vt['MoorDyn']['MassDen'] = fst_vt['MAP']['MassDenInAir'] = inputs["line_mass_density"]
            fst_vt['MoorDyn']['EA'] = inputs["line_stiffness"]
            fst_vt['MoorDyn']['EI'] = np.zeros(n_lines)     # MoorPy does not have EI, yet
            fst_vt['MoorDyn']['BA_zeta'] = -1*np.ones(n_lines, dtype=np.int64)
            fst_vt['MoorDyn']['Ca'] = inputs["line_transverse_added_mass"]
            fst_vt['MoorDyn']['CaAx'] = inputs["line_tangential_added_mass"]
            fst_vt['MoorDyn']['Cd'] = inputs["line_transverse_drag"]
            fst_vt['MoorDyn']['CdAx'] = inputs["line_tangential_drag"]

            # Connection properties - Points
            n_nodes = mooropt["n_nodes"]
            fst_vt['MoorDyn']['NConnects'] = n_nodes
            fst_vt['MoorDyn']['Point_ID'] = np.arange(n_nodes)+1
            fst_vt['MoorDyn']['Attachment'] = mooropt["node_type"][:]
            fst_vt['MoorDyn']['X'] = inputs['nodes_location_full'][:,0]
            fst_vt['MoorDyn']['Y'] = inputs['nodes_location_full'][:,1]
            fst_vt['MoorDyn']['Z'] = inputs['nodes_location_full'][:,2]
            fst_vt['MoorDyn']['M'] = inputs['nodes_mass']
            fst_vt['MoorDyn']['V'] = inputs['nodes_volume']
            fst_vt['MoorDyn']['CdA'] = inputs['nodes_drag_area']
            fst_vt['MoorDyn']['CA'] = inputs['nodes_added_mass']

            # Lines
            fst_vt['MoorDyn']['NLines'] = n_lines
            fst_vt['MoorDyn']['Line_ID'] = np.arange(n_lines)+1
            fst_vt['MoorDyn']['LineType'] = line_names
            fst_vt['MoorDyn']['UnstrLen'] = inputs['unstretched_length']
            fst_vt['MoorDyn']['NumSegs'] = 20*np.ones(n_lines, dtype=np.int64)      # TODO: make this a modeling option, pull in EAB modeling branch
            fst_vt['MoorDyn']['AttachA'] = np.zeros(n_lines, dtype=np.int64)
            fst_vt['MoorDyn']['AttachB'] = np.zeros(n_lines, dtype=np.int64)
            fst_vt['MoorDyn']['Outputs'] = ['-'] * n_lines
            fst_vt['MoorDyn']['CtrlChan'] = np.zeros(n_lines, dtype=np.int64)

            for k in range(n_lines):
                id1 = discrete_inputs['node_names'].index( mooropt["node1"][k] )
                id2 = discrete_inputs['node_names'].index( mooropt["node2"][k] )
                if (fst_vt['MoorDyn']['Attachment'][id1].lower() == 'vessel' and
                    fst_vt['MoorDyn']['Attachment'][id2].lower().find('fix') >= 0):
                    fst_vt['MoorDyn']['AttachB'][k] = id1+1
                    fst_vt['MoorDyn']['AttachA'][k] = id2+1
                elif (fst_vt['MoorDyn']['Attachment'][id2].lower() == 'vessel' and
                    fst_vt['MoorDyn']['Attachment'][id1].lower().find('fix') >= 0):
                    fst_vt['MoorDyn']['AttachB'][k] = id2+1
                    fst_vt['MoorDyn']['AttachA'][k] = id1+1
                else:
                    logger.warning(discrete_inputs['node_names'])
                    logger.warning(mooropt["node1"][k], mooropt["node2"][k])
                    logger.warning(fst_vt['MoorDyn']['Attachment'][id1], fst_vt['MoorDyn']['Attachment'][id2])
                    raise ValueError('Mooring line seems to be between unknown endpoint types.')

            # MoorDyn Control - Optional
            fst_vt['MoorDyn']['ChannelID'] = []
            
            # MAP - linearization only
            for key in fst_vt['MoorDyn']:
                fst_vt['MAP'][key] = copy.copy(fst_vt['MoorDyn'][key])

            for idx, node_type in enumerate(fst_vt['MAP']['Attachment']):
                if node_type == 'fixed':
                    fst_vt['MAP']['Attachment'][idx] = 'fix'

            # TODO: FIXME: these values are hardcoded for the IEA15MW linearization studies
            fst_vt['MAP']['LineType'] = ['main', 'main', 'main']
            fst_vt['MAP']['CB'] = np.ones(n_lines)
            fst_vt['MAP']['CIntDamp'] = np.zeros(n_lines)
            fst_vt['MAP']['Ca'] = np.zeros(n_lines)
            fst_vt['MAP']['Cdn'] = np.zeros(n_lines)
            fst_vt['MAP']['Cdt'] = np.zeros(n_lines)
            fst_vt['MAP']['B'] = np.zeros( n_nodes )
            fst_vt['MAP']['Option'] = ["outer_tol 1e-5"]


        # Structural Control
        fst_vt['ServoDyn']['NumBStC']       = 0
        fst_vt['ServoDyn']['BStCfiles']     = ["unused"]
        fst_vt['ServoDyn']['NumNStC']       = 0
        fst_vt['ServoDyn']['NStCfiles']     = ["unused"]
        fst_vt['ServoDyn']['NumTStC']       = 0 
        fst_vt['ServoDyn']['TStCfiles']     = []
        fst_vt['ServoDyn']['NumSStC']       = 0
        fst_vt['ServoDyn']['SStCfiles']     = []
        
        if modopt['flags']['TMDs']:
            for i_TMD in range(modopt['TMDs']['n_TMDs']):

                StC_i = default_StC_vt()

                StC_i['StC_DOF_MODE']   = 1
                StC_i['StC_X_DOF']     = modopt['TMDs']['X_DOF'][i_TMD]
                StC_i['StC_Y_DOF']     = modopt['TMDs']['Y_DOF'][i_TMD]
                StC_i['StC_Z_DOF']     = modopt['TMDs']['Z_DOF'][i_TMD]

                if StC_i['StC_X_DOF'] and StC_i['StC_Y_DOF'] and not StC_i['StC_Z_DOF']:
                    StC_i['StC_DOF_MODE']   = 2
                    StC_i['StC_XY_M']       = inputs['TMD_mass'][i_TMD]

                # Compute spring offset for each direction, initializing
                g = modopt['OpenFAST']['simulation']['Gravity']
                spring_offset = np.zeros(3)
                
                # Set Mass, Stiffness, Damping only in DOFs enabled
                if StC_i['StC_X_DOF']:
                    StC_i['StC_X_M'] = inputs['TMD_mass'][i_TMD]
                    StC_i['StC_X_K'] = inputs['TMD_stiffness'][i_TMD]
                    StC_i['StC_X_C'] = inputs['TMD_damping'][i_TMD]
                
                if StC_i['StC_Y_DOF']:
                    StC_i['StC_Y_M'] = inputs['TMD_mass'][i_TMD]
                    StC_i['StC_Y_K'] = inputs['TMD_stiffness'][i_TMD]
                    StC_i['StC_Y_C'] = inputs['TMD_damping'][i_TMD]

                if StC_i['StC_Z_DOF']:
                    StC_i['StC_Z_M'] = inputs['TMD_mass'][i_TMD]
                    StC_i['StC_Z_K'] = inputs['TMD_stiffness'][i_TMD]
                    StC_i['StC_Z_C'] = inputs['TMD_damping'][i_TMD]
                    spring_offset[2] = StC_i['StC_Z_M'] * g / StC_i['StC_Z_K']

                # Set position
                StC_i['StC_P_X']  = modopt['TMDs']['location'][i_TMD][0]
                StC_i['StC_P_Y']  = modopt['TMDs']['location'][i_TMD][1]
                StC_i['StC_P_Z']  = modopt['TMDs']['location'][i_TMD][2]
                
                if modopt['TMDs']['preload_spring'][i_TMD]:
                    StC_i['StC_Z_PreLd']  = "gravity"
                    

                if modopt['TMDs']['component'][i_TMD] == 'tower':
                    fst_vt['ServoDyn']['NumTStC'] += 1
                    fst_vt['ServoDyn']['TStCfiles'].append(os.path.join(self.FAST_runDirectory,self.FAST_namingOut + f"_StC_Twr_{i_TMD}.dat"))
                    fst_vt['TStC'].append(StC_i)

                elif modopt['TMDs']['component'][i_TMD] in modopt['floating']['members']['name']:
                    fst_vt['ServoDyn']['NumSStC'] += 1
                    fst_vt['ServoDyn']['SStCfiles'].append(os.path.join(self.FAST_runDirectory,self.FAST_namingOut + f"_StC_Ptfm_{i_TMD}.dat"))
                    fst_vt['SStC'].append(StC_i)

            # If no StC file assigned, set to unused
            if not fst_vt['ServoDyn']['TStCfiles']:
                fst_vt['ServoDyn']['TStCfiles'] = ["unused"]
            if not fst_vt['ServoDyn']['SStCfiles']:
                fst_vt['ServoDyn']['SStCfiles'] = ["unused"]


        return fst_vt

    def output_channels(self,fst_vt):
        modopt = self.options['modeling_options']

        # Mandatory output channels to include
        # TODO: what else is needed here?
        channels_out  = ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxc2", "TipDyc2", "TipDzc2"]
        channels_out += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2"]
        channels_out += ["TipDxb1", "TipDyb1", "TipDzb1", "TipDxb2", "TipDyb2", "TipDzb2"]
        channels_out += ["RootMxb1", "RootMyb1", "RootMzb1", "RootMxb2", "RootMyb2", "RootMzb2"]
        channels_out += ["RootFxc1", "RootFyc1", "RootFzc1", "RootFxc2", "RootFyc2", "RootFzc2"]
        channels_out += ["RootFxb1", "RootFyb1", "RootFzb1", "RootFxb2", "RootFyb2", "RootFzb2"]
        channels_out += ["Spn1FLzb1", "Spn2FLzb1", "Spn3FLzb1", "Spn4FLzb1", "Spn5FLzb1", "Spn6FLzb1", "Spn7FLzb1", "Spn8FLzb1", "Spn9FLzb1"]
        channels_out += ["Spn1MLxb1", "Spn2MLxb1", "Spn3MLxb1", "Spn4MLxb1", "Spn5MLxb1", "Spn6MLxb1", "Spn7MLxb1", "Spn8MLxb1", "Spn9MLxb1"]
        channels_out += ["Spn1MLyb1", "Spn2MLyb1", "Spn3MLyb1", "Spn4MLyb1", "Spn5MLyb1", "Spn6MLyb1", "Spn7MLyb1", "Spn8MLyb1", "Spn9MLyb1"]
        channels_out += ["Spn1FLzb2", "Spn2FLzb2", "Spn3FLzb2", "Spn4FLzb2", "Spn5FLzb2", "Spn6FLzb2", "Spn7FLzb2", "Spn8FLzb2", "Spn9FLzb2"]
        channels_out += ["Spn1MLxb2", "Spn2MLxb2", "Spn3MLxb2", "Spn4MLxb2", "Spn5MLxb2", "Spn6MLxb2", "Spn7MLxb2", "Spn8MLxb2", "Spn9MLxb2"]
        channels_out += ["Spn1MLyb2", "Spn2MLyb2", "Spn3MLyb2", "Spn4MLyb2", "Spn5MLyb2", "Spn6MLyb2", "Spn7MLyb2", "Spn8MLyb2", "Spn9MLyb2"]
        channels_out += ["Spn1FLzb3", "Spn2FLzb3", "Spn3FLzb3", "Spn4FLzb3", "Spn5FLzb3", "Spn6FLzb3", "Spn7FLzb3", "Spn8FLzb3", "Spn9FLzb3"]
        channels_out += ["Spn1MLxb3", "Spn2MLxb3", "Spn3MLxb3", "Spn4MLxb3", "Spn5MLxb3", "Spn6MLxb3", "Spn7MLxb3", "Spn8MLxb3", "Spn9MLxb3"]
        channels_out += ["Spn1MLyb3", "Spn2MLyb3", "Spn3MLyb3", "Spn4MLyb3", "Spn5MLyb3", "Spn6MLyb3", "Spn7MLyb3", "Spn8MLyb3", "Spn9MLyb3"]
        channels_out += ["TipClrnc1", "TipClrnc2", "TipClrnc3"]
        channels_out += ["RtFldCp", "RtFldCt"]
        channels_out += ["RtFldFxh", "RtFldFyh", "RtFldFzh", "RtFldMxh", "RtFldMyh", "RtFldMzh"]
        channels_out += ["RotSpeed", "GenSpeed", "NacYaw", "Azimuth"]
        channels_out += ["GenPwr", "GenTq", "BldPitch1", "BldPitch2", "BldPitch3"]
        channels_out += ["Wind1VelX", "Wind1VelY", "Wind1VelZ"]
        channels_out += ["RtVAvgxh", "RtVAvgyh", "RtVAvgzh"]
        channels_out += ["TwrBsFxt",  "TwrBsFyt", "TwrBsFzt", "TwrBsMxt",  "TwrBsMyt", "TwrBsMzt"]
        channels_out += ["YawBrFxp", "YawBrFyp", "YawBrFzp", "YawBrMxp", "YawBrMyp", "YawBrMzp"]
        channels_out += ["TwHt1FLxt", "TwHt2FLxt", "TwHt3FLxt", "TwHt4FLxt", "TwHt5FLxt", "TwHt6FLxt", "TwHt7FLxt", "TwHt8FLxt", "TwHt9FLxt"]
        channels_out += ["TwHt1FLyt", "TwHt2FLyt", "TwHt3FLyt", "TwHt4FLyt", "TwHt5FLyt", "TwHt6FLyt", "TwHt7FLyt", "TwHt8FLyt", "TwHt9FLyt"]
        channels_out += ["TwHt1FLzt", "TwHt2FLzt", "TwHt3FLzt", "TwHt4FLzt", "TwHt5FLzt", "TwHt6FLzt", "TwHt7FLzt", "TwHt8FLzt", "TwHt9FLzt"]
        channels_out += ["TwHt1MLxt", "TwHt2MLxt", "TwHt3MLxt", "TwHt4MLxt", "TwHt5MLxt", "TwHt6MLxt", "TwHt7MLxt", "TwHt8MLxt", "TwHt9MLxt"]
        channels_out += ["TwHt1MLyt", "TwHt2MLyt", "TwHt3MLyt", "TwHt4MLyt", "TwHt5MLyt", "TwHt6MLyt", "TwHt7MLyt", "TwHt8MLyt", "TwHt9MLyt"]
        channels_out += ["TwHt1MLzt", "TwHt2MLzt", "TwHt3MLzt", "TwHt4MLzt", "TwHt5MLzt", "TwHt6MLzt", "TwHt7MLzt", "TwHt8MLzt", "TwHt9MLzt"]
        channels_out += ["RotThrust", "LSShftFxs", "LSShftFys", "LSShftFzs", "LSShftFxa", "LSShftFya", "LSShftFza"]
        channels_out += ["RotTorq", "LSSTipMxs", "LSSTipMys", "LSSTipMzs", "LSSTipMxa", "LSSTipMya", "LSSTipMza"]
        channels_out += ["B1N1Alpha", "B1N2Alpha", "B1N3Alpha", "B1N4Alpha", "B1N5Alpha", "B1N6Alpha", "B1N7Alpha", "B1N8Alpha", "B1N9Alpha", "B2N1Alpha", "B2N2Alpha", "B2N3Alpha", "B2N4Alpha", "B2N5Alpha", "B2N6Alpha", "B2N7Alpha", "B2N8Alpha","B2N9Alpha"]
        channels_out += ["PtfmSurge", "PtfmSway", "PtfmHeave", "PtfmRoll", "PtfmPitch", "PtfmYaw"]
        channels_out += ["NcIMUTAxs","NcIMUTAys","NcIMUTAzs","NcIMURAxs","NcIMURAys","NcIMURAzs"]
        if self.n_blades == 3:
            channels_out += ["TipDxc3", "TipDyc3", "TipDzc3", "RootMxc3", "RootMyc3", "RootMzc3", "TipDxb3", "TipDyb3", "TipDzb3", "RootMxb3",
                             "RootMyb3", "RootMzb3", "RootFxc3", "RootFyc3", "RootFzc3", "RootFxb3", "RootFyb3", "RootFzb3", "BldPitch3"]
            channels_out += ["B3N1Alpha", "B3N2Alpha", "B3N3Alpha", "B3N4Alpha", "B3N5Alpha", "B3N6Alpha", "B3N7Alpha", "B3N8Alpha", "B3N9Alpha"]

        # Channels for distributed aerodynamic control
        if self.options['modeling_options']['ROSCO']['Flp_Mode']:   # we're doing flap control
            channels_out += ['BLFLAP1', 'BLFLAP2', 'BLFLAP3']

        # Channels for wave outputs
        if modopt['flags']['offshore']:
            channels_out += ["Wave1Elev","WavesF1xi","WavesF1zi","WavesM1yi"]
            channels_out += ["WavesF2xi","WavesF2yi","WavesF2zi","WavesM2xi","WavesM2yi","WavesM2zi"]

        # Channels for monopile-based structure
        if modopt['flags']['monopile']:
            if modopt['OpenFAST']['simulation']['CompSub']:
                k=1
                for i in range(len(self.Z_out_SD_mpl)):
                    if k==len(fst_vt['SubDyn']['NodeCnt']):
                        Node=2
                    else:
                        Node=1
                    channels_out += ["M" + str(k) + "N" + str(Node) + "FKxe"]
                    channels_out += ["M" + str(k) + "N" + str(Node) + "FKye"]
                    channels_out += ["M" + str(k) + "N" + str(Node) + "FKze"]
                    channels_out += ["M" + str(k) + "N" + str(Node) + "MKxe"]
                    channels_out += ["M" + str(k) + "N" + str(Node) + "MKye"]
                    channels_out += ["M" + str(k) + "N" + str(Node) + "MKze"]
                    k+=1
                channels_out += ['ReactFXss', 'ReactFYss', 'ReactFZss', 'ReactMXss', 'ReactMYss', 'ReactMZss']
            else:
                raise Exception('CompSub must be 1 in the modeling options to run SubDyn and compute monopile loads')

        # Floating output channels
        if modopt['flags']['floating']:
            channels_out += ["PtfmPitch", "PtfmRoll", "PtfmYaw", "PtfmSurge", "PtfmSway", "PtfmHeave"]

        # Structural Control Channels
        if modopt['flags']['TMDs']:
            for i_SStC in range(len(fst_vt['SStC'])):
                channels_out += [f'SStC{i_SStC+1}_Fxi',f'SStC{i_SStC+1}_Fyi',f'SStC{i_SStC+1}_Fzi']
                channels_out += [f'SStC{i_SStC+1}_Mxi',f'SStC{i_SStC+1}_Myi',f'SStC{i_SStC+1}_Mzi']
                channels_out += [f'SStC{i_SStC+1}_Fxl',f'SStC{i_SStC+1}_Fyl',f'SStC{i_SStC+1}_Fzl']
                channels_out += [f'SStC{i_SStC+1}_Mxl',f'SStC{i_SStC+1}_Myl',f'SStC{i_SStC+1}_Mzl']
                channels_out += [f'SStC{i_SStC+1}_XQ',f'SStC{i_SStC+1}_YQ',f'SStC{i_SStC+1}_ZQ']
                channels_out += [f'SStC{i_SStC+1}_XQD',f'SStC{i_SStC+1}_YQD',f'SStC{i_SStC+1}_ZQD']

            for i_TStC in range(len(fst_vt['TStC'])):
                channels_out += [f'TStC{i_TStC+1}_Fxi',f'TStC{i_TStC+1}_Fyi',f'TStC{i_TStC+1}_Fzi']
                channels_out += [f'TStC{i_TStC+1}_Mxi',f'TStC{i_TStC+1}_Myi',f'TStC{i_TStC+1}_Mzi']
                channels_out += [f'TStC{i_TStC+1}_Fxl',f'TStC{i_TStC+1}_Fyl',f'TStC{i_TStC+1}_Fzl']
                channels_out += [f'TStC{i_TStC+1}_Mxl',f'TStC{i_TStC+1}_Myl',f'TStC{i_TStC+1}_Mzl']
                channels_out += [f'TStC{i_TStC+1}_XQ',f'TStC{i_TStC+1}_YQ',f'TStC{i_TStC+1}_ZQ']
                channels_out += [f'TStC{i_TStC+1}_XQD',f'TStC{i_TStC+1}_YQD',f'TStC{i_TStC+1}_ZQD']


        channels = {}
        for var in channels_out:
            channels[var] = True

        return channels

    def run_FAST(self, inputs, discrete_inputs, fst_vt):

        modopt = self.options['modeling_options']
        modopt_dir = os.path.dirname(modopt['fname_input_modeling'])
        DLCs = modopt['DLC_driver']['DLCs']
        # Initialize the DLC generator
        cut_in = float(inputs['V_cutin'])
        cut_out = float(inputs['V_cutout'])
        rated = float(inputs['Vrated'])
        ws_class = discrete_inputs['turbine_class']
        wt_class = discrete_inputs['turbulence_class']
        hub_height = float(inputs['hub_height'])
        rotorD = float(inputs['Rtip'])*2.
        PLExp = float(inputs['shearExp'])
        fix_wind_seeds = modopt['DLC_driver']['fix_wind_seeds']
        fix_wave_seeds = modopt['DLC_driver']['fix_wave_seeds']
        metocean = modopt['DLC_driver']['metocean_conditions']
        
        # Set initial rotor speed and pitch if the WT operates in this DLC and available,
        # otherwise set pitch to 90 deg and rotor speed to 0 rpm when not operating
        # set rotor speed to rated and pitch to 15 deg if operating
        if self.options['modeling_options']['OpenFAST']['from_openfast']:
            reg_traj = os.path.join(self.modopt_dir,self.options['modeling_options']['OpenFAST']['regulation_trajectory'])
            if os.path.isfile(reg_traj):
                data = load_yaml(reg_traj)
                cases = data['cases']
                U_interp = [case["configuration"]["wind_speed"] for case in cases]
                pitch_interp = [case["configuration"]["pitch"] for case in cases]
                rot_speed_interp = [case["configuration"]["rotor_speed"] for case in cases]
                Ct_aero_interp = [case["outputs"]["integrated"]["ct"] for case in cases]
            else:
                logger.warning("A yaml file with rotor speed, pitch, and Ct is required in modeling options->OpenFAST->regulation_trajectory.",
                        " This file does not exist. Check WEIS example 02 for a template file")
                U_interp = np.arange(cut_in, cut_out)
                pitch_interp = np.ones_like(U_interp) * 5. # fixed initial pitch at 5 deg
                rot_speed_interp = np.ones_like(U_interp) * 5. # fixed initial omega at 5 rpm
                Ct_aero_interp = np.ones_like(U_interp) * 0.7 # fixed initial ct at 0.7
        else:
            U_interp = inputs['U']
            pitch_interp = inputs['pitch']
            rot_speed_interp = inputs['Omega']
            Ct_aero_interp = inputs['Ct_aero']
        
        tau1_const_interp = np.zeros_like(Ct_aero_interp)
        for i in range(len(Ct_aero_interp)):
            a = 1. / 2. * (1. - np.sqrt(1. - np.min([Ct_aero_interp[i],1])))    # don't allow Ct_aero > 1
            tau1_const_interp[i] = 1.1 / (1. - 1.3 * np.min([a, 0.5])) * inputs['Rtip'][0] / U_interp[i]

        initial_condition_table = {}
        initial_condition_table['U'] = U_interp
        initial_condition_table['pitch_initial'] = pitch_interp
        initial_condition_table['rot_speed_initial'] = rot_speed_interp
        initial_condition_table['Ct_aero'] = Ct_aero_interp
        initial_condition_table['tau1_const'] = tau1_const_interp
        
        
        dlc_generator = DLCGenerator(
            metocean,
            **{
                'ws_cut_in': cut_in, 
                'ws_cut_out':cut_out, 
                'MHK': modopt['flags']['marine_hydro'],
                'fix_wind_seeds': fix_wind_seeds,
                'fix_wave_seeds': fix_wave_seeds,                
            })
        # Generate cases from user inputs
        for i_DLC in range(len(DLCs)):
            DLCopt = DLCs[i_DLC]
            dlc_generator.generate(DLCopt['DLC'], DLCopt)

        # Initialize parametric inputs
        WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
        WindFile_name = [''] * dlc_generator.n_cases

        self.TMax = np.zeros(dlc_generator.n_cases)
        self.TStart = np.zeros(dlc_generator.n_cases)

        # fix hub height if MHK
        if modopt['flags']['marine_hydro']:
            # make grid span whole water depth for now, ref height will be actual hub height to get speed right
            # in deeper water, will need something better than this
            grid_height = 2. * np.abs(hub_height) - 1.e-3
            hub_height = grid_height/ 2
            ref_height = float(inputs['water_depth'] - np.abs(inputs['hub_height']))

            # Inflow wind wants these relative to sea bed
            fst_vt['InflowWind']['WindVziList'] = [ref_height]

        else:
            ref_height = hub_height


        for i_case in range(dlc_generator.n_cases):
            if dlc_generator.cases[i_case].turbulent_wind:
                # Assign values common to all DLCs
                # Wind turbulence class
                if dlc_generator.cases[i_case].IECturbc > 0:    # use custom TI for DLC case
                    dlc_generator.cases[i_case].IECturbc = str(dlc_generator.cases[i_case].IECturbc)
                    dlc_generator.cases[i_case].IEC_WindType = 'NTM'        # must use NTM for custom TI
                else:
                    dlc_generator.cases[i_case].IECturbc = wt_class
                # Reference height for wind speed
                if not dlc_generator.cases[i_case].RefHt:   # default RefHt is 0, use hub_height if not set
                    dlc_generator.cases[i_case].RefHt = ref_height
                # Center of wind grid (TurbSim confusingly calls it HubHt)
                if not dlc_generator.cases[i_case].HubHt:   # default HubHt is 0, use hub_height if not set
                    dlc_generator.cases[i_case].HubHt = np.abs(hub_height)

                if not dlc_generator.cases[i_case].GridHeight:   # default GridHeight is 0, use hub_height if not set
                    dlc_generator.cases[i_case].GridHeight =  2. * np.abs(hub_height) - 1.e-3

                if not dlc_generator.cases[i_case].GridWidth:   # default GridWidth is 0, use hub_height if not set
                    dlc_generator.cases[i_case].GridWidth =  2. * hub_height - 1.e-3

                # Power law exponent of wind shear
                if dlc_generator.cases[i_case].PLExp < 0:    # use PLExp based on environment options (shear_exp), otherwise use custom DLC PLExp
                    dlc_generator.cases[i_case].PLExp = PLExp
                # Length of wind grids
                dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].total_time

        # Generate wind files
        if MPI and not self.options['opt_options']['driver']['design_of_experiments']['flag']:
            # mpi comm management
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            sub_ranks = self.mpi_comm_map_down[rank]
            size = len(sub_ranks)

            N_cases = dlc_generator.n_cases # total number of cases
            N_loops = int(np.ceil(float(N_cases)/float(size)))  # number of times function calls need to "loop"
            # iterate loops
            for i in range(N_loops):
                idx_s = i*size
                idx_e = min((i+1)*size, N_cases)

                for idx, i_case in enumerate(np.arange(idx_s,idx_e)):
                    data = [partial(generate_wind_files, dlc_generator, self.FAST_namingOut, self.wind_directory, rotorD, hub_height, self.turbsim_exe), i_case]
                    rank_j = sub_ranks[idx]
                    comm.send(data, dest=rank_j, tag=0)

                for idx, i_case in enumerate(np.arange(idx_s, idx_e)):
                    rank_j = sub_ranks[idx]
                    WindFile_type[i_case] , WindFile_name[i_case] = comm.recv(source=rank_j, tag=1)
        else:
            for i_case in range(dlc_generator.n_cases):
                WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
                    dlc_generator, self.FAST_namingOut, self.wind_directory, rotorD, hub_height, self.turbsim_exe, i_case)


        # Apply olaf settings, should be similar to above?
        if dlc_generator.default_options['wake_mod'] == 3:  # OLAF is used 
            apply_olaf_parameters(dlc_generator,fst_vt)
                    
        # Parameteric inputs
        case_name = []
        case_list = []
        for i_case, case_inputs in enumerate(dlc_generator.openfast_case_inputs):
            # Generate case list for DLC i
            dlc_label = DLCs[i_case]['DLC']
            case_list_i, case_name_i = CaseGen_General(case_inputs, self.FAST_runDirectory, self.FAST_InputFile, filename_ext=f'_DLC{dlc_label}_{i_case}')
            # Add DLC to case names
            case_name_i = [f'DLC{dlc_label}_{i_case}_{cni}' for cni in case_name_i]   # TODO: discuss case labeling with stakeholders
            
            # Extend lists of cases
            case_list.extend(case_list_i)
            case_name.extend(case_name_i)

        # Apply wind files to case_list (this info will be in combined case matrix, but not individual DLCs)
        for case_i, wt, wf in zip(case_list,WindFile_type,WindFile_name):
            case_i[('InflowWind','WindType')] = wt
            case_i[('InflowWind','FileName_Uni')] = wf
            case_i[('InflowWind','FileName_BTS')] = wf

        # Save some case info
        self.TMax = [c.total_time for c in dlc_generator.cases]
        self.TStart = [c.transient_time for c in dlc_generator.cases]
        dlc_label = [c.label for c in dlc_generator.cases]


        # Merge various cases into single case matrix
        case_df = pd.DataFrame(case_list)
        case_df.index = case_name
        # Add case name and dlc label to front for readability
        case_df.insert(0,'DLC',dlc_label)
        case_df.insert(0,'case_name',case_name)
        text_table = case_df.to_string(index=False)

        # Write the text table to a yaml, text file
        write_yaml(case_df.to_dict(),os.path.join(self.FAST_runDirectory,'case_matrix_combined.yaml'))
        with open(os.path.join(self.FAST_runDirectory,'case_matrix_combined.txt'), 'w') as file:
            file.write(text_table)
            
            
        channels= self.output_channels(fst_vt)

        # Delete the extra case_inputs because they don't play nicely with aeroelasticse
        for case in case_list:
            for key in list(case):
                if key[0] in ['DLC','TurbSim']:
                    del case[key]
        
        
        # FAST wrapper setup
        # JJ->DZ: here is the first point in logic for linearization
        if modopt['OpenFAST_Linear']['flag']:
            linearization_options               = modopt['OpenFAST_Linear']['linearization']

            # Use openfast binary until library works
            fastBatch                           = LinearFAST(**linearization_options)
            fastBatch.FAST_runDirectory         = self.FAST_runDirectory
            fastBatch.use_exe                   = True      # linearization not working with library
            fastBatch.fst_vt                    = fst_vt
            fastBatch.cores                     = self.cores

            lin_case_list, lin_case_name        = fastBatch.gen_linear_cases(inputs)
            fastBatch.case_list                 = lin_case_list
            fastBatch.case_name_list            = lin_case_name

            # Save this list of linear cases for making linear model, not the best solution, but it works
            self.lin_case_name                  = lin_case_name
        else:
            fastBatch                           = fastwrap.runFAST_pywrapper_batch()
            fastBatch.FAST_runDirectory         = self.FAST_runDirectory
            fastBatch.case_list                 = case_list
            fastBatch.case_name_list            = case_name     
            fastBatch.use_exe                   = modopt['General']['openfast_configuration']['use_exe']
        
        fastBatch.channels          = channels
        fastBatch.FAST_InputFile    = self.FAST_InputFile
        fastBatch.fst_vt            = fst_vt
        fastBatch.keep_time         = modopt['General']['openfast_configuration']['keep_time']
        fastBatch.post              = FAST_IO_timeseries
        fastBatch.allow_fails       = modopt['General']['openfast_configuration']['allow_fails']
        fastBatch.fail_value        = modopt['General']['openfast_configuration']['fail_value']
        fastBatch.write_stdout      = modopt['General']['openfast_configuration']['write_stdout']
        if self.FAST_exe_user is not None:
            fastBatch.FAST_exe      = self.FAST_exe_user
        if self.FAST_lib_user is not None:
            fastBatch.FAST_lib      = self.FAST_lib_user

        fastBatch.overwrite_outfiles = True  #<--- Debugging only, set to False to prevent OpenFAST from running if the .outb already exists

        # Initialize fatigue channels and setings
        # TODO: Stress Concentration Factor?
        magnitude_channels = dict( fastwrap.magnitude_channels_default )
        fatigue_channels =  dict( fastwrap.fatigue_channels_default )

        # Nacelle accelleration
        magnitude_channels['NcIMUTA'] = ['NcIMUTAxs','NcIMUTAys','NcIMUTAzs']

        # Blade fatigue: spar caps at the root (upper & lower?), TE at max chord
        # Convert ultstress and S_intercept values to kPa with 1e-3 factor
        if not modopt['OpenFAST']['from_openfast']:
            for u in ['U','L']:
                blade_fatigue_root = FatigueParams(load2stress=1.0,
                                                   slope=inputs[f'blade_spar{u}_wohlerexp'],
                                                   ultimate_stress=1e-3*inputs[f'blade_spar{u}_ultstress'],
                                                   S_intercept=1e-3*inputs[f'blade_spar{u}_wohlerA'])
                blade_fatigue_te = FatigueParams(load2stress=1.0,
                                                 slope=inputs[f'blade_te{u}_wohlerexp'],
                                                 ultimate_stress=1e-3*inputs[f'blade_te{u}_ultstress'],
                                                 S_intercept=1e-3*inputs[f'blade_te{u}_wohlerA'])
                
                for k in range(1,self.n_blades+1):
                    blade_root_Fz = blade_fatigue_root.copy()
                    blade_root_Fz.load2stress = inputs[f'blade_root_spar{u}_load2stress'][2]
                    fatigue_channels[f'RootSpar{u}_Fzb{k}'] = blade_root_Fz
                    magnitude_channels[f'RootSpar{u}_Fzb{k}'] = [f'RootFzb{k}']

                    blade_root_Mx = blade_fatigue_root.copy()
                    blade_root_Mx.load2stress = inputs[f'blade_root_spar{u}_load2stress'][3]
                    fatigue_channels[f'RootSpar{u}_Mxb{k}'] = blade_root_Mx
                    magnitude_channels[f'RootSpar{u}_Mxb{k}'] = [f'RootMxb{k}']

                    blade_root_My = blade_fatigue_root.copy()
                    blade_root_My.load2stress = inputs[f'blade_root_spar{u}_load2stress'][4]
                    fatigue_channels[f'RootSpar{u}_Myb{k}'] = blade_root_My
                    magnitude_channels[f'RootSpar{u}_Myb{k}'] = [f'RootMyb{k}']

                    blade_maxc_Fz = blade_fatigue_te.copy()
                    blade_maxc_Fz.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][2]
                    fatigue_channels[f'Spn2te{u}_FLzb{k}'] = blade_maxc_Fz
                    magnitude_channels[f'Spn2te{u}_FLzb{k}'] = [f'Spn2FLzb{k}']

                    blade_maxc_Mx = blade_fatigue_te.copy()
                    blade_maxc_Mx.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][3]
                    fatigue_channels[f'Spn2te{u}_MLxb{k}'] = blade_maxc_Mx
                    magnitude_channels[f'Spn2te{u}_MLxb{k}'] = [f'Spn2MLxb{k}']

                    blade_maxc_My = blade_fatigue_te.copy()
                    blade_maxc_My.load2stress = inputs[f'blade_maxc_te{u}_load2stress'][4]
                    fatigue_channels[f'Spn2te{u}_MLyb{k}'] = blade_maxc_My
                    magnitude_channels[f'Spn2te{u}_MLyb{k}'] = [f'Spn2MLyb{k}']

            # Low speed shaft fatigue
            # Convert ultstress and S_intercept values to kPa with 1e-3 factor
            # Use DNV RP C203 Fatigue steel B1 in air
            lss_fatigue = FatigueParams(load2stress=1.0,
                                        dnv_name='B1',
                                        dnv_type='air',
                                        ultimate_stress=1e-3*inputs['lss_ultstress'],
                                        S_intercept=1e-3*inputs['lss_wohlerA'])
            for s in ['Ax','Sh']:
                sstr = 'axial' if s=='Ax' else 'shear'
                for ik, k in enumerate(['F','M']):
                    for ix, x in enumerate(['x','yz']):
                        idx = 3*ik+ix
                        lss_fatigue_ii = lss_fatigue.copy()
                        lss_fatigue_ii.load2stress = inputs[f'lss_{sstr}_load2stress'][idx]
                        fatigue_channels[f'LSShft{s}{k}{x}a'] = lss_fatigue_ii
                        if ix==0:
                            magnitude_channels[f'LSShft{s}{k}{x}a'] = ['RotThrust'] if ik==0 else ['RotTorq']
                        else:
                            magnitude_channels[f'LSShft{s}{k}{x}a'] = ['LSShftFya', 'LSShftFza'] if ik==0 else ['LSSTipMya', 'LSSTipMza']

            # Aero-only hub loads
            magnitude_channels["RtFldF"] = ["RtFldFxh", "RtFldFyh", "RtFldFzh"]
            magnitude_channels["RtFldM"] = ["RtFldMxh", "RtFldMyh", "RtFldMzh"]

            # Fatigue at the tower base
            # Convert ultstress and S_intercept values to kPa with 1e-3 factor
            # Use DNV RP C203 Fatigue steel D in air
            tower_fatigue_base = FatigueParams(load2stress=1.0,
                                               dnv_name='D',
                                               dnv_type='air',
                                               ultimate_stress=1e-3*inputs['tower_ultstress'][0],
                                               S_intercept=1e-3*inputs['tower_wohlerA'][0])
            for s in ['Ax','Sh']:
                sstr = 'axial' if s=='Ax' else 'shear'
                for ik, k in enumerate(['F','M']):
                    for ix, x in enumerate(['xy','z']):
                        idx = 3*ik+2*ix
                        tower_fatigue_ii = tower_fatigue_base.copy()
                        tower_fatigue_ii.load2stress = inputs[f'tower_{sstr}_load2stress'][0,idx]
                        fatigue_channels[f'TwrBs{s}{k}{x}t'] = tower_fatigue_ii
                        magnitude_channels[f'TwrBs{s}{k}{x}t'] = [f'TwrBs{k}{x}t'] if x=='z' else [f'TwrBs{k}xt', f'TwrBs{k}yt']

            # Fatigue at monopile base (mudline)
            # No need to convert to kPa here since SubDyn reports in N already
            # Use DNV RP C203 Fatigue steel D in seawater
            if modopt['flags']['monopile']:
                monopile_fatigue_base = FatigueParams(load2stress=1.0,
                                                      dnv_name='D',
                                                      dnv_type='sea',
                                                      ultimate_stress=inputs['monopile_ultstress'][0],
                                                      S_intercept=inputs['monopile_wohlerA'][0])
                for s in ['Ax','Sh']:
                    sstr = 'axial' if s=='Ax' else 'shear'
                    for ik, k in enumerate(['F','M']):
                        for ix, x in enumerate(['xy','z']):
                            idx = 3*ik+2*ix
                            monopile_fatigue_ii = monopile_fatigue_base.copy()
                            monopile_fatigue_ii.load2stress = inputs[f'monopile_{sstr}_load2stress'][0,idx]
                            fatigue_channels[f'M1N1{s}{k}K{x}e'] = monopile_fatigue_ii
                            magnitude_channels[f'M1N1{s}{k}K{x}e'] = [f'M1N1{k}K{x}e'] if x=='z' else [f'M1N1{k}Kxe', f'M1N1{k}Kye']

        # Store settings
        fastBatch.goodman            = modopt['General']['goodman_correction'] # Where does this get placed in schema?
        fastBatch.fatigue_channels   = fatigue_channels
        fastBatch.magnitude_channels = magnitude_channels

        # Run FAST
        if self.mpi_run and not self.options['opt_options']['driver']['design_of_experiments']['flag']:
            self.cruncher = fastBatch.run_mpi(self.mpi_comm_map_down)
        else:
            if self.cores == 1:
                self.cruncher = fastBatch.run_serial()
            else:
                self.cruncher = fastBatch.run_multi(self.cores)

        self.fst_vt = fst_vt
        self.of_inumber = self.of_inumber + 1
        sys.stdout.flush()

        return case_list, case_name, dlc_generator

    def post_process(self, case_list, dlc_generator, inputs, discrete_inputs, outputs, discrete_outputs):
        modopt = self.options['modeling_options']

        # Analysis
        if self.options['modeling_options']['flags']['blade'] and bool(self.fst_vt['Fst']['CompAero']):
            outputs = self.get_blade_loading(inputs, outputs)
            
        if self.options['modeling_options']['flags']['tower']:
            outputs = self.get_tower_loading(inputs, outputs)
            
        # SubDyn is only supported in Level3: linearization in OpenFAST will be available in 3.0.0
        if modopt['flags']['monopile']:
            outputs = self.get_monopile_loading(inputs, outputs)

        # If DLC 1.1 not used, calculate_AEP will just compute average power of simulations
        outputs = self.calculate_AEP(case_list, dlc_generator, discrete_inputs, outputs)

        outputs = self.get_weighted_DELs(dlc_generator, inputs, discrete_inputs, outputs)
        
        outputs = self.get_control_measures(inputs, outputs)

        if modopt['flags']['floating'] or (modopt['OpenFAST']['from_openfast'] and self.fst_vt['Fst']['CompMooring']>0):
            outputs = self.get_floating_measures(inputs, outputs)

        # Did any OpenFAST runs fail?
        if modopt['OpenFAST']['flag']:
            if any(self.cruncher.summary_stats['openfast_failed']['mean'] > 0):
                outputs['openfast_failed'] = 2

        # # Did any OpenFAST runs fail?
        # if any(summary_stats['openfast_failed']['mean'] > 0):
        #     outputs['openfast_failed'] = 2

        # Save Data
        if modopt['General']['openfast_configuration']['save_timeseries']:
            self.save_timeseries()

        if modopt['General']['openfast_configuration']['save_iterations']:
            self.save_iterations(discrete_outputs)

        # Open loop to closed loop error, move this to before save_timeseries when finished
        if modopt['OL2CL']['flag']:
            outputs = self.get_OL2CL_error(outputs)

    def get_blade_loading(self, inputs, outputs):
        """
        Find the spanwise loading along the blade span.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        extreme_table : dict
        """
        sum_stats = self.cruncher.summary_stats
        extreme_table = self.cruncher.extremes
        
        # Determine maximum deflection magnitudes
        if self.n_blades == 2:
            defl_mag = [max(sum_stats['TipDxc1']['max']), max(sum_stats['TipDxc2']['max'])]
        else:
            defl_mag = [max(sum_stats['TipDxc1']['max']), max(sum_stats['TipDxc2']['max']), max(sum_stats['TipDxc3']['max'])]
        # Get the maximum out of plane blade deflection
        outputs["max_TipDxc"] = np.max(defl_mag)

        # Return moments around x and y and axial force along blade span at instance of largest flapwise bending moment at each node
        My_chans = ["RootMyb", "Spn1MLyb", "Spn2MLyb", "Spn3MLyb", "Spn4MLyb", "Spn5MLyb", "Spn6MLyb", "Spn7MLyb", "Spn8MLyb", "Spn9MLyb"]
        Mx_chans = ["RootMxb", "Spn1MLxb", "Spn2MLxb", "Spn3MLxb", "Spn4MLxb", "Spn5MLxb", "Spn6MLxb", "Spn7MLxb", "Spn8MLxb", "Spn9MLxb"]
        Fz_chans = ["RootFzb", "Spn1FLzb", "Spn2FLzb", "Spn3FLzb", "Spn4FLzb", "Spn5FLzb", "Spn6FLzb", "Spn7FLzb", "Spn8FLzb", "Spn9FLzb"]
            
        Fz = []
        Mx = []
        My = []
        for My_chan,Mx_chan,Fz_chan in zip(My_chans, Mx_chans, Fz_chans):
            if self.n_blades == 2:
                bld_idx_max = np.argmax([max(sum_stats[My_chan+'1']['max']), max(sum_stats[My_chan+'2']['max'])])
            else:
                bld_idx_max = np.argmax([max(sum_stats[My_chan+'1']['max']), max(sum_stats[My_chan+'2']['max']), max(sum_stats[My_chan+'3']['max'])])
            My_max_chan = My_chan + str(bld_idx_max+1)
            My.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][My_chan+str(bld_idx_max+1)])
            Mx.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][Mx_chan+str(bld_idx_max+1)])
            Fz.append(extreme_table[My_max_chan][np.argmax(sum_stats[My_max_chan]['max'])][Fz_chan+str(bld_idx_max+1)])


        if np.any(np.isnan(Fz)):
            logger.warning('WARNING: nans found in Fz extremes')
            Fz[np.isnan(Fz)] = 0.0
        if np.any(np.isnan(Mx)):
            logger.warning('WARNING: nans found in Mx extremes')
            Mx[np.isnan(Mx)] = 0.0
        if np.any(np.isnan(My)):
            logger.warning('WARNING: nans found in My extremes')
            My[np.isnan(My)] = 0.0
        spline_Fz = PchipInterpolator(np.hstack((self.R_out_ED_bl, inputs['Rtip'])), np.hstack((Fz, 0.)))
        spline_Mx = PchipInterpolator(np.hstack((self.R_out_ED_bl, inputs['Rtip'])), np.hstack((Mx, 0.)))
        spline_My = PchipInterpolator(np.hstack((self.R_out_ED_bl, inputs['Rtip'])), np.hstack((My, 0.)))

        r = inputs['r']
        Fz_out = spline_Fz(r).flatten()
        Mx_out = spline_Mx(r).flatten()
        My_out = spline_My(r).flatten()

        outputs['blade_maxTD_Mx'] = Mx_out
        outputs['blade_maxTD_My'] = My_out
        outputs['blade_maxTD_Fz'] = Fz_out

        # Determine maximum root moment
        if self.n_blades == 2:
            blade_root_flap_moment = max([max(sum_stats['RootMyb1']['max']), max(sum_stats['RootMyb2']['max'])])
            blade_root_oop_moment  = max([max(sum_stats['RootMyc1']['max']), max(sum_stats['RootMyc2']['max'])])
            blade_root_tors_moment  = max([max(sum_stats['RootMzb1']['max']), max(sum_stats['RootMzb2']['max'])])
        else:
            blade_root_flap_moment = max([max(sum_stats['RootMyb1']['max']), max(sum_stats['RootMyb2']['max']), max(sum_stats['RootMyb3']['max'])])
            blade_root_oop_moment  = max([max(sum_stats['RootMyc1']['max']), max(sum_stats['RootMyc2']['max']), max(sum_stats['RootMyc3']['max'])])
            blade_root_tors_moment  = max([max(sum_stats['RootMzb1']['max']), max(sum_stats['RootMzb2']['max']), max(sum_stats['RootMzb3']['max'])])
        outputs['max_RootMyb'] = blade_root_flap_moment
        outputs['max_RootMyc'] = blade_root_oop_moment
        outputs['max_RootMzb'] = blade_root_tors_moment

        ## Get hub moments and forces in the non-rotating frame
        outputs['hub_Fxyz'] = np.array([extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['RotThrust'],
                                        extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['LSShftFys'],
                                        extreme_table['LSShftF'][np.argmax(sum_stats['LSShftF']['max'])]['LSShftFzs']])*1.e3
        outputs['hub_Mxyz'] = np.array([extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['RotTorq'],
                                        extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['LSSTipMys'],
                                        extreme_table['LSShftM'][np.argmax(sum_stats['LSShftM']['max'])]['LSSTipMzs']])*1.e3
        
        # Aero-only for WISDEM (outputs are in N and N-m)
        outputs['hub_Fxyz_aero'] = np.array([extreme_table['RtFldF'][np.argmax(sum_stats['RtFldF']['max'])]['RtFldFxh'],
                                             extreme_table['RtFldF'][np.argmax(sum_stats['RtFldF']['max'])]['RtFldFyh'],
                                             extreme_table['RtFldF'][np.argmax(sum_stats['RtFldF']['max'])]['RtFldFzh']])
        outputs['hub_Mxyz_aero'] = np.array([extreme_table['RtFldM'][np.argmax(sum_stats['RtFldM']['max'])]['RtFldMxh'],
                                             extreme_table['RtFldM'][np.argmax(sum_stats['RtFldM']['max'])]['RtFldMyh'],
                                             extreme_table['RtFldM'][np.argmax(sum_stats['RtFldM']['max'])]['RtFldMzh']])

        ## Post process aerodynamic data
        # Angles of attack - max, std, mean
        blade1_chans_aoa = ["B1N1Alpha", "B1N2Alpha", "B1N3Alpha", "B1N4Alpha", "B1N5Alpha", "B1N6Alpha", "B1N7Alpha", "B1N8Alpha", "B1N9Alpha"]
        blade2_chans_aoa = ["B2N1Alpha", "B2N2Alpha", "B2N3Alpha", "B2N4Alpha", "B2N5Alpha", "B2N6Alpha", "B2N7Alpha", "B2N8Alpha", "B2N9Alpha"]
        aoa_max_B1  = [np.max(sum_stats[var]['max'])    for var in blade1_chans_aoa]
        aoa_mean_B1 = [np.mean(sum_stats[var]['mean'])  for var in blade1_chans_aoa]
        aoa_std_B1  = [np.mean(sum_stats[var]['std'])   for var in blade1_chans_aoa]
        aoa_max_B2  = [np.max(sum_stats[var]['max'])    for var in blade2_chans_aoa]
        aoa_mean_B2 = [np.mean(sum_stats[var]['mean'])  for var in blade2_chans_aoa]
        aoa_std_B2  = [np.mean(sum_stats[var]['std'])   for var in blade2_chans_aoa]
        if self.n_blades == 2:
            spline_aoa_max      = PchipInterpolator(self.R_out_AD, np.max([aoa_max_B1, aoa_max_B2], axis=0))
            spline_aoa_std      = PchipInterpolator(self.R_out_AD, np.mean([aoa_std_B1, aoa_std_B2], axis=0))
            spline_aoa_mean     = PchipInterpolator(self.R_out_AD, np.mean([aoa_mean_B1, aoa_mean_B2], axis=0))
        elif self.n_blades == 3:
            blade3_chans_aoa    = ["B3N1Alpha", "B3N2Alpha", "B3N3Alpha", "B3N4Alpha", "B3N5Alpha", "B3N6Alpha", "B3N7Alpha", "B3N8Alpha", "B3N9Alpha"]
            aoa_max_B3          = [np.max(sum_stats[var]['max'])    for var in blade3_chans_aoa]
            aoa_mean_B3         = [np.mean(sum_stats[var]['mean'])  for var in blade3_chans_aoa]
            aoa_std_B3          = [np.mean(sum_stats[var]['std'])   for var in blade3_chans_aoa]
            spline_aoa_max      = PchipInterpolator(self.R_out_AD, np.max([aoa_max_B1, aoa_max_B2, aoa_max_B3], axis=0))
            spline_aoa_std      = PchipInterpolator(self.R_out_AD, np.mean([aoa_max_B1, aoa_std_B2, aoa_std_B3], axis=0))
            spline_aoa_mean     = PchipInterpolator(self.R_out_AD, np.mean([aoa_mean_B1, aoa_mean_B2, aoa_mean_B3], axis=0))
        else:
            raise Exception('The calculations only support 2 or 3 bladed rotors')

        outputs['max_aoa']  = spline_aoa_max(r)
        outputs['std_aoa']  = spline_aoa_std(r)
        outputs['mean_aoa'] = spline_aoa_mean(r)

        return outputs

    def get_tower_loading(self, inputs, outputs):
        """
        Find the loading along the tower height.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        extreme_table : dict
        """
        sum_stats = self.cruncher.summary_stats
        extreme_table = self.cruncher.extremes

        tower_chans_Fx = ["TwrBsFxt", "TwHt1FLxt", "TwHt2FLxt", "TwHt3FLxt", "TwHt4FLxt", "TwHt5FLxt", "TwHt6FLxt", "TwHt7FLxt", "TwHt8FLxt", "TwHt9FLxt", "YawBrFxp"]
        tower_chans_Fy = ["TwrBsFyt", "TwHt1FLyt", "TwHt2FLyt", "TwHt3FLyt", "TwHt4FLyt", "TwHt5FLyt", "TwHt6FLyt", "TwHt7FLyt", "TwHt8FLyt", "TwHt9FLyt", "YawBrFyp"]
        tower_chans_Fz = ["TwrBsFzt", "TwHt1FLzt", "TwHt2FLzt", "TwHt3FLzt", "TwHt4FLzt", "TwHt5FLzt", "TwHt6FLzt", "TwHt7FLzt", "TwHt8FLzt", "TwHt9FLzt", "YawBrFzp"]
        tower_chans_Mx = ["TwrBsMxt", "TwHt1MLxt", "TwHt2MLxt", "TwHt3MLxt", "TwHt4MLxt", "TwHt5MLxt", "TwHt6MLxt", "TwHt7MLxt", "TwHt8MLxt", "TwHt9MLxt", "YawBrMxp"]
        tower_chans_My = ["TwrBsMyt", "TwHt1MLyt", "TwHt2MLyt", "TwHt3MLyt", "TwHt4MLyt", "TwHt5MLyt", "TwHt6MLyt", "TwHt7MLyt", "TwHt8MLyt", "TwHt9MLyt", "YawBrMyp"]
        tower_chans_Mz = ["TwrBsMzt", "TwHt1MLzt", "TwHt2MLzt", "TwHt3MLzt", "TwHt4MLzt", "TwHt5MLzt", "TwHt6MLzt", "TwHt7MLzt", "TwHt8MLzt", "TwHt9MLzt", "YawBrMzp"]

        fatb_max_chan   = "TwrBsMyt"
        fatb_max = np.max(sum_stats[fatb_max_chan]['max'])
        idx      = np.argmax(sum_stats[fatb_max_chan]['max'])
        # Get the maximum fore-aft moment at tower base
        outputs["max_TwrBsMyt"] = fatb_max
        outputs["max_TwrBsMyt_ratio"] = fatb_max / self.options['opt_options']['constraints']['control']['Max_TwrBsMyt']['max']
        # Return forces and moments along tower height at instance of largest fore-aft tower base moment
        Fx = [extreme_table[fatb_max_chan][idx][var] for var in tower_chans_Fx]
        Fy = [extreme_table[fatb_max_chan][idx][var] for var in tower_chans_Fy]
        Fz = [extreme_table[fatb_max_chan][idx][var] for var in tower_chans_Fz]
        Mx = [extreme_table[fatb_max_chan][idx][var] for var in tower_chans_Mx]
        My = [extreme_table[fatb_max_chan][idx][var] for var in tower_chans_My]
        Mz = [extreme_table[fatb_max_chan][idx][var] for var in tower_chans_Mz]

        # Spline results on tower basic grid
        spline_Fx      = PchipInterpolator(self.Z_out_ED_twr, Fx)
        spline_Fy      = PchipInterpolator(self.Z_out_ED_twr, Fy)
        spline_Fz      = PchipInterpolator(self.Z_out_ED_twr, Fz)
        spline_Mx      = PchipInterpolator(self.Z_out_ED_twr, Mx)
        spline_My      = PchipInterpolator(self.Z_out_ED_twr, My)
        spline_Mz      = PchipInterpolator(self.Z_out_ED_twr, Mz)

        z_full = inputs['tower_z_full']
        z_sec, _ = util.nodal2sectional(z_full)
        z = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])

        outputs['tower_maxMy_Fx'] = spline_Fx(z)
        outputs['tower_maxMy_Fy'] = spline_Fy(z)
        outputs['tower_maxMy_Fz'] = spline_Fz(z)
        outputs['tower_maxMy_Mx'] = spline_Mx(z)
        outputs['tower_maxMy_My'] = spline_My(z)
        outputs['tower_maxMy_Mz'] = spline_Mz(z)
        
        return outputs

    def get_monopile_loading(self, inputs, outputs):
        """
        Find the loading along the monopile length.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        extreme_table : dict
        """
        sum_stats = self.cruncher.summary_stats
        extreme_table = self.cruncher.extremes

        monopile_chans_Fx = []
        monopile_chans_Fy = []
        monopile_chans_Fz = []
        monopile_chans_Mx = []
        monopile_chans_My = []
        monopile_chans_Mz = []
        k=1
        for i in range(len(self.Z_out_SD_mpl)):
            if k==len(self.fst_vt['SubDyn']['NodeCnt']):
                Node=2
            else:
                Node=1
            monopile_chans_Fx += [f"M{k}N{Node}FKxe"]
            monopile_chans_Fy += [f"M{k}N{Node}FKye"]
            monopile_chans_Fz += [f"M{k}N{Node}FKze"]
            monopile_chans_Mx += [f"M{k}N{Node}MKxe"]
            monopile_chans_My += [f"M{k}N{Node}MKye"]
            monopile_chans_Mz += [f"M{k}N{Node}MKze"]
            k+=1

        # # Get the maximum of signal M1N1MKye
        max_chan = "M1N1MKye"
        outputs["max_M1N1MKye"] = np.max(sum_stats[max_chan]['max'])
        idx = np.argmax(sum_stats[max_chan]['max'])
        # # Return forces and moments along monopile at instance of largest fore-aft tower base moment
        Fx = [extreme_table[max_chan][idx][var] for var in monopile_chans_Fx]
        Fy = [extreme_table[max_chan][idx][var] for var in monopile_chans_Fy]
        Fz = [extreme_table[max_chan][idx][var] for var in monopile_chans_Fz]
        Mx = [extreme_table[max_chan][idx][var] for var in monopile_chans_Mx]
        My = [extreme_table[max_chan][idx][var] for var in monopile_chans_My]
        Mz = [extreme_table[max_chan][idx][var] for var in monopile_chans_Mz]

        # # Spline results on grid of channel locations along the monopile
        spline_Fx      = PchipInterpolator(self.Z_out_SD_mpl, Fx)
        spline_Fy      = PchipInterpolator(self.Z_out_SD_mpl, Fy)
        spline_Fz      = PchipInterpolator(self.Z_out_SD_mpl, Fz)
        spline_Mx      = PchipInterpolator(self.Z_out_SD_mpl, Mx)
        spline_My      = PchipInterpolator(self.Z_out_SD_mpl, My)
        spline_Mz      = PchipInterpolator(self.Z_out_SD_mpl, Mz)

        z_full = inputs['monopile_z_full']
        z_sec, _ = util.nodal2sectional(z_full)
        z = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])

        # SubDyn reports in N, but ElastoDyn and units here report in kN, so scale by 0.001
        outputs['monopile_maxMy_Fx'] = 1e-3*spline_Fx(z)
        outputs['monopile_maxMy_Fy'] = 1e-3*spline_Fy(z)
        outputs['monopile_maxMy_Fz'] = 1e-3*spline_Fz(z)
        outputs['monopile_maxMy_Mx'] = 1e-3*spline_Mx(z)
        outputs['monopile_maxMy_My'] = 1e-3*spline_My(z)
        outputs['monopile_maxMy_Mz'] = 1e-3*spline_Mz(z)

        return outputs

    def calculate_AEP(self, case_list, dlc_generator, discrete_inputs, outputs):
        """
        Calculates annual energy production of the relevant DLCs in `case_list`.

        Parameters
        ----------
        sum_stats : pd.DataFrame
        case_list : list
        dlc_list : list
        """
        ## Get AEP and power curve

        # determine which dlc will be used for the powercurve calculations, allows using dlc 1.1 if specific power curve calculations were not run
        sum_stats = self.cruncher.summary_stats

        modopts = self.options['modeling_options']
        DLCs = [i_dlc['DLC'] for i_dlc in modopts['DLC_driver']['DLCs']]
        if 'AEP' in DLCs:
            DLC_label_for_AEP = 'AEP'
        else:
            DLC_label_for_AEP = '1.1'
            logger.warning('WARNING: DLC 1.1 is being used for AEP calculations.  Use the AEP DLC for more accurate wind modeling with constant TI.')

        idx_pwrcrv = []
        U = []
        for i_case in range(dlc_generator.n_cases):
            if dlc_generator.cases[i_case].label == DLC_label_for_AEP:
                idx_pwrcrv.append(i_case)
                U.append(dlc_generator.cases[i_case].URef)

        if len(U) > 0:
            self.cruncher.set_probability_turbine_class(U, discrete_inputs['turbine_class'], idx=idx_pwrcrv)

        if 'user_probability' in self.options['modeling_options']['DLC_driver']['metocean_conditions']:
            speed = modopts['DLC_driver']['metocean_conditions']['user_probability']['speed']
            user_probability = modopts['DLC_driver']['metocean_conditions']['user_probability']['probability']

            self.cruncher.set_probability_wind_distribution(U, 10., idx=idx_pwrcrv, kind='user', v_prob=speed, probability=user_probability)
            
        AEP, _ = self.cruncher.compute_aep("GenPwr", idx=idx_pwrcrv)
        outputs['AEP'] = AEP

        if len(idx_pwrcrv) > 0:
            sum_stats = sum_stats.iloc[idx_pwrcrv]
            outputs['V_out'] = np.unique(U)
            prob = self.cruncher.prob[idx_pwrcrv]
        else:
            outputs['V_out'] = dlc_generator.cases[0].URef
            prob = self.cruncher.prob
            logger.warning('WARNING: OpenFAST is not run using DLC AEP, 1.1, or 1.2. AEP cannot be estimated well. Using average power instead.')

        if len(U) == 1:
            logger.warning('WARNING: OpenFAST is run at a single wind speed. AEP cannot be estimated. Using average power instead.')
            
        # Calculate AEP and Performance Data
        outputs['Cp_out'] = np.sum(prob * sum_stats['RtFldCp']['mean'])
        outputs['Ct_out'] = np.sum(prob * sum_stats['RtFldCt']['mean'])
        outputs['Omega_out'] = np.sum(prob * sum_stats['RotSpeed']['mean'])
        outputs['pitch_out'] = np.sum(prob * sum_stats['BldPitch1']['mean'])
        if self.fst_vt['Fst']['CompServo'] == 1:
            outputs['P_out'] = np.sum(prob * sum_stats['GenPwr']['mean']) * 1e3

        return outputs

    def get_weighted_DELs(self, dlc_generator, inputs, discrete_inputs, outputs):
        modopt = self.options['modeling_options']
        fatigue_dlc_operation = ['1.1', '3.1', '4.1']
        fatigue_dlc_parked = ['6.4']
        fatigue_dlc_fault = ['2.4', '7.2']
        fatigue_dlcs = fatigue_dlc_operation + fatigue_dlc_parked + fatigue_dlc_fault
        
        # See if we have fatigue DLCs
        U = np.zeros(dlc_generator.n_cases)
        ifat = []
        iop = []
        inonop = []
        ifault = []
        for k in range(dlc_generator.n_cases):
            U[k] = dlc_generator.cases[k].URef
            
            if str(dlc_generator.cases[k].label) in fatigue_dlcs:
                ifat.append( k )
                if str(dlc_generator.cases[k].label) in fatigue_dlc_operation:
                    iop.append( k )
                elif str(dlc_generator.cases[k].label) in fatigue_dlc_parked:
                    inonop.append( k )
                elif str(dlc_generator.cases[k].label) in fatigue_dlc_fault:
                    ifault.append( k )

        # If fatigue DLCs are present, then limit analysis to those only
        if len(ifat) > 0:
            U = U[ifat]
        
        # Get wind distribution probabilities, make sure they are normalized
        self.cruncher.set_probability_turbine_class(U, discrete_inputs['turbine_class'], idx=ifat)

        if 'user_probability' in self.options['modeling_options']['DLC_driver']['metocean_conditions']:
            speed = modopt['DLC_driver']['metocean_conditions']['user_probability']['speed']
            user_probability = modopt['DLC_driver']['metocean_conditions']['user_probability']['probability']

            self.cruncher.set_probability_wind_distribution(U, 10., idx=iop, kind='user', v_prob=speed, probability=user_probability)

        # Scale all DELs and damage by probability and collapse over the various DLCs (inner dot product)
        dels_total, damage_total = self.cruncher.compute_total_fatigue(lifetime=inputs['lifetime'],
                                                                       idx=iop, idx_park=inonop, idx_fault=ifault, n_fault=10)
        dels_total = dels_total.loc['Weighted']
        damage_total = damage_total.loc['Weighted']
        
        # Standard DELs for blade root and tower base
        outputs['DEL_RootMyb'] = np.max([dels_total[f'RootMyb{k+1}'] for k in range(self.n_blades)])
        outputs['DEL_TwrBsMyt'] = dels_total['TwrBsM']
        outputs['DEL_TwrBsMyt_ratio'] = dels_total['TwrBsM']/self.options['opt_options']['constraints']['control']['DEL_TwrBsMyt']['max']
            
        # Compute total fatigue damage in spar caps at blade root and trailing edge at max chord location
        if not modopt['OpenFAST']['from_openfast']:
            for k in range(1,self.n_blades+1):
                for u in ['U','L']:
                    damage_total[f'BladeRootSpar{u}_Axial{k}'] = (damage_total[f'RootSpar{u}_Fzb{k}'] +
                                                                  damage_total[f'RootSpar{u}_Mxb{k}'] +
                                                                  damage_total[f'RootSpar{u}_Myb{k}'])
                    damage_total[f'BladeMaxcTE{u}_Axial{k}'] = (damage_total[f'Spn2te{u}_FLzb{k}'] +
                                                                damage_total[f'Spn2te{u}_MLxb{k}'] +
                                                                damage_total[f'Spn2te{u}_MLyb{k}'])

            # Compute total fatigue damage in low speed shaft, tower base, monopile base
            # (Really these channel components should be added to togther to compute stress before rainflow counting,
            # but haven't figured out how to do that via 'calculate_channel' in streaming mode in pcrunch)
            damage_total['LSSAxial'] = damage_total['LSShftAxFxa'] + damage_total['LSShftAxMyza']
            damage_total['LSSShear'] = damage_total['LSShftAxFyza'] + damage_total['LSShftAxMxa']
            damage_total['TowerBaseAxial'] = damage_total['TwrBsAxFzt'] + damage_total['TwrBsAxMxyt']
            damage_total['TowerBaseShear'] = damage_total['TwrBsAxFxyt'] + damage_total['TwrBsAxMzt']
            if modopt['flags']['monopile']:
                damage_total['MonopileBaseAxial'] = damage_total['M1N1AxFKze'] + damage_total['M1N1AxMKxye']
                damage_total['MonopileBaseShear'] = damage_total['M1N1AxFKxye'] + damage_total['M1N1AxMKze']
            else:
                damage_total['MonopileBaseAxial'] = damage_total['MonopileBaseShear'] = 0.0

            # Assemble damages
            outputs['damage_blade_root_sparU'] = np.max([damage_total[f'BladeRootSparU_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_root_sparL'] = np.max([damage_total[f'BladeRootSparL_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_maxc_teU'] = np.max([damage_total[f'BladeMaxcTEU_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_blade_maxc_teL'] = np.max([damage_total[f'BladeMaxcTEL_Axial{k+1}'] for k in range(self.n_blades)])
            outputs['damage_lss'] = np.max( [damage_total['LSSAxial'], damage_total['LSSShear']] )
            outputs['damage_tower_base'] = np.max( [damage_total['TowerBaseAxial'], damage_total['TowerBaseShear']] )
            outputs['damage_monopile_base'] = np.max( [damage_total['MonopileBaseAxial'], damage_total['MonopileBaseShear']] )

            # Log damages
            if self.options['opt_options']['constraints']['damage']['tower_base']['log']:
                outputs['damage_tower_base'] = np.log(outputs['damage_tower_base'])

        return outputs

    def get_control_measures(self, inputs, outputs):
        '''
        calculate control measures:
            - rotor_overspeed

        given:
            - sum_stats : pd.DataFrame
        '''
        nblades = self.fst_vt['ElastoDyn']['NumBl']
        chanmax = [f'dBldPitch{k+1}' for k in range(nblades)]
        chanmax += ['GenSpeed','NcIMUTA']   # Note: this order needs to be maintained for the indexing below to work
        maxes   = self.cruncher.get_load_rankings(chanmax, ['abs'])
        
        # rotor overspeed
        max_gen_speed = maxes['val'].loc[maxes['channel'] == 'GenSpeed'].values[0]
        outputs['rotor_overspeed'] = (max_gen_speed * np.pi/30. / self.fst_vt['DISCON_in']['PC_RefSpd'] ) - 1.0     # Convert to rad/s (like DISCON) and normalize

        # nacelle accelleration
        outputs['max_nac_accel'] = maxes['val'].loc[maxes['channel'] == 'NcIMUTA'].values[0]

        # Pitch rates
        max_pitch_rates = maxes['val'].to_numpy()[:nblades]
        outputs['max_pitch_rate_sim'] = max_pitch_rates.max() / np.rad2deg(self.fst_vt['DISCON_in']['PC_MaxRat'])        # normalize by ROSCO pitch rate
        
        # pitch travel and duty cycle
        if self.options['modeling_options']['General']['openfast_configuration']['keep_time']:
            tot_time = 0
            tot_travel = 0
            num_dir_changes = 0
            max_pitch_rates = [0,0,0]
            for i_ts in range(self.cruncher.noutputs):
                iout = self.cruncher.outputs[i_ts].copy()
                iout.trim_data(self.TStart[i_ts], self.TMax[i_ts])
                
                # total time
                tot_time += iout.elapsed_time
                
                for i_blade in range(nblades):
                    # total pitch travel (\int |\dot{\frac{d\theta}{dt}| dt)
                    tot_travel += iout.total_travel(f'BldPitch{i_blade+1}')

                    # number of direction changes on each blade
                    num_dir_changes += 0.5 * np.sum(np.abs(np.diff(np.sign(iout[f'dBldPitch{i_blade+1}']))))

            # Normalize by number of blades, total time
            outputs['avg_pitch_travel'] = tot_travel / nblades / tot_time
            outputs['pitch_duty_cycle'] = num_dir_changes / nblades / tot_time

        else:
            logger.warning('openmdao_openfast warning: avg_pitch_travel, and pitch_duty_cycle require keep_time = True')

        return outputs

    def get_floating_measures(self, inputs, outputs):
        '''
        calculate floating measures:
            - Std_PtfmPitch (max over all dlcs if constraint, mean otheriwse)
            - Max_PtfmPitch

        given:
            - sum_stats : pd.DataFrame
        '''
        sum_stats = self.cruncher.summary_stats

        if self.options['opt_options']['constraints']['control']['Std_PtfmPitch']['flag']:
            outputs['Std_PtfmPitch'] = np.max(sum_stats['PtfmPitch']['std'])
        else:
            # Let's just average the standard deviation of PtfmPitch for now
            # TODO: weight based on WS distribution, or something else
            outputs['Std_PtfmPitch'] = np.mean(sum_stats['PtfmPitch']['std'])

        outputs['Max_PtfmPitch']  = np.max(np.abs(np.r_[sum_stats['PtfmPitch']['max'],sum_stats['PtfmPitch']['min']]))

        # Max platform offset        
        outputs['Max_Offset'] = np.max(sum_stats['PtfmOffset']['max'])

        return outputs

    def get_OL2CL_error(self, outputs):
        ol_case_names = [os.path.join(
            weis_dir,
            self.options['modeling_options']['OL2CL']['trajectory_dir'],
            case_name + '.p'
        ) for case_name in case_naming(self.options['modeling_options']['DLC_driver']['n_cases'],'oloc')]

        rms_pitch_error = np.full(self.cruncher.noutputs, fill_value=1000.)
        for i_ts in range(self.cruncher.noutputs):
            # Get closed loop timeseries
            cl_ts = self.cruncher.outputs[i_ts]

            # Get open loop timeseries
            ol_ts = pd.read_pickle(ol_case_names[i_ts])

            # resample OL timeseries to match closed loop timeseries
            ol_resample = np.interp(cl_ts['Time'], ol_ts['Time'], ol_ts['BldPitch1'])

            # difference between open loop and closed loop (deg.)
            pitch_error = cl_ts['BldPitch1'] - ol_resample

            rms_pitch_error[i_ts] = np.sqrt(np.mean(pitch_error**2))

            if self.options['modeling_options']['OL2CL']['save_error']:
                save_dir = os.path.join(self.FAST_runDirectory,'iteration_'+str(self.of_inumber),'timeseries')
                pitch_error.to_pickle(os.path.join(save_dir,'pitch_error_'+ str(i_ts) + '.p'))

        # Average over DLCs and return, TODO: weight in future?  only works for a few wind speeds currently
        outputs['OL2CL_pitch'] = np.mean(rms_pitch_error)
        return outputs


    def get_ac_axis(self, inputs):
        
        # Get the absolute offset between pitch axis (rotation center) and aerodynamic center
        ch_offset = inputs['chord'] * (inputs['ac'] - inputs['le_location'])
        # Rotate it by the twist using the AD15 coordinate system
        x , y = util.rotate(0., 0., 0., ch_offset, -np.deg2rad(inputs['theta']))
        # Apply offset to determine the AC axis
        BlCrvAC = inputs['ref_axis_blade'][:,0] + x
        BlSwpAC = inputs['ref_axis_blade'][:,1] + y
        
        return BlCrvAC, BlSwpAC
    

    def write_FAST(self, fst_vt):
        writer                   = InputWriter_OpenFAST()
        writer.fst_vt            = fst_vt
        writer.FAST_runDirectory = self.FAST_runDirectory
        writer.FAST_namingOut    = self.FAST_namingOut
        writer.execute()

    def writeCpsurfaces(self, inputs):

        modopt = self.options['modeling_options']
        FASTpref  = modopt['openfast']['FASTpref']
        file_name = os.path.join(FASTpref['file_management']['FAST_runDirectory'], FASTpref['file_management']['FAST_namingOut'] + '_Cp_Ct_Cq.dat')

        # Write Cp-Ct-Cq-TSR tables file
        n_pitch = len(inputs['pitch_vector'])
        n_tsr   = len(inputs['tsr_vector'])
        n_U     = len(inputs['U_vector'])

        file = open(file_name,'w')
        file.write('# ------- Rotor performance tables ------- \n')
        file.write('# ------------ Written using AeroElasticSE with data from CCBlade ------------\n')
        file.write('\n')
        file.write('# Pitch angle vector - x axis (matrix columns) (deg)\n')
        for i in range(n_pitch):
            file.write('%.2f   ' % inputs['pitch_vector'][i])
        file.write('\n# TSR vector - y axis (matrix rows) (-)\n')
        for i in range(n_tsr):
            file.write('%.2f   ' % inputs['tsr_vector'][i])
        file.write('\n# Wind speed vector - z axis (m/s)\n')
        for i in range(n_U):
            file.write('%.2f   ' % inputs['U_vector'][i])
        file.write('\n')

        file.write('\n# Power coefficient\n\n')

        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Cp_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')

        file.write('\n#  Thrust coefficient\n\n')
        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Ct_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')

        file.write('\n# Torque coefficient\n\n')
        for i in range(n_U):
            for j in range(n_tsr):
                for k in range(n_pitch):
                    file.write('%.5f   ' % inputs['Cq_aero_table'][j,k,i])
                file.write('\n')
            file.write('\n')

        file.close()


        return file_name

    def save_timeseries(self):
        '''
        Save ALL the timeseries: each iteration and openfast run thereof
        '''

        # Make iteration directory
        save_dir = os.path.join(self.FAST_runDirectory,'iteration_'+str(self.of_inumber),'timeseries')
        os.makedirs(save_dir, exist_ok=True)

        # Save each timeseries as a pickled dataframe
        for i_ts in range(self.cruncher.noutputs):
            self.cruncher.outputs[i_ts].save( os.path.join(save_dir,f'{self.FAST_namingOut}_{i_ts}.p'))

    def save_iterations(self, discrete_outputs):
        '''
        Save summary stats, DELs of each iteration
        '''
        sum_stats = self.cruncher.summary_stats
        DELs = self.cruncher.dels
        
        # Make iteration directory
        save_dir = os.path.join(self.FAST_runDirectory,'iteration_'+str(self.of_inumber))
        os.makedirs(save_dir, exist_ok=True)

        # Save dataframes as pickles
        sum_stats.to_pickle(os.path.join(save_dir,'summary_stats.p'))
        DELs.to_pickle(os.path.join(save_dir,'DELs.p'))

        # Save fst_vt as pickle
        with open(os.path.join(save_dir,'fst_vt.p'), 'wb') as f:
            pickle.dump(self.fst_vt,f)

        discrete_outputs['ts_out_dir'] = save_dir

def apply_olaf_parameters(dlc_generator,fst_vt):
    '''
    Apply OLAF parameters using wind speed, rotor speed, and rotor radius
    Parameters are applied for each case, if Wake_Mod = 3

    This method requires that the case inputs have been generated for each case in dlc_generator

    OLAF parameters will be appended to proper openfast_case_input for each dlc
    '''

    # Loop over cases
    for case_input in dlc_generator.openfast_case_inputs:

        min_TMax = min(case_input[('Fst', 'TMax')]['vals'])

        # find group with wind_speed
        wind_group = case_input[('InflowWind', 'HWindSpeed')]['group']

        wind_speeds = case_input[('InflowWind', 'HWindSpeed')]['vals']
        rotor_speeds = case_input[('ElastoDyn', 'RotSpeed')]['vals']

        dt_fvw = len(wind_speeds) * [None]
        tMin = len(wind_speeds) * [None]
        nNWPanels = len(wind_speeds) * [None]
        nNWPanelsFree = len(wind_speeds) * [None]
        nFWPanels = len(wind_speeds) * [None]
        nFWPanelsFree = len(wind_speeds) * [None]

        for i_case in range(len(wind_speeds)):
            dt_fvw[i_case], tMin[i_case], nNWPanels[i_case], nNWPanelsFree[i_case], nFWPanels[i_case], nFWPanelsFree[i_case] = OLAFParams(rotor_speeds[i_case], wind_speeds[i_case], fst_vt['ElastoDyn']['TipRad'])

            # Check that runs are long enough
            if tMin[i_case] > min_TMax:
                logger.warning("OLAF runs are too short in time, the wake is not at convergence")
            
            case_input[("AeroDyn","OLAF","DTfvw")] = {'vals': dt_fvw, 'group': wind_group}
            case_input[("AeroDyn","OLAF","nNWPanels")] = {'vals': nNWPanels, 'group': wind_group}
            case_input[("AeroDyn","OLAF","nNWPanelsFree")] = {'vals': nNWPanelsFree, 'group': wind_group}
            case_input[("AeroDyn","OLAF","nFWPanels")] = {'vals': nFWPanels, 'group': wind_group}
            case_input[("AeroDyn","OLAF","nFWPanelsFree")] = {'vals': nFWPanelsFree, 'group': wind_group}