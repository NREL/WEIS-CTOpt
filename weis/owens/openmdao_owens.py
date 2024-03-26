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
from wisdem.commonse.mpi_tools              import MPI
from wisdem.commonse import NFREQ
from wisdem.commonse.cylinder_member import get_nfull
import wisdem.commonse.utilities              as util
from wisdem.rotorse.rotor_power             import eval_unsteady
from weis.aeroelasticse.FAST_writer         import InputWriter_OpenFAST
from weis.aeroelasticse.FAST_reader         import InputReader_OpenFAST
import weis.aeroelasticse.runFAST_pywrapper as fastwrap
from weis.aeroelasticse.FAST_post         import FAST_IO_timeseries
from wisdem.floatingse.floating_frame import NULL, NNODES_MAX, NELEM_MAX
from weis.dlc_driver.dlc_generator    import DLCGenerator
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from functools import partial
from pCrunch import PowerProduction
from weis.aeroelasticse.LinearFAST import LinearFAST
from weis.control.LinearModel import LinearTurbineModel, LinearControlModel
from weis.aeroelasticse import FileTools
from weis.aeroelasticse.turbsim_file   import TurbSimFile
from weis.aeroelasticse.turbsim_util import generate_wind_files
from weis.aeroelasticse.utils import OLAFParams
from rosco.toolbox import control_interface as ROSCO_ci
from pCrunch.io import OpenFASTOutput
from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
from weis.control.dtqp_wrapper          import dtqp_wrapper
from weis.aeroelasticse.StC_defaults        import default_StC_vt
from weis.aeroelasticse.CaseGen_General import case_naming
from wisdem.inputs import load_yaml

if MPI:
    from mpi4py   import MPI


class OWENSSetup(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modopt = self.options['modeling_options']
        # analysisType: unsteady # unsteady, steady, modal
        # turbineType: Darrieus #Darrieus, H-VAWT, ARCUS   
        # These should be in modeling options?   
        self.analysis_type = modopt[][]
        self.turbine_type = modopt[][]
        self.control_strategy = modopt[][]
        self.aeroMOodel = modopt[][]
        self.adi_lib = modopt[][]
        self.adi_rootname = modopt[][]
        self.NumadSpec = modopt[][] # All files go here

        # Blade inputs, geometry and discretization
        self.add_input("eta", val=0.5, desc="blade mount point ratio, 0.5 is the blade half chord is perpendicular with the axis of rotation, 0.25 is the quarter chord, etc")
        self.add_input("Nbld", val=3, desc="number of blades")
        self.add_input("Blade_Radius", val=54.01123056)
        self.add_input("Blade_Height", val=110.1829092)
        self.add_input("towerHeight", val=3.0)


        # Blade inputs for composites
        self.add_input('structuralModel', val='GX', desc="Structural models, GX, TNB, or ROM")
        self.add_input('nonlinear', val=False, desc="[]")
        self.add_input("ntelem", val=10, desc="Tower elements in each")
        self.add_input("nbelem", val=60, desc="Blade elements in each")
        self.add_input("ncelem", val=10, desc="Central cable elements in each if turbine type is ARCUS")
        self.add_input("nselem", val=60, desc="Blade elements in each")


        # Solver options (come from modeling options)
        
        # Control inputs
        if self.control_strategy == "constantRPM":
            self.add_input("RPM", val=17.2, desc="RPM")
            self.add_input("numTS", val=100)
            self.add_input("delta_t", val=0.01, units="s")

        # Aero parameters
        self.add_input("NSlices", desc="number of VAWTAero discritizations #TODO: AD parameters")
        self.add_input("ntheta", desc="number of VAWTAero azimuthal discretizations")

        # Environmental conditions
        # Maybe this fits better into load cases?


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
        self.add_discrete_input("platform_elem_memid", [0]*NELEM_MAX)
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


        # Moordyn inputs
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

            