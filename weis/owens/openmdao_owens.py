import numpy as np
import pandas as pd
# import os
# import shutil
# import sys
# import copy
# import glob
# import logging
# import pickle
# TODO: remove unnecessary imports when finalizing everything
# from pathlib import Path
# from scipy.interpolate                      import PchipInterpolator
import openmdao.api as om
from openmdao.api                           import ExplicitComponent
# from wisdem.commonse.mpi_tools              import MPI
# from wisdem.commonse import NFREQ
# from wisdem.commonse.cylinder_member import get_nfull
# import wisdem.commonse.utilities              as util
# from wisdem.rotorse.rotor_power             import eval_unsteady
# from weis.aeroelasticse.FAST_writer         import InputWriter_OpenFAST
# from weis.aeroelasticse.FAST_reader         import InputReader_OpenFAST
# import weis.aeroelasticse.runFAST_pywrapper as fastwrap
# from weis.aeroelasticse.FAST_post         import FAST_IO_timeseries
# from wisdem.floatingse.floating_frame import NULL, NNODES_MAX, NELEM_MAX
# from weis.dlc_driver.dlc_generator    import DLCGenerator
# from weis.aeroelasticse.CaseGen_General import CaseGen_General
# from functools import partial
# from pCrunch import PowerProduction
# from weis.aeroelasticse.LinearFAST import LinearFAST
# from weis.control.LinearModel import LinearTurbineModel, LinearControlModel
# from weis.aeroelasticse import FileTools
# from weis.aeroelasticse.turbsim_file   import TurbSimFile
# from weis.aeroelasticse.turbsim_util import generate_wind_files
# from weis.aeroelasticse.utils import OLAFParams
# from rosco.toolbox import control_interface as ROSCO_ci
# from pCrunch.io import OpenFASTOutput
# from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
# from weis.control.dtqp_wrapper          import dtqp_wrapper
# from weis.aeroelasticse.StC_defaults        import default_StC_vt
# from weis.aeroelasticse.CaseGen_General import case_naming
# from wisdem.inputs import load_yaml
# Juliacall for OWENS
from juliacall import Main as jl
from juliacall import Pkg as jlPkg


# if MPI:
#     from mpi4py   import MPI


class OWENSSetup(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        # self.options.declare("opt_options")

    def setup(self):
        modopt = self.options['modeling_options']
        OWENS_directory = modopt["OWENS_directory"]


        jlPkg.activate(OWENS_directory)
        jl.seval("using OWENS")

        master_file = modopt["master_input"]

        # Initialize an OWENS model
        self.model = jl.OWENS.MasterInput(master_file)


        # analysisType: unsteady # unsteady, steady, modal
        # turbineType: Darrieus #Darrieus, H-VAWT, ARCUS   
        # These should be in modeling options?   
        # Currently, they are just normal openmdao component options,
        # Not considering any WEIS data/input structure
        # Probably needs to be like modopt["crossflow"]["analysis_type"]
        self.analysis_type = modopt["analysis_type"]
        self.turbine_type = modopt["turbine_type"]
        self.control_strategy = modopt["control_strategy"]
        self.aeroMOodel = modopt["aeroModel"]
        self.adi_lib = modopt["adi_lib"]
        self.adi_rootname = modopt["adi_rootname"]
        self.NumadSpec = modopt["NumadSpec"] # All files go here
        self.Turbulence = modopt["TurbulenceSpec"] # All files go here
        self.n_blades = modopt["number_of_blades"]
        self.structuralModel = modopt["structuralModel"]
        self.structuralNonlinear = modopt["structuralNonlinear"]
        self.run_path = modopt["run_path"] # What exactly is this path?

        # Reinitialize the model with the inputs from modeling options
        self.initialize_model()

        # Blade inputs, geometry and discretization
        self.add_input("eta", val=0.5, desc="blade mount point ratio, 0.5 is the blade half chord is perpendicular with the axis of rotation, 0.25 is the quarter chord, etc")
        self.add_input("Nbld", val=3, desc="number of blades")
        self.add_input("Blade_Radius", val=54.01123056)
        self.add_input("Blade_Height", val=110.1829092)
        self.add_input("towerHeight", val=3.0)


        # Blade inputs for composites
        # self.add_input('structuralModel', val='GX', desc="Structural models, GX, TNB, or ROM")
        # self.add_input('nonlinear', val=False, desc="[]")
        self.add_input("ntelem", val=10.0, desc="Tower elements in each")
        self.add_input("nbelem", val=60.0, desc="Blade elements in each")
        self.add_input("ncelem", val=10.0, desc="Central cable elements in each if turbine type is ARCUS")
        self.add_input("nselem", val=60.0, desc="Blade elements in each")


        # Solver options (come from modeling options)
        
        # Control inputs
        if self.control_strategy == "constantRPM":
            self.add_input("RPM", val=17.2, desc="RPM")
            self.add_input("numTS", val=100)
            self.add_input("delta_t", val=0.01, units="s")

        # Aero parameters
        self.add_input("NSlices", val=30, desc="number of VAWTAero discritizations #TODO: AD parameters")
        self.add_input("ntheta", val=30, desc="number of VAWTAero azimuthal discretizations")

        # Environmental conditions
        # Maybe this fits better into load cases?

        # operation parameters
        self.add_input("rho", val=1.225, units="kg", desc="Fluid dendity")
        self.add_input("Vinf", val=17.2, units="m/s", desc="Inflow velocity")


        # Ignore openfast parts for now
        # TODO: work on openfast parts
        # Floating platform inputs
        # self.add_input("transition_node", np.zeros(3), units="m")
        # self.add_input("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        # self.add_input("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        # self.add_input("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        # self.add_input("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        # self.add_input("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        # self.add_input("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        # self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        # self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        # self.add_discrete_input("platform_elem_memid", [0]*NELEM_MAX)
        # self.add_input("platform_total_center_of_mass", np.zeros(3), units="m")
        # self.add_input("platform_mass", 0.0, units="kg")
        # self.add_input("platform_I_total", np.zeros(6), units="kg*m**2")

        # if modopt['flags']["floating"]:
        #     n_member = modopt["floating"]["members"]["n_members"]
        #     for k in range(n_member):
        #         n_height_mem = modopt["floating"]["members"]["n_height"][k]
        #         self.add_input(f"member{k}:joint1", np.zeros(3), units="m")
        #         self.add_input(f"member{k}:joint2", np.zeros(3), units="m")
        #         self.add_input(f"member{k}:s", np.zeros(n_height_mem))
        #         self.add_input(f"member{k}:s_ghost1", 0.0)
        #         self.add_input(f"member{k}:s_ghost2", 0.0)
        #         self.add_input(f"member{k}:outer_diameter", np.zeros(n_height_mem), units="m")
        #         self.add_input(f"member{k}:wall_thickness", np.zeros(n_height_mem-1), units="m")


        # # Moordyn inputs
        # mooropt = modopt["mooring"]
        # if self.options["modeling_options"]["flags"]["mooring"]:
        #     n_nodes = mooropt["n_nodes"]
        #     n_lines = mooropt["n_lines"]
        #     self.add_input("line_diameter", val=np.zeros(n_lines), units="m")
        #     self.add_input("line_mass_density", val=np.zeros(n_lines), units="kg/m")
        #     self.add_input("line_stiffness", val=np.zeros(n_lines), units="N")
        #     self.add_input("line_transverse_added_mass", val=np.zeros(n_lines), units="kg/m")
        #     self.add_input("line_tangential_added_mass", val=np.zeros(n_lines), units="kg/m")
        #     self.add_input("line_transverse_drag", val=np.zeros(n_lines))
        #     self.add_input("line_tangential_drag", val=np.zeros(n_lines))
        #     self.add_input("nodes_location_full", val=np.zeros((n_nodes, 3)), units="m")
        #     self.add_input("nodes_mass", val=np.zeros(n_nodes), units="kg")
        #     self.add_input("nodes_volume", val=np.zeros(n_nodes), units="m**3")
        #     self.add_input("nodes_added_mass", val=np.zeros(n_nodes))
        #     self.add_input("nodes_drag_area", val=np.zeros(n_nodes), units="m**2")
        #     self.add_input("unstretched_length", val=np.zeros(n_lines), units="m")
        #     self.add_discrete_input("node_names", val=[""] * n_nodes)

        # Outputs
        self.add_output("tower_mass", units="kg", val=np.ones(7), desc="Tower mass breakdown")
        self.add_output("first_tower_mass", units="kg", val=1.0)


    def initialize_model(self):
        self.model.analysisType = self.analysis_type
        self.model.turbineType = self.turbine_type
        self.model.controlStrategy = self.control_strategy
        self.model.AModel = self.aeroMOodel
        self.model.adi_lib = self.adi_lib
        self.model.adi_rootname = self.adi_rootname
        self.model.Nbld = int(self.n_blades)

        # This live in initialize model
        # TODO: If at any point we want to change the material parameters, we need to take this out to compute
        self.model.NuMad_geom_xlscsv_file_twr = self.NumadSpec["NuMad_geom_xlscsv_file_twr"]
        self.model.NuMad_mat_xlscsv_file_twr = self.NumadSpec["NuMad_mat_xlscsv_file_twr"]
        self.NuMad_geom_xlscsv_file_bld = self.NumadSpec["NuMad_geom_xlscsv_file_bld"]
        self.NuMad_mat_xlscsv_file_bld = self.NumadSpec["NuMad_mat_xlscsv_file_bld"]
        self.NuMad_geom_xlscsv_file_strut = self.NumadSpec["NuMad_geom_xlscsv_file_strut"]
        self.NuMad_mat_xlscsv_file_strut = self.NumadSpec["NuMad_mat_xlscsv_file_strut"]

        # Initialize turbulenece
        self.model.ifw = self.Turbulence["ifw"]
        self.model.WindType = self.Turbulence["WindType"]
        self.model.windINPfilename = self.Turbulence["windINPfilename"]
        self.model.ifw_libfile = self.Turbulence["ifw_libfile"]

        # initialize structural model
        self.model.structuralModel = self.structuralModel
        # self.model.nonlinear = self.structuralNonlinear

    def setup_partials(self):
        # This can be set analytically from julia AD
        self.declare_partials("first_tower_mass", "towerHeight", method="fd")

    def compute(self, inputs, outputs):

        eta = inputs["eta"][0]
        Blade_Radius = inputs["Blade_Radius"][0]
        Blade_Height = inputs["Blade_Height"][0]
        towerHeight = inputs["towerHeight"][0]

        rho = inputs["rho"][0]
        Vinf = inputs["Vinf"][0]

        RPM = inputs["RPM"][0]
        numTS = int(inputs["numTS"][0])
        delta_t = inputs["delta_t"][0]

        Nslices = int(inputs["NSlices"][0])
        ntheta = int(inputs["ntheta"][0])

        ntelem = int(inputs["ntelem"][0])
        nbelem = int(inputs["nbelem"][0])
        ncelem = int(inputs["ncelem"][0])
        nselem = int(inputs["nselem"][0])

        self.model.eta = eta
        self.model.Blade_Radius = Blade_Radius
        self.model.Blade_Height = Blade_Height
        self.model.towerHeight = towerHeight

        self.model.rho = rho
        self.model.Vinf = Vinf
        self.model.RPM = RPM
        self.model.numTS = numTS
        self.model.delta_t = delta_t
        self.model.Nslices = Nslices
        self.model.ntheta = ntheta

        self.model.ntelem = ntelem
        self.model.nbelem = nbelem
        self.model.ncelem = ncelem
        self.model.nselem = nselem

        results = jl.OWENS.runOWENS(self.model,"../../../OWENS.jl/docs/src/literate/", verbosity=0)

        # Unpack outputs
        outputs["tower_mass"] = results
        outputs["first_tower_mass"] = results[0]


if __name__ == "__main__":

    model = om.Group()

    options = {
        "OWENS_directory": "/Users/yliao/Library/CloudStorage/OneDrive-NREL/CT-OPT/OWENS",
        "master_input": "sampleOWENS.yml",
        "analysis_type": "unsteady",
        "turbine_type": "Darrieus",
        "control_strategy": "constantRPM",
        "aeroModel":"DMS",
        "adi_lib":"./../../../openfast/build/modules/aerodyn/libaerodyn_inflow_c_binding",
        "adi_rootname": "./ExampleB",
        "NumadSpec":{
            "NuMad_geom_xlscsv_file_twr": "../../../OWENS.jl/docs/src/literate/data/NuMAD_Geom_SNL_5MW_D_TaperedTower.csv",
            "NuMad_mat_xlscsv_file_twr": "../../../OWENS.jl/docs/src/literate/data/NuMAD_Materials_SNL_5MW.csv",
            "NuMad_geom_xlscsv_file_bld": "../../../OWENS.jl/docs/src/literate/data/NuMAD_Geom_SNL_5MW_D_Carbon_LCDT_ThickFoils_ThinSkin.csv",
            "NuMad_mat_xlscsv_file_bld": "../../../OWENS.jl/docs/src/literate/data/NuMAD_Materials_SNL_5MW.csv",
            "NuMad_geom_xlscsv_file_strut": "../../../OWENS.jl/docs/src/literate/data/NuMAD_Geom_SNL_5MW_Struts.csv",
            "NuMad_mat_xlscsv_file_strut": "../../../OWENS.jl/docs/src/literate/data/NuMAD_Materials_SNL_5MW.csv",

        },
        "TurbulenceSpec":{
            "ifw": False,
            "WindType": 3,
            "windINPfilename": "../../../OWENS.jl/docs/src/literate/data/turbsim/115mx115m_30x30_20.0msETM.bts",
            "ifw_libfile": "./../../../openfast/build/modules/inflowwind/libifw_c_binding"
        },
        "number_of_blades": 3,
        "structuralModel": "GX",
        "structuralNonlinear": False,
        "run_path": "../../../OWENS.jl/docs/src/literate/",
    }
    model.add_subsystem("crossflow", OWENSSetup(modeling_options=options))

    prob = om.Problem(model)
    prob.setup()
    prob.set_val("crossflow.towerHeight", 5.0)
    prob.run_model()
    print("Tower height of 5 m: "+str(prob.get_val("crossflow.tower_mass")))
    prob.set_val("crossflow.towerHeight", 3.0)
    prob.run_model()
    print("Tower height of 3 m: "+str(prob.get_val("crossflow.tower_mass")))

    # run opt
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    prob.model.add_design_var('crossflow.towerHeight', lower=3.0, upper=5.0)
    prob.model.add_objective('crossflow.first_tower_mass')
    prob.setup()
    prob.run_driver()

    print("Optimal height: "+str(prob.get_val("crossflow.towerHeight")))
    print("This should be 3 m")







            
            