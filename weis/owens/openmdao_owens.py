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
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
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


class OWENS_WEIS(om.Group):
    # TODO: This was intended to do common setup

    def intialize(self):
        self.options.declare("modeling_options")
        self.options.declare("analysis_options")

    def setup(self):
        modeling_opt = self.options["modeling_options"]
        analysis_opt = self.options["analysis_options"]

        level = modeling_opt['Level']


class OWENSStructSetup(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modopt = self.options['modeling_options']
        OWENS_directory = modopt["OWENS_directory"]

        jlPkg.activate(OWENS_directory)
        jl.seval("using OWENS")
        jl.seval("using OWENSAero")

        master_file = modopt["master_input"]

        # Initialize an OWENS model
        self.model = jl.OWENS.MasterInput(master_file)

        self.analysis_type = modopt["analysis_type"]
        self.turbine_type = modopt["turbine_type"]
        self.control_strategy = modopt["control_strategy"]
        self.aeroModel = modopt["aeroModel"]
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
        self.add_input("Nbld", val=2, desc="number of blades")
        self.add_input("Blade_Radius", val=17.1)
        self.add_input("Blade_Height", val=41.9)
        self.add_input("towerHeight", val=0.5)


        # Blade inputs for composites
        # self.add_input('structuralModel', val='GX', desc="Structural models, GX, TNB, or ROM")
        # self.add_input('nonlinear', val=False, desc="[]")
        self.add_input("ntelem", val=10, desc="Tower elements in each")
        self.add_input("nbelem", val=60, desc="Blade elements in each")
        self.add_input("ncelem", val=10, desc="Central cable elements in each if turbine type is ARCUS")
        self.add_input("nselem", val=5, desc="Blade elements in each")


        # Solver options (come from modeling options)
        
        # Control inputs
        self.add_input("RPM", val=34.0, desc="RPM")
        self.add_input("numTS", val=6)
        self.add_input("delta_t", val=0.05, units="s")

        # Aero parameters
        self.add_input("NSlices", val=35, desc="number of VAWTAero discritizations #TODO: AD parameters")
        self.add_input("ntheta", val=30, desc="number of VAWTAero azimuthal discretizations")

        # Environmental conditions
        # Maybe this fits better into load cases?

        # operation parameters
        self.add_input("rho", val=0.94, units="kg/m**3", desc="Fluid dendity")
        self.add_input("Vinf", val=10.1, units="m/s", desc="Inflow velocity")

        # external force
        self.add_input("Fexternal", val=np.array([1000.0, 0.0, 1000.0]), units="N", desc="External forces")


        self.add_output("blade_mass", units="kg", val=0.0, desc="Blade mass breakdown")
        self.add_output("displacement", units="m", val=0.0)

        # discretization
        # Maybe these controlpts are grid?
        # n_control_pts = len()
        self.add_input("controlpts", val=np.array([3.6479257474344826, 6.226656883619295, 9.082267631309085, 11.449336766507562, 13.310226748873827, 14.781369210504563, 15.8101544043681, 16.566733104331984, 17.011239869982738, 17.167841319391137, 17.04306679619916, 16.631562597633675, 15.923729603782338, 14.932185789551408, 13.62712239754136, 12.075292152969496, 10.252043906945818, 8.124505683235517, 5.678738418596312, 2.8959968657512207]), units=None)

    def initialize_model(self):
        self.model.analysisType = self.analysis_type
        self.model.turbineType = self.turbine_type
        self.model.controlStrategy = self.control_strategy
        self.model.AModel = self.aeroModel
        self.model.adi_lib = self.adi_lib
        self.model.adi_rootname = self.adi_rootname
        self.model.Nbld = int(self.n_blades)

        # This live in initialize model
        # TODO: If at any point we want to change the material parameters, we need to take this out to compute
        self.model.NuMad_geom_xlscsv_file_twr = self.NumadSpec["NuMad_geom_xlscsv_file_twr"]
        self.model.NuMad_mat_xlscsv_file_twr = self.NumadSpec["NuMad_mat_xlscsv_file_twr"]
        self.model.NuMad_geom_xlscsv_file_bld = self.NumadSpec["NuMad_geom_xlscsv_file_bld"]
        self.model.NuMad_mat_xlscsv_file_bld = self.NumadSpec["NuMad_mat_xlscsv_file_bld"]
        self.model.NuMad_geom_xlscsv_file_strut = self.NumadSpec["NuMad_geom_xlscsv_file_strut"]
        self.model.NuMad_mat_xlscsv_file_strut = self.NumadSpec["NuMad_mat_xlscsv_file_strut"]

        # Initialize turbulenece
        self.model.ifw = self.Turbulence["ifw"]
        self.model.WindType = self.Turbulence["WindType"]
        self.model.windINPfilename = self.Turbulence["windINPfilename"]
        self.model.ifw_libfile = self.Turbulence["ifw_libfile"]

        # initialize structural model
        self.model.structuralModel = self.structuralModel
        # self.model.nonlinear = self.structuralNonlinear

    def setup_partials(self):
        self.declare_partials("blade_mass", ["Blade_Radius", "Blade_Height"], method="fd")

    def compute(self, inputs, outputs):
        path = self.run_path
        rho = inputs["rho"][0]
        RPM = inputs["RPM"][0]
        Vinf = inputs["Vinf"][0]
        eta = inputs["eta"][0]
        B = int(inputs["Nbld"][0])
        H = inputs["Blade_Height"][0]
        R = inputs["Blade_Radius"][0]
        Ht = inputs["towerHeight"][0]

        # z_shape1 = collect(LinRange(0,41.9,length(controlpts)+2))
        # x_shape1 = [0.0;controlpts;0.0]
        # z_shape = collect(LinRange(0,41.9,60))
        # x_shape = FLOWMath.akima(z_shape1,x_shape1,z_shape)#[0.0,1.7760245854312287, 5.597183088188207, 8.807794161662574, 11.329376903432605, 13.359580331518579, 14.833606099357858, 15.945156349709, 16.679839160110422, 17.06449826588358, 17.10416552269884, 16.760632435904647, 16.05982913536134, 15.02659565585254, 13.660910465851046, 11.913532434360155, 9.832615229216344, 7.421713825584581, 4.447602800040282, 0.0]
        # toweroffset = 4.3953443986241725
        # SNL34_unit_xz = [x_shape;;z_shape]
        # SNL34x = SNL34_unit_xz[:,1]./maximum(SNL34_unit_xz[:,1])
        # SNL34z = SNL34_unit_xz[:,2]./maximum(SNL34_unit_xz[:,2])
        # SNL34Z = SNL34z.*Blade_Height #windio
        # SNL34X = SNL34x.*Blade_Radius #windio
        controlpts = inputs["controlpts"]
        # These are initial grid
        z_shape1 = np.linspace(0, 41.9, len(controlpts)+2)
        x_shape1 = np.insert(controlpts,0,0)
        x_shape1 = np.insert(x_shape1,-1,0)
        # These are dicretization
        z_shape = np.linspace(0, 41.9, 60)
        x_shape = Akima1DInterpolator(z_shape1, x_shape1)(z_shape)
        SNL34_unit_xz = np.vstack((x_shape, z_shape))
        SNL34x = SNL34_unit_xz[0,:]/np.max(SNL34_unit_xz[0,:]) # indexing different
        SNL34z = SNL34_unit_xz[1,:]/np.max(SNL34_unit_xz[1,:])
        SNL34Z = SNL34z*H #windio
        SNL34X = SNL34x*R #windio

        shapeY = SNL34Z#collect(LinRange(0,H,Nslices+1))
        shapeX = SNL34X#R.*(1.0.-4.0.*(shapeY/H.-.5).^2)#shapeX_spline(shapeY)

        ifw = self.Turbulence["ifw"]
        delta_t = inputs["delta_t"][0]
        adi_lib = self.adi_lib
        adi_rootname = self.adi_rootname
        windINPfilename = self.Turbulence["windINPfilename"]
        ifw_libfile = self.Turbulence["ifw_libfile"]

        NuMad_geom_xlscsv_file_twr = self.NumadSpec["NuMad_geom_xlscsv_file_twr"]
        NuMad_mat_xlscsv_file_twr = self.NumadSpec["NuMad_mat_xlscsv_file_twr"]
        NuMad_geom_xlscsv_file_bld = self.NumadSpec["NuMad_geom_xlscsv_file_bld"]
        NuMad_mat_xlscsv_file_bld = self.NumadSpec["NuMad_mat_xlscsv_file_bld"]
        NuMad_geom_xlscsv_file_strut = self.NumadSpec["NuMad_geom_xlscsv_file_strut"]
        NuMad_mat_xlscsv_file_strut = self.NumadSpec["NuMad_mat_xlscsv_file_strut"]

        numTS = int(inputs["numTS"][0])

        Nslices = int(inputs["NSlices"][0])
        ntheta = int(inputs["ntheta"][0])

        ntelem = int(inputs["ntelem"][0])
        nbelem = int(inputs["nbelem"][0])
        ncelem = int(inputs["ncelem"][0])
        nselem = int(inputs["nselem"][0])

        AModel = self.aeroModel
        mesh_type = self.turbine_type
        setup_outputs = jl.OWENS.setupOWENS(jl.OWENSAero, path, 
                                            rho=rho,
                                            Nslices=Nslices,
                                            ntheta=ntheta,
                                            RPM=RPM,
                                            Vinf=Vinf,
                                            eta=eta,
                                            B = B,
                                            H = H,
                                            R = R,
                                            shapeY=shapeY,
                                            shapeX=shapeX,
                                            ifw=ifw,
                                            delta_t=delta_t,
                                            numTS=numTS,
                                            adi_lib=adi_lib,
                                            adi_rootname=adi_rootname,
                                            AD15hubR = 0,
                                            windINPfilename=windINPfilename,
                                            ifw_libfile=ifw_libfile,
                                            NuMad_geom_xlscsv_file_twr=NuMad_geom_xlscsv_file_twr,
                                            NuMad_mat_xlscsv_file_twr=NuMad_mat_xlscsv_file_twr,
                                            NuMad_geom_xlscsv_file_bld=NuMad_geom_xlscsv_file_bld,
                                            NuMad_mat_xlscsv_file_bld=NuMad_mat_xlscsv_file_bld,
                                            NuMad_geom_xlscsv_file_strut=NuMad_geom_xlscsv_file_strut,
                                            NuMad_mat_xlscsv_file_strut=NuMad_mat_xlscsv_file_strut,
                                            Ht=Ht,
                                            ntelem=ntelem,
                                            nbelem=nbelem,
                                            ncelem=ncelem,
                                            nselem=nselem,
                                            joint_type = 0,
                                            strut_mountpointbot = 0.03,
                                            strut_mountpointtop = 0.03,
                                            AModel=AModel, #AD, DMS, AC
                                            DSModel="BV",
                                            RPI=True,
                                            cables_connected_to_blade_base = True,
                                            angularOffset = np.pi/2,
                                            meshtype = mesh_type
                                            )
        
        top_idx = 23
        pBC = np.array([[1, 1, 0],
        [1, 2, 0],
        [1, 3, 0],
        [1, 4, 0],
        [1, 5, 0],
        [1, 6, 0],
        [top_idx, 1, 0],
        [top_idx, 2, 0],
        [top_idx, 3, 0],
        [top_idx, 4, 0],
        [top_idx, 5, 0]])

        mymesh = setup_outputs[0]
        
        feamodel = jl.OWENS.FEAModel(analysisType = self.structuralModel,
                                joint = setup_outputs[3],
                                platformTurbineConnectionNodeNumber = 1,
                                pBC=pBC,
                                nlOn=False,
                                numNodes = mymesh.numNodes,
                                numModes = 200,
                                RayleighAlpha = 0.05,
                                RayleighBeta = 0.05,
                                iterationType = "DI")
        
        feamodel.analysisType = "S"
        Fdof = np.array([15,16,17])
        Fexternal = inputs["Fexternal"]

        Omega = RPM*np.pi/30

        myel = setup_outputs[1]

        displ = np.zeros(mymesh.numNodes*6)
        elStorage = jl.OWENS.OWENSFEA.initialElementCalculations(feamodel,myel,mymesh)
        staticAnalysis_outputs = jl.OWENS.OWENSFEA.staticAnalysis(feamodel,mymesh,myel,displ,Omega,0.0,elStorage,reactionNodeNumber=1,Fdof=Fdof,Fexternal=Fexternal)
        
        blade_mass = setup_outputs[6]
        displ = staticAnalysis_outputs[0][-1]

        outputs["blade_mass"] = np.sum(blade_mass)[0,0]
        outputs["displacement"] = displ

        
class OWENSUnsteadySetup(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        # self.options.declare("opt_options")
        # TODO: we can add an option here to output the intermediate yaml file, like
        # self.options.declare("owens_yaml")

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
        self.aeroModel = modopt["aeroModel"]
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
        # Note: Change the return line in topRunDLC in OWENS to "return mass_breakout_twr, genPower, massOwens" for this to work
        self.add_output("tower_mass", units="kg", val=np.ones(7), desc="Tower mass breakdown")
        self.add_output("first_tower_mass", units="kg", val=1.0)
        self.add_output("GenPower", units="W", val=0.0)
        self.add_output("mass", units="kg", val=np.ones(7), desc="Mass")

    def initialize_model(self):
        # TODO: depending on the owens_yaml option, we can either update the model options directly, or write the intermediate yaml
        # and then update the model options
        # For example
        # use_yaml = self.options("owens_yaml")
        # If use_yaml:
        #     write updated yaml input given the input values
        #     call self.model = jl.OWENS.MasterInput(new_master_file) to initialize the model with the updated yaml input file
        # else: change the inputs directly as follows
        self.model.analysisType = self.analysis_type
        self.model.turbineType = self.turbine_type
        self.model.controlStrategy = self.control_strategy
        self.model.AModel = self.aeroModel
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

    # def update_model_with_yaml(self, inputs):
    # TODO: this is for updating the model using the yaml files, will be called in compute
    # First take the inputs and write out the yaml file
    # a = inputs["a"]
    # write updated yaml input given the input values
    # call self.model = jl.OWENS.MasterInput(new_master_file) to initialize the model with the updated yaml input file

    def setup_partials(self):
        # This can be set analytically from julia AD
        self.declare_partials("first_tower_mass", "towerHeight", method="fd")
        self.declare_partials("GenPower", "*", method="fd")
        self.declare_partials("mass", "*", method="fd")

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

        # TODO: depending on the owens_yaml option, we can either update the model options directly, or write the intermediate yaml
        # and then update the model inputs
        # if use_yaml:
        #   self.update_model_with_yaml(inputs):
        # else update the model directly as follows:

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
        # Note: Change the return line in topRunDLC in OWENS to "return mass_breakout_twr, genPower, massOwens"
        outputs["tower_mass"] = results[0]
        outputs["first_tower_mass"] = results[0][0]
        outputs["GenPower"] = np.mean(results[1])
        outputs["mass"] = results[2]








            
            