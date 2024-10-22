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
from wisdem.commonse.utilities import arc_length
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
# from OWENS_output_reader import *
from collections import OrderedDict

# if MPI:
#     from mpi4py   import MPI

        
class OWENSUnsteadySetup(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("rotorse_options")
        self.options.declare("towerse_options")
        self.options.declare("strut_options")
        self.options.declare("opt_options")
        # TODO: we can add an option here to output the intermediate yaml file, like
        # self.options.declare("owens_yaml")

    def setup(self):
        modopt = self.options['modeling_options']
        rotorse_options = self.options["rotorse_options"]
        towerse_options = self.options["towerse_options"]
        strut_options = self.options["strut_options"]
        OWENS_path = modopt["OWENS"]["OWENS_project_path"]


        jlPkg.activate(OWENS_path)
        jl.seval("using OWENS")
        jl.seval("using OWENSAero")

        master_file = modopt["OWENS"]["master_input"]

        # Initialize an OWENS model
        self.model = jl.OWENS.MasterInput(master_file)

        self.n_span = rotorse_options["n_span"]
        self.n_layers = rotorse_options["n_layers"]
        self.n_webs = rotorse_options["n_webs"]

        # Tower options
        n_height_tower = towerse_options["n_height"]
        n_layers_tower = towerse_options["n_layers"]
        n_mat = modopt["materials"]["n_mat"]

        # Strut options
        n_span_strut = strut_options["n_af_span"]
        n_layers_strut = strut_options["n_layers"]
        n_webs_strut = strut_options["n_webs"]


        # analysisType: unsteady # unsteady, steady, modal
        # turbineType: Darrieus #Darrieus, H-VAWT, ARCUS   
        # These should be in modeling options?   
        # Currently, they are just normal openmdao component options,
        # Not considering any WEIS data/input structure
        # Probably needs to be like modopt["crossflow"]["analysis_type"]
        self.analysis_type = modopt["OWENS"]["analysisType"]
        self.turbine_type = modopt["OWENS"]["turbineType"]
        self.eta = modopt["OWENS"]["eta"] # Now eta is a modeling option
        self.control_strategy = modopt["OWENS"]["controlStrategy"]
        self.aeroModel = modopt["OWENS"]["AModel"]
        self.adi_lib = modopt["OWENS"]["adi_lib"]
        self.adi_rootname = modopt["OWENS"]["adi_rootname"]
        # self.NumadSpec = modopt["OWENS"]["NumadSpec"] # All files go here
        # self.Turbulence = modopt["OWENS"]["TurbulenceSpec"] # All files go here
        self.ifw = modopt["OWENS"]["ifw"]
        self.windType = modopt["OWENS"]["windType"]
        self.windINPfilename = modopt["OWENS"]["windINPfilename"]
        self.ifw_libfiles = modopt["OWENS"]["ifw_libfiles"]
        self.n_blades = modopt["assembly"]["number_of_blades"]
        self.structuralModel = modopt["OWENS"]["structuralModel"]
        self.structuralNonlinear = modopt["OWENS"]["structuralNonlinear"]
        self.run_path = modopt["OWENS"]["run_path"] # What exactly is this path?

        # Reinitialize the model with the inputs from modeling options
        self.initialize_model()

        # number_of_grid_pts = modopt["number_of_grid_pts"]

        # Blade inputs, geometry and discretization
        self.add_input("Nbld", val=3, desc="number of blades")
        self.add_input("Blade_Radius", val=54.01123056)
        self.add_input("Blade_Height", val=110.1829092)
        self.add_input("towerHeight", val=3.0) # Towerheight is a modeling option in OWENS example, but I think it makes sense to be a input so that it can potentially be a dv


        # Blade inputs for composites
        # self.add_input('structuralModel', val='GX', desc="Structural models, GX, TNB, or ROM")
        # self.add_input('nonlinear', val=False, desc="[]")


        # Solver options (come from modeling options)
        
        # Control inputs
        if self.control_strategy == "constantRPM":
            self.add_input("RPM", val=17.2, desc="RPM")
            self.add_input("numTS", val=100)
            self.add_input("delta_t", val=0.01, units="s")

        # Aero parameters
        self.add_input("NSlices", val=30, desc="number of VAWTAero discritizations")
        self.add_input("ntheta", val=30, desc="number of VAWTAero azimuthal discretizations")

        # Environmental conditions
        # Maybe this fits better into load cases?

        # operation parameters
        self.add_input("rho", val=1.225, units="kg", desc="Fluid dendity")
        self.add_input("mu", val=1.7894e-5, units='kg/(m*s)', desc="Fluid dynamic viscosity")
        self.add_input("Vinf", val=17.2, units="m/s", desc="Inflow velocity") # Same for Vinf, it is now a modeling option in owens example, keeping it as input for now, so that DLCs can be taken care internally. currently connect to V_mean in geometry schema 

        # Blade outer_shape inputs
        self.add_input("airfoil_grid", val=np.linspace(0,1,self.n_span))
        self.add_discrete_input("airfoil_labels", val=self.n_span * [""], desc="1D array of names of airfoil shape labels.")
        self.add_input("chord_grid", val=np.linspace(0,1,self.n_span))
        self.add_input("chord_values", val=np.ones(self.n_span))
        self.add_input("twist_grid", val=np.linspace(0,1,self.n_span))
        self.add_input("twist_values", val=np.zeros(self.n_span))
        self.add_input("pitch_axis_grid", val=np.linspace(0,1,self.n_span))
        self.add_input("pitch_axis_values", val=np.zeros(self.n_span))

        # Blade structure inputs
        self.add_input("structure_grid", val=np.linspace(0,1,self.n_span))
        self.add_input("structure_ref_axis", val=np.zeros((self.n_span, 3)))

        self.add_input("web_start_nd_arc", val=np.zeros((self.n_webs, self.n_span)))
        self.add_input("web_end_nd_arc", val=np.zeros((self.n_webs, self.n_span)))

        self.add_discrete_input("layer_material", val="")
        self.add_input("layer_thickness", val=np.zeros((self.n_layers, self.n_span)))
        self.add_input("layer_start_nd_arc", val=np.zeros((self.n_layers, self.n_span)))
        self.add_input("layer_end_nd_arc", val=np.zeros((self.n_layers, self.n_span)))
        self.add_input("layer_fiber_orientation", val=np.zeros((self.n_layers, self.n_span)))

        # Tower inputs
        self.add_input("tower_grid", val=np.linspace(0,1, n_height_tower))
        self.add_input("tower_diameter", val=np.ones(n_height_tower))
        self.add_input("tower_ref_axis", val=np.ones([n_height_tower, 3]))

      
        self.add_discrete_input("tower_layer_name", val=n_layers_tower*[""])
        self.add_discrete_input("tower_layer_material", val=n_layers_tower*[""])
        self.add_input("tower_layer_thickness", val=np.zeros((n_layers_tower, n_height_tower)))
            # WEIS does not use these 
            # self.add_input("tower_layer_%d_start_nd_arc"%i, val=np.zeros(n_height_tower))
            # self.add_input("tower_layer_%d_end_nd_arc"%i, val=np.zeros(n_height_tower))
            # self.add_input("tower_layer_%d_fiber_orientation"%i, val=np.zeros(n_height_tower))

        # Strut inputs
        # YL: looks like the subcomponents in strut all use the same grid so now just one strut grid for everything
        self.add_input("strut_grid", val=np.linspace(0,1,n_span_strut))
        self.add_discrete_input("strut_airfoils", val=n_span_strut*[""])
        self.add_input("strut_chord", val=np.ones(n_span_strut))
        self.add_input("strut_twist", val=np.zeros(n_span_strut))
        self.add_input("strut_pitch_axis", val=np.zeros(n_span_strut))
        self.add_input("strut_ref_axis", val=np.zeros((n_span_strut, 3)))

        self.add_input("strut_web_start_nd_arc", val=0.35*np.ones((n_webs_strut,n_span_strut)))
        self.add_input("strut_web_end_nd_arc", val=0.65*np.ones((n_webs_strut, n_span_strut)))


        self.add_discrete_input("strut_layer_material", val=n_layers_strut*[""])
        self.add_input("strut_layer_thickness", val=np.zeros((n_layers_strut, n_span_strut)))
        self.add_input("strut_layer_start_nd_arc", val=np.zeros((n_layers_strut, n_span_strut)))
        self.add_input("strut_layer_end_nd_arc", val=np.ones((n_layers_strut, n_span_strut)))
        self.add_input("strut_layer_fiber_orientation", val=np.zeros((n_layers_strut, n_span_strut)))

        # Material inputs
        self.add_discrete_input("mat_name", val=n_mat * [""], desc="1D array of names of materials.")
        self.add_input('E',            val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
        self.add_input('G',            val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')
        self.add_input('nu',            val=np.zeros([n_mat, 3]), desc='2D array of the Poisson ratio of the materials. Each row represents a material, the three columns represent nu12, nu13 and nu23.')
        self.add_input('mat_rho',            val=np.zeros(n_mat), units="kg/m**3", desc='1D array of the density of the materials. For composites, this is the density of the laminate.')
        self.add_input('Xt',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.')
        self.add_input('Xc',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.')
        self.add_input('S',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Shear Strength (USS) of the materials. Each row represents a material, the three columns represent S12, S13 and S23.')
        self.add_input('wohler_m_mat',            val=np.zeros([n_mat]),                desc='2D array of the S-N fatigue slope exponent for the materials')
        self.add_input('ply_t',            val=np.zeros(n_mat), units="m", desc='1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.')
        self.add_input('unit_cost',            val=np.zeros(n_mat), units="USD/kg", desc='1D array of the unit costs of the materials.')
        self.add_input('wohler_A_mat',            val=np.zeros(n_mat), desc='1D array of the wohler intercept of the materials.')


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
        self.add_output("power", units="W", val=0.0)
        self.add_output("lcoe", units="USD/MW/h", val=0.0, desc="Pseudo levelized cost of energy")
        self.add_output("SF", val=0.0, desc="Safety factor constraint")
        self.add_output("fatigue_damage", val=0.0, desc="20 year fatigue damage")
        self.add_output("mass", units="kg", val=0.0)

        # discretizations

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
        # self.model.NuMad_geom_xlscsv_file_twr = self.NumadSpec["NuMad_geom_xlscsv_file_twr"]
        # self.model.NuMad_mat_xlscsv_file_twr = self.NumadSpec["NuMad_mat_xlscsv_file_twr"]
        # self.NuMad_geom_xlscsv_file_bld = self.NumadSpec["NuMad_geom_xlscsv_file_bld"]
        # self.NuMad_mat_xlscsv_file_bld = self.NumadSpec["NuMad_mat_xlscsv_file_bld"]
        # self.NuMad_geom_xlscsv_file_strut = self.NumadSpec["NuMad_geom_xlscsv_file_strut"]
        # self.NuMad_mat_xlscsv_file_strut = self.NumadSpec["NuMad_mat_xlscsv_file_strut"]

        # Initialize turbulenece
        self.model.ifw = self.ifw
        self.model.WindType = self.windType
        self.model.windINPfilename = self.windINPfilename
        self.model.ifw_libfile = self.ifw_libfiles

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
        self.declare_partials("power", "*", method="fd")
        self.declare_partials("lcoe", "*", method="fd")
        self.declare_partials("SF", "*", method="fd")
        self.declare_partials("fatigue_damage", "*", method="fd")

    def compute(self, inputs, outputs):
        modopt = self.options["modeling_options"]
        rotorse_options = self.options["rotorse_options"]
        towerse_options = self.options["towerse_options"]
        strut_options = self.options["strut_options"]
        n_span = rotorse_options["n_span"]
        n_layers = rotorse_options["n_layers"]
        n_webs = rotorse_options["n_webs"]
        n_mat = modopt["materials"]["n_mat"]

        blade_web_names = rotorse_options["web_name"]
        blade_layer_names = rotorse_options["layer_name"]
        blade_layer_materials = rotorse_options["layer_mat"]

        # WEIS does not have tower web and it's also empty in OWENS input yaml
        # tower_web_names = towerse_options["web_name"]
        # WEIS does not store tower names and materials in options
        # They are inputs and outputs
        tower_layer_names = towerse_options["layer_name"]
        tower_layer_materials = towerse_options["layer_mat"]
        n_height_tower = towerse_options["n_height"] # or towerse_options["n_height_tower"]
        n_layers_tower = towerse_options["n_layers"]
        n_webs_tower = towerse_options["n_webs"]

        strut_web_names = strut_options["web_name"]
        strut_layer_names = strut_options["layer_name"]
        # strut_layer_materials = strut_options["layer_mat"] # Get these from inputs


        path = self.run_path
        eta = modopt["eta"]
        number_of_blades = int(inputs["Nbld"][0])
        Blade_Radius = inputs["Blade_Radius"][0]
        Blade_Height = inputs["Blade_Height"][0]
        towerHeight = inputs["towerHeight"][0]

        # Blade dicretization
        blade_x = inputs["structure_ref_axis"][:,0]
        blade_y = inputs["structure_ref_axis"][:,1]
        blade_z = inputs["structure_ref_axis"][:,2]

        rho = inputs["rho"][0]
        Vinf = inputs["Vinf"][0]

        RPM = inputs["RPM"][0]
        numTS = int(inputs["numTS"][0])
        delta_t = inputs["delta_t"][0]

        Nslices = int(inputs["NSlices"][0])
        ntheta = int(inputs["ntheta"][0])

        ntelem = int(modopt["ntelem"])
        nbelem = int(modopt["nbelem"])
        ncelem = int(modopt["ncelem"])
        nselem = int(modopt["nselem"])

        # TODO: depending on the owens_yaml option, we can either update the model options directly, or write the intermediate yaml
        # and then update the model inputs
        # if use_yaml:
        #   self.update_model_with_yaml(inputs):
        # else update the model directly as follows:

        AModel = self.aeroModels
        mesh_type = self.turbine_type

        # Assemble the ordered dictionaries for the readNumad
        # NuMad_geom_xlscsv_file_twr should contain everything under tower from the yaml inputs

        # ---------- blade structure --------------
        blade_geo_dict = {}
        blade_geo_dict["outer_shape_bem"] = {}
        blade_geo_dict["outer_shape_bem"]["airfoil_position"] = {}
        blade_geo_dict["outer_shape_bem"]["airfoil_position"]["grid"] = inputs["airfoil_grid"]
        blade_geo_dict["outer_shape_bem"]["airfoil_position"]["labels"] = inputs["airfoil_labels"]
        blade_geo_dict["outer_shape_bem"]["chord"] = {}
        blade_geo_dict["outer_shape_bem"]["chord"]["grid"] = inputs["chord_grid"]
        blade_geo_dict["outer_shape_bem"]["chord"]["values"] = inputs["chord"]
        blade_geo_dict["outer_shape_bem"]["twist"] = {}
        blade_geo_dict["outer_shape_bem"]["twist"]["grid"] = inputs["twist_grid"]
        blade_geo_dict["outer_shape_bem"]["twist"]["values"] = inputs["twist"]
        blade_geo_dict["outer_shape_bem"]["pitch_axis"] = {}
        blade_geo_dict["outer_shape_bem"]["pitch_axis"]["grid"] = inputs["pitch_axis_grid"]
        blade_geo_dict["outer_shape_bem"]["pitch_axis"]["values"] = inputs["pitch_axis"]
        blade_geo_dict["outer_shape_bem"]["reference_axis"] = inputs["reference_axis"]
        blade_geo_dict["internal_structure_2d_fem"] = {}
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"] = inputs["structure_reference_axis"]
        blade_geo_dict["internal_structure_2d_fem"]["webs"] = {}
        blade_geo_dict["internal_structure_2d_fem"]["layers"] = {}

        # The grid for the internal structure 2d gem are all the same, they all the same as internal_structuure_2d_fem.s
        for i in range(n_webs):
            blade_geo_dict["internal_structure_2d_fem"]["webs"][i]["name"] = blade_web_names[i]
            blade_geo_dict["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"]["grid"] = inputs["structure_grid"]
            blade_geo_dict["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"]["values"] = inputs["web_start_nd_arc"][i,:]
            blade_geo_dict["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"]["grid"] = inputs["structure_grid"]
            blade_geo_dict["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"]["values"] = inputs["web_end_nd_arc"][i,:]

        for i in range(n_layers):
            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["name"] = blade_layer_names[i]
            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["material"] = inputs["layer_material"][i,:]
            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"]["values"] = inputs["layer_start_nd_arc"][i,:]
            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"]["grid"] = inputs["structure_grid"]
            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"]["values"] = inputs["layer_end_nd_arc"][i,:]
            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"]["grid"] = inputs["structure_grid"]

            # loop to find the material ply thickness and compute the n_plies for OWENS
            for m, name in enumerate(inputs["mat_name"]):
                if inputs["layer_%d_material"%i] == name:
                    n_plies = inputs["layer_thickness"][i,:]/inputs["ply_t"][m]

            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["n_plies"]["values"] = n_plies

            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["n_plies"]["grid"] = inputs["structure_grid"]
            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"]["values"] = inputs["layer_fiber_orientation"][i,:]
            blade_geo_dict["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"]["grid"] = inputs["structure_grid"]


        # ---------- tower structure --------------
        nd_tower_grid = inputs["tower_grid"] # WEIS gives non-dimensional grid
        tower_diameter = inputs["tower_diameter"]
        tower_length = arc_length(inputs["tower_reference_axis"])
        tower_grid = nd_tower_grid*tower_length
        
        
        tower_geo_dict = {}
        tower_geo_dict["outer_shape_bem"] = {}
        tower_geo_dict["outer_shape_bem"]["airfoil_position"] = {}
        tower_geo_dict["outer_shape_bem"]["airfoil_position"]["grid"] = tower_grid
        tower_geo_dict["outer_shape_bem"]["airfoil_position"]["labels"] = n_height_tower *["Circular"]
        tower_geo_dict["outer_shape_bem"]["chord"] = {}
        tower_geo_dict["outer_shape_bem"]["chord"]["grid"] = inputs["tower_grid"]
        tower_geo_dict["outer_shape_bem"]["chord"]["values"] = inputs["tower_chord"]
        tower_geo_dict["outer_shape_bem"]["twist"] = {}
        tower_geo_dict["outer_shape_bem"]["twist"]["grid"] = inputs["tower_grid"]
        tower_geo_dict["outer_shape_bem"]["twist"]["values"] = inputs["tower_twist"]
        tower_geo_dict["outer_shape_bem"]["pitch_axis"] = {}
        tower_geo_dict["outer_shape_bem"]["pitch_axis"]["grid"] = inputs["tower_grid"]
        tower_geo_dict["outer_shape_bem"]["pitch_axis"]["values"] = inputs["tower_pitch_axis"]
        tower_geo_dict["outer_shape_bem"]["reference_axis"] = inputs["tower_reference_axis"]
        tower_geo_dict["internal_structure_2d_fem"] = {}
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"] = inputs["tower_structure_reference_axis"] # Not needed in readNuMad
        tower_geo_dict["internal_structure_2d_fem"]["webs"] = {}
        tower_geo_dict["internal_structure_2d_fem"]["layers"] = {}

        # There should be no web for tower, web just an empty dict
        # for i in range(n_webs):
        #     tower_geo_dict["internal_structure_2d_fem"]["webs"][i]["name"] = rotorse_options["web_name"][i]
        #     tower_geo_dict["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"]["grid"] = inputs["tower_grid"]
        #     tower_geo_dict["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"]["values"] = inputs["web_%d_start_nd_arc"%i]
        #     tower_geo_dict["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"]["grid"] = inputs["tower_grid"]
        #     tower_geo_dict["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"]["values"] = inputs["web_%d_end_nd_arc"%i]

        for i in range(n_layers_tower):
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i] = {}
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["name"] = tower_layer_names[i]
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["material"] = inputs["tower_layer_material"][i]
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"] = {}
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"]["values"] = np.zeros(n_height_tower)
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"]["grid"] = tower_grid
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"] = {}
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"]["values"] = np.ones(n_height_tower)
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"]["grid"] = tower_grid

            # loop to find the material ply thickness and compute the n_plies for OWENS
            for m, name in enumerate(inputs["mat_name"]):
                if inputs["layer_%d_material"%i] == name:
                    n_plies = inputs["tower_layer_thickness"][i,:]/inputs["ply_t"][m]


            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["n_plies"] = {}
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["n_plies"]["values"] = n_plies
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["n_plies"]["grid"] = tower_grid
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"] = {}
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"]["values"] = np.zeros(n_height_tower)
            tower_geo_dict["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"]["grid"] = tower_grid

        # ---------- strut structure --------------
        strut_geo_dict = {}
        strut_geo_dict["outer_shape_bem"] = {}
        strut_geo_dict["outer_shape_bem"]["airfoil_position"] = {}
        # strut grid in OWENS yaml is dimensional but it doesn't matter
        # the strut dimension is determined by the strut mountpoint to tower and blade
        strut_geo_dict["outer_shape_bem"]["airfoil_position"]["grid"] = inputs["strut_grid"]
        strut_geo_dict["outer_shape_bem"]["airfoil_position"]["labels"] = inputs["strut_airfoil_labels"]
        strut_geo_dict["outer_shape_bem"]["chord"]["grid"] = inputs["strut_grid"]
        strut_geo_dict["outer_shape_bem"]["chord"]["values"] = inputs["strut_chord"]
        strut_geo_dict["outer_shape_bem"]["twist"]["grid"] = inputs["strut_grid"]
        strut_geo_dict["outer_shape_bem"]["twist"]["values"] = inputs["strut_twist"]
        strut_geo_dict["outer_shape_bem"]["pitch_axis"]["grid"] = inputs["strut_grid"]
        strut_geo_dict["outer_shape_bem"]["pitch_axis"]["values"] = inputs["strut_pitch_axis"]
        strut_geo_dict["outer_shape_bem"]["reference_axis"] = inputs["strut_reference_axis"] # Not needed in readNuMad
        strut_geo_dict["internal_structure_2d_fem"] = {}
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"] = inputs["strut_structure_reference_axis"]
        strut_geo_dict["internal_structure_2d_fem"]["webs"] = {}
        strut_geo_dict["internal_structure_2d_fem"]["layers"] = {}

        # The grid for the internal structure 2d gem are all the same, they all the same as internal_structure_2d_fem.s
        for i in range(n_webs):
            strut_geo_dict["internal_structure_2d_fem"]["webs"][i]["name"] = strut_web_names[i]
            strut_geo_dict["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"]["grid"] = inputs["strut_grid"]
            strut_geo_dict["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"]["values"] = inputs["strut_web_start_nd_arc"][i,:]
            strut_geo_dict["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"]["grid"] = inputs["strut_grid"]
            strut_geo_dict["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"]["values"] = inputs["strut_web_end_nd_arc"][i,:]

        for i in range(n_layers):
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["name"] = strut_layer_names[i]
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["material"] = inputs["strut_layer_material"][i,:]
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"]["values"] = inputs["strut_layer_start_nd_arc"][i,:]
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"]["grid"] = inputs["strut_grid"]
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"]["values"] = inputs["strut_layer_end_nd_arc"][i,:]
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"]["grid"] = inputs["strut_grid"]
            # strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["n_plies"]["values"] = inputs["strut_layer_nplies"]
            # loop to find the material ply thickness and compute the n_plies for OWENS
            for m, name in enumerate(inputs["mat_name"]):
                if inputs["strut_layer_material"][i] == name:
                    n_plies = inputs["strut_layer_thickness"][i,:]/inputs["ply_t"][m]
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["n_plies"]["values"] = n_plies
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["n_plies"]["grid"] = inputs["strut_grid"]
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"]["values"] = inputs["strut_layer_fiber_orientation"][i,:]
            strut_geo_dict["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"]["grid"] = inputs["strut_grid"]



        # materials dict
        material_dict = OrderedDict()
        for i in range(n_mat):
            material_dict[i]["name"] = inputs["mat_name"][i]
            material_dict[i]["ply_t"] = inputs["ply_t"][i]
            material_dict[i]["E"] = inputs["E"][i,:]
            material_dict[i]["G"] = inputs["G"][i,:]
            material_dict[i]["nu"] = inputs["nu"][i,:]
            material_dict[i]["rho"] = inputs["mat_rho"][i]
            material_dict[i]["Xt"] = inputs["Xt"][i,:]
            material_dict[i]["Xc"] = inputs["Xc"][i,:]
            material_dict[i]["unit_cost"] = inputs["unit_cost"][i]
            material_dict[i]["A"] = inputs["wohler_A_mat"][i]
            material_dict[i]["m"] = inputs["wohler_m_mat"][i]

        setup_outputs = jl.OWENS.setupOWENS(jl.OWENSAero, path, 
                                            rho=rho,
                                            Nslices=Nslices,
                                            ntheta=ntheta,
                                            RPM=RPM,
                                            Vinf=Vinf,
                                            eta=eta,
                                            B = number_of_blades,
                                            H = np.max(blade_z),
                                            R = np.max(blade_x),
                                            shapeZ=blade_z,
                                            shapeX=blade_x,
                                            ifw=self.Turbulence["ifw"],
                                            WindType=self.Turbulence["WindType"],
                                            delta_t=delta_t,
                                            numTS=numTS,
                                            adi_lib=self.adi_lib,
                                            adi_rootname=self.adi_rootname,
                                            AD15hubR = 0.1,
                                            windINPfilename=self.Turbulence["windINPfilename"],
                                            ifw_libfile=self.Turbulence["ifw_libfile"],
                                            NuMad_geom_xlscsv_file_twr=tower_geo_dict,
                                            NuMad_mat_xlscsv_file_twr=material_dict,
                                            NuMad_geom_xlscsv_file_bld=blade_geo_dict,
                                            NuMad_mat_xlscsv_file_bld=material_dict,
                                            NuMad_geom_xlscsv_file_strut=strut_geo_dict,
                                            NuMad_mat_xlscsv_file_strut=material_dict,
                                            ntelem=ntelem,
                                            nbelem=nbelem,
                                            ncelem=ncelem,
                                            nselem=nselem,
                                            joint_type = 0,
                                            c_mount_ratio = 0.05,
                                            strut_twr_mountpoint = [0.11,0.89], #TODO
                                            strut_bld_mountpoint = [0.11,0.89],
                                            AModel=AModel, #AD, DMS, AC
                                            DSModel="BV",
                                            RPI=True,
                                            cables_connected_to_blade_base = True,
                                            meshtype = mesh_type
                                            )

        # Parse setup outputs
        mymesh = setup_outputs[0]
        myel = setup_outputs[1]
        myort = setup_outputs[2]
        myjoint = setup_outputs[3]
        sectionPropsArray = setup_outputs[4]
        mass_twr = setup_outputs[5]
        mass_bld = setup_outputs[6]
        stiff_twr = setup_outputs[7]
        stiff_bld = setup_outputs[8]
        bld_precompinput = setup_outputs[9]
        # print("blade_precompinput omdao: ", bld_precompinput)
        bld_precompoutput = setup_outputs[10]
        plyprops_bld = setup_outputs[11]
        numadIn_bld = setup_outputs[12]
        lam_U_bld = setup_outputs[13]
        lam_L_bld = setup_outputs[14]
        twr_precompinput = setup_outputs[15]
        twr_precompoutput = setup_outputs[16]
        plyprops_twr = setup_outputs[17]
        numadIn_twr = setup_outputs[18]
        lam_U_twr = setup_outputs[19]
        lam_L_twr = setup_outputs[20]
        aeroForces = setup_outputs[21]
        deformAero = setup_outputs[22]
        mass_breakout_blds = setup_outputs[23]
        mass_breakout_twr = setup_outputs[24]
        system = setup_outputs[25]
        assembly = setup_outputs[26]
        sections = setup_outputs[27]
        AD15bldNdIdxRng = setup_outputs[28]
        AD15bldElIdxRng = setup_outputs[29]


        # This is boundary condition
        pBC = np.array([[1, 1, 0],
        [1, 2, 0],
        [1, 3, 0],
        [1, 4, 0],
        [1, 5, 0],
        [1, 6, 0]])

        if AModel == "AD":
            AD15On = True
        else:
            AD15On = False

        
        structural_model_inputs = jl.OWENS.Inputs(analysisType = self.structuralModel,
                                 tocp = np.array([0.0, 100000.1]),
                                 Omegaocp = np.array([RPM, RPM])/60,
                                 tocp_Vinf = np.array([0.0, 100000.1]),
                                 Vinfocp = np.array([Vinf, Vinf]),
                                 numTS=numTS,
                                 delta_t = delta_t,
                                 AD15On = AD15On,
                                 aeroLoadsOn = 2)


        feamodel = jl.OWENS.FEAModel(analysisType = self.structuralModel,
                                     outFilename = "none",
                                joint = myjoint,
                                platformTurbineConnectionNodeNumber = 1,
                                pBC=pBC,
                                nlOn=True,
                                numNodes = mymesh.numNodes,
                                RayleighAlpha = 0.05,
                                RayleighBeta = 0.05,
                                iterationType = "DI")
        
        
        unsteady_outputs = jl.OWENS.Unsteady_Land(structural_model_inputs,
                                                  system = system,
                                                  assembly = assembly,
                                                  topModel = feamodel,
                                                  topMesh = mymesh,
                                                  topEl = myel,
                                                  aero = aeroForces,
                                                  deformAero = deformAero
                                                  )
        
        # Parse unsteady run output
        t = unsteady_outputs[0]
        aziHist = unsteady_outputs[1]
        OmegaHist = unsteady_outputs[2]
        OmegaDotHist = unsteady_outputs[3]
        gbHist = unsteady_outputs[4]
        gbDotHist = unsteady_outputs[5]
        gbDotDotHist = unsteady_outputs[6]
        FReactionHist = unsteady_outputs[7]
        FTwrBsHist = unsteady_outputs[8]
        genTorque = unsteady_outputs[9]
        genPower = unsteady_outputs[10]
        torqueDriveShaft = unsteady_outputs[11]
        uHist = unsteady_outputs[12]
        uHist_prp = unsteady_outputs[13]
        epsilon_x_hist = unsteady_outputs[14]
        epsilon_y_hist = unsteady_outputs[15]
        epsilon_z_hist = unsteady_outputs[16]
        kappa_x_hist = unsteady_outputs[17]
        kappa_y_hist = unsteady_outputs[18]
        kappa_z_hist = unsteady_outputs[19]

        # Extract ultimate failure
        structural_failure_outputs = jl.OWENS.extractSF(bld_precompinput,
                                                        bld_precompoutput,
                                                        plyprops_bld,
                                                        numadIn_bld,
                                                        lam_U_bld,
                                                        lam_L_bld,
                                                        twr_precompinput,
                                                        twr_precompoutput,
                                                        plyprops_twr,
                                                        numadIn_twr,
                                                        lam_U_twr,
                                                        lam_L_twr,
                                                        mymesh,
                                                        myel,
                                                        myort,
                                                        number_of_blades,
                                                        epsilon_x_hist,
                                                        kappa_y_hist,
                                                        kappa_z_hist,
                                                        epsilon_z_hist,
                                                        kappa_x_hist,
                                                        epsilon_y_hist,
                                                        verbosity =2, #Verbosity 0:no printing, 1: summary, 2: summary and spanwise worst safety factor # epsilon_x_hist_1,kappa_y_hist_1,kappa_z_hist_1,epsilon_z_hist_1,kappa_x_hist_1,epsilon_y_hist_1,
                                                        LE_U_idx=1,
                                                        TE_U_idx=6,
                                                        SparCapU_idx=3,
                                                        ForePanelU_idx=2,
                                                        AftPanelU_idx=5,
                                                        LE_L_idx=1,
                                                        TE_L_idx=6,
                                                        SparCapL_idx=3,
                                                        ForePanelL_idx=2,
                                                        AftPanelL_idx=5,
                                                        Twr_LE_U_idx=1,
                                                        Twr_LE_L_idx=1,
                                                        AD15bldNdIdxRng = AD15bldNdIdxRng,
                                                        AD15bldElIdxRng = AD15bldElIdxRng,
                                                        strut_precompoutput=None)
        
        # Parse structural failure outputs
        massOwens = structural_failure_outputs[0]
        stress_U = structural_failure_outputs[1]
        SF_ult_U = structural_failure_outputs[2]
        SF_buck_U = structural_failure_outputs[3]
        stress_L = structural_failure_outputs[4]
        SF_ult_L = structural_failure_outputs[5]
        SF_buck_L = structural_failure_outputs[6]
        stress_TU = structural_failure_outputs[7]
        SF_ult_TU = structural_failure_outputs[8]
        SF_buck_TU = structural_failure_outputs[9]
        stress_TL = structural_failure_outputs[10]
        SF_ult_TL = structural_failure_outputs[11]
        SF_buck_TL = structural_failure_outputs[12]
        topstrainout_blade_U = structural_failure_outputs[13]
        topstrainout_blade_L = structural_failure_outputs[14]
        topstrainout_tower_U = structural_failure_outputs[15]
        topstrainout_tower_L = structural_failure_outputs[16]
        topDamage_blade_U = np.asarray(structural_failure_outputs[17])
        topDamage_blade_L = structural_failure_outputs[18]
        topDamage_tower_U = structural_failure_outputs[19]
        topDamage_tower_L = structural_failure_outputs[20]

        # Unpack outputs
        # Note: Change the return line in topRunDLC in OWENS to "return mass_breakout_twr, genPower, massOwens"
        # print("-FReactionHist[:,5]: ", -FReactionHist[:,5])
        outputs["mass"] = massOwens
        outputs["power"] = np.mean(-FReactionHist[:,5])*(RPM*2*np.pi/60)
        outputs["lcoe"] = massOwens/outputs["power"]

        # OWENS example uses ks aggregation
        maxFatiguePer20yr = np.max(topDamage_blade_U/t[-1]*60*60*20*365*24)
        minSF = np.min(SF_ult_U)

        # Other outputs for constraints
        outputs["SF"] = minSF
        outputs["fatigue_damage"] = maxFatiguePer20yr # - 1.0
        print("fatigue damage: ", outputs["fatigue_damage"])
        # power constraint can be imposed elsewhere
        # since it is already an output








            
            