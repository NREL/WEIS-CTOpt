import numpy as np
import pandas as pd
import yaml
# from pathlib import Path
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
import openmdao.api as om
from openmdao.api                           import ExplicitComponent
from wisdem.commonse.utilities import arc_length
from openfast_io import FileTools
from weis.glue_code.mpi_tools              import MPI
from pCrunch import FatigueParams, AeroelasticOutput, Crunch
from weis.dlc_driver.dlc_generator    import DLCGenerator

# Juliacall for OWENS
try:
    from juliacall import Main as jl
    from juliacall import Pkg as jlPkg
except:
    print("Juliacall not installed. Please install it to use OWENS.")
from weis.owens.OWENS_output_reader import *
from collections import OrderedDict

if MPI:
    from mpi4py   import MPI

        
class OWENSUnsteadySetup(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("rotorse_options")
        self.options.declare("towerse_options")
        self.options.declare("strut_options")
        self.options.declare("opt_options")

    def setup(self):
        self.modopt = self.options['modeling_options']
        self.metocean = self.modopt['DLC_driver']['metocean_conditions']
        rotorse_options = self.options["rotorse_options"]
        towerse_options = self.options["towerse_options"]
        strut_options = self.options["strut_options"]
        opt_options = self.options["opt_options"]

        # set up an counter for output files
        self.design_counter = 0

        # jl.seval("println(Base.active_project())") # Print out the current project path
        jl.seval("using OWENS")
        

        self.n_span = rotorse_options["n_span"]
        self.n_layers = rotorse_options["n_layers"]
        self.n_webs = rotorse_options["n_webs"]

        # Tower options
        n_height_tower = towerse_options["n_height"]
        n_layers_tower = towerse_options["n_layers"]
        n_mat = self.modopt["materials"]["n_mat"]

        # Strut options
        n_span_strut = strut_options["n_af_span"]
        n_layers_strut = strut_options["n_layers"]
        n_webs_strut = strut_options["n_webs"]

        # Initialize directory
        self.OWENS_dir_base = opt_options['general']['folder_output']
        if not os.path.isabs(self.OWENS_dir_base):
            self.OWENS_dir_base = os.path.join(os.getcwd(), self.OWENS_dir_base)

        if MPI:
            rank = MPI.COMM_WORLD.Get_rank()
            self.OWENS_run_dir = os.path.join(self.OWENS_dir_base, "rank_%000d"%int(rank))
            # self.OWENS_namingOut = self.OWENS_InputFile+'_%000d'%int(rank)
        else:
            self.OWENS_run_dir = self.OWENS_dir_base
            # self.OWENS_namingOut = self.OWENS_InputFile

        if not os.path.exists(self.OWENS_run_dir):
            os.makedirs(self.OWENS_run_dir, exist_ok=True)

        # path to parent folder of airfoils folder
        self.data_path = self.modopt["OWENS"]["general"]["run_path"]


        # Blade inputs, geometry and discretization
        self.add_input("Nbld", val=3, desc="number of blades")

        # Configuration inputs
        self.add_input("hub_height", val=0.0) # Hub height

        
        # Control inputs
        if self.modopt["OWENS"]["general"]["controlStrategy"] == "tsrTracking":
            self.add_input("tsr", val=5.0, desc="TSR")

        # operation parameters
        self.add_input("rho", val=1.225, units="kg/m**3", desc="Fluid dendity")
        self.add_input("mu", val=1.7894e-5, units='kg/(m*s)', desc="Fluid dynamic viscosity")
        self.add_input("Vinf", val=17.2, units="m/s", desc="Inflow velocity") # Same for Vinf, it is now a modeling option in owens example, keeping it as input for now, so that DLCs can be taken care internally. currently connect to V_mean in geometry schema 

        # Blade outer_shape inputs
        self.add_input("blade_airfoil_grid", val=np.linspace(0,1,self.n_span))
        # self.add_discrete_input("airfoil_labels", val=self.n_span * [""], desc="1D array of names of airfoil shape labels.")
        self.add_input("blade_chord_grid", val=np.linspace(0,1,self.n_span))
        self.add_input("blade_chord_values", val=np.ones(self.n_span))
        self.add_input("blade_twist_grid", val=np.linspace(0,1,self.n_span))
        self.add_input("blade_twist_values", val=np.zeros(self.n_span))
        self.add_input("blade_pitch_axis_grid", val=np.linspace(0,1,self.n_span))
        self.add_input("blade_pitch_axis_values", val=np.zeros(self.n_span))

        # Blade structure inputs
        self.add_input("blade_structure_grid", val=np.linspace(0,1,self.n_span))
        # The reference axis is usually the same between outer shape bem and internal sturctgure fem
        self.add_input("blade_ref_axis", val=np.zeros((self.n_span, 3)))

        self.add_input("blade_web_start_nd_arc", val=np.zeros((self.n_webs, self.n_span)))
        self.add_input("blade_web_end_nd_arc", val=np.zeros((self.n_webs, self.n_span)))

        # self.add_discrete_input("layer_material", val="")
        self.add_input("blade_layer_thickness", val=np.zeros((self.n_layers, self.n_span)))
        self.add_input("blade_layer_start_nd_arc", val=np.zeros((self.n_layers, self.n_span)))
        self.add_input("blade_layer_end_nd_arc", val=np.zeros((self.n_layers, self.n_span)))
        self.add_input("blade_layer_fiber_orientation", val=np.zeros((self.n_layers, self.n_span)))

        # Tower inputs
        self.add_input("tower_grid", val=np.linspace(0,1, n_height_tower))
        self.add_input("tower_diameter", val=np.ones(n_height_tower))
        self.add_input("tower_ref_axis", val=np.ones([n_height_tower, 3]))

      
        # self.add_discrete_input("tower_layer_name", val=n_layers_tower*[""])
        # self.add_discrete_input("tower_layer_material", val=n_layers_tower*[""])
        self.add_input("tower_layer_thickness", val=np.zeros((n_layers_tower, n_height_tower)))
            # WEIS does not use these 
            # self.add_input("tower_layer_%d_start_nd_arc"%i, val=np.zeros(n_height_tower))
            # self.add_input("tower_layer_%d_end_nd_arc"%i, val=np.zeros(n_height_tower))
            # self.add_input("tower_layer_%d_fiber_orientation"%i, val=np.zeros(n_height_tower))

        # Strut inputs
        # YL: looks like the subcomponents in strut all use the same grid so now just one strut grid for everything
        self.add_input("strut_grid", val=np.linspace(0,1,n_span_strut))
        # self.add_discrete_input("strut_airfoils", val=n_span_strut*[""])
        self.add_input("strut_chord", val=np.ones(n_span_strut))
        self.add_input("strut_twist", val=np.zeros(n_span_strut))
        self.add_input("strut_pitch_axis", val=np.zeros(n_span_strut))
        self.add_input("strut_ref_axis", val=np.zeros((n_span_strut, 3)))

        self.add_input("strut_web_start_nd_arc", val=0.35*np.ones((n_webs_strut,n_span_strut)))
        self.add_input("strut_web_end_nd_arc", val=0.65*np.ones((n_webs_strut, n_span_strut)))


        # self.add_discrete_input("strut_layer_material", val=n_layers_strut*[""])
        self.add_input("strut_layer_thickness", val=np.zeros((n_layers_strut, n_span_strut)))
        self.add_input("strut_layer_start_nd_arc", val=np.zeros((n_layers_strut, n_span_strut)))
        self.add_input("strut_layer_end_nd_arc", val=np.ones((n_layers_strut, n_span_strut)))
        self.add_input("strut_layer_fiber_orientation", val=np.zeros((n_layers_strut, n_span_strut)))

        # Material inputs
        self.add_input('E',            val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
        self.add_input('G',            val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.')
        self.add_input('nu',            val=np.zeros([n_mat, 3]), desc='2D array of the Poisson ratio of the materials. Each row represents a material, the three columns represent nu12, nu13 and nu23.')
        self.add_input('mat_rho',            val=np.zeros(n_mat), units="kg/m**3", desc='1D array of the density of the materials. For composites, this is the density of the laminate.')
        self.add_input('Xt',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.')
        self.add_input('Xc',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.')
        self.add_input('S',           val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Ultimate Shear Strength (USS) of the materials. Each row represents a material, the three columns represent S12, S13 and S23.')
        self.add_input('wohler_m_mat',            val=np.zeros(n_mat),                desc='2D array of the S-N fatigue slope exponent for the materials')
        self.add_input('ply_t',            val=np.zeros(n_mat), units="m", desc='1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.')
        self.add_input('unit_cost',            val=np.zeros(n_mat), units="USD/kg", desc='1D array of the unit costs of the materials.')
        self.add_input('wohler_A_mat',            val=np.zeros(n_mat), desc='1D array of the wohler intercept of the materials.')

        # for fatigue
        self.add_input('blade_sparU_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
        self.add_input('blade_sparU_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
        self.add_input('blade_sparU_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')
        self.add_input('blade_sparL_wohlerexp',   val=1.0,   desc='Blade root Wohler exponent, m, in S/N curve S=A*N^-(1/m)')
        self.add_input('blade_sparL_wohlerA',   val=1.0, units="Pa",   desc='Blade root parameter, A, in S/N curve S=A*N^-(1/m)')
        self.add_input('blade_sparL_ultstress',   val=1.0, units="Pa",   desc='Blade root ultimate stress for material')

        # Environmental conditions
        self.add_input('V_cutin',     val=0.0, units='m/s',      desc='Minimum wind speed where turbine operates (cut-in)')
        self.add_input('V_cutout',    val=0.0, units='m/s',      desc='Maximum wind speed where turbine operates (cut-out)')
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')

        # Ignore floating parts for now
        # TODO: work on floating parts
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
        #     n_member = self.modopt["floating"]["members"]["n_members"]
        #     for k in range(n_member):
        #         n_height_mem = self.modopt["floating"]["members"]["n_height"][k]
        #         self.add_input(f"member{k}:joint1", np.zeros(3), units="m")
        #         self.add_input(f"member{k}:joint2", np.zeros(3), units="m")
        #         self.add_input(f"member{k}:s", np.zeros(n_height_mem))
        #         self.add_input(f"member{k}:s_ghost1", 0.0)
        #         self.add_input(f"member{k}:s_ghost2", 0.0)
        #         self.add_input(f"member{k}:outer_diameter", np.zeros(n_height_mem), units="m")
        #         self.add_input(f"member{k}:wall_thickness", np.zeros(n_height_mem-1), units="m")


        # # Moordyn inputs
        # mooropt = self.modopt["mooring"]
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
        self.add_output("power", units="W", val=0.0)
        self.add_output("lcoe", units="USD/MW/h", val=0.0, desc="Pseudo levelized cost of energy")
        self.add_output("SF", val=0.0, desc="Safety factor constraint")
        self.add_output("fatigue_damage", val=0.0, desc="20 year fatigue damage")
        self.add_output("mass", units="kg", val=0.0)

        # discretizations

    # def initialize_model(self):
    #     self.model.analysisType = self.analysis_type
    #     self.model.turbineType = self.turbine_type
    #     self.model.controlStrategy = self.control_strategy
    #     self.model.RPM = self.RPM
    #     self.model.AeroModel = self.aeroModel
    #     self.model.adi_lib = self.adi_lib
    #     self.model.adi_rootname = self.adi_rootname
    #     self.model.Nbld = int(self.n_blades)
    #     self.model.Nslices = self.Nslices
    #     self.model.ntheta = self.ntheta


    #     # Initialize turbulenece
    #     self.model.ifw = self.ifw
    #     self.model.WindType = self.windType
    #     self.model.windINPfilename = self.windINPfilename
    #     self.model.ifw_libfile = self.ifw_libfiles

    #     # initialize structural model
    #     self.model.structuralModel = self.structuralModel
    #     # self.model.nonlinear = self.structuralNonlinear

    #     # Simulation time
    #     self.model.numTS = self.numTS
    #     self.model.delta_t = self.delta_t


    def setup_partials(self):
        # This can be set analytically from julia AD
        self.declare_partials("power", ["blade_chord_values", "blade_twist_values", "blade_ref_axis", "tsr"], method="fd")
        self.declare_partials("lcoe", ["blade_chord_values", "blade_twist_values", "blade_ref_axis", "tsr"], method="fd")
        self.declare_partials("SF", ["blade_chord_values", "blade_twist_values", "blade_ref_axis", "tsr"], method="fd")
        self.declare_partials("fatigue_damage", ["blade_chord_values", "blade_twist_values", "blade_ref_axis", "tsr"], method="fd")

    def compute(self, inputs, outputs,  discrete_inputs, dicrete_outputs):
        modopt = self.options["modeling_options"]
        rotorse_options = self.options["rotorse_options"]
        towerse_options = self.options["towerse_options"]
        strut_options = self.options["strut_options"]
        n_span = rotorse_options["n_span"]
        n_layers = rotorse_options["n_layers"]
        n_webs = rotorse_options["n_webs"]
        n_mat = self.modopt["materials"]["n_mat"]

        blade_web_names = rotorse_options["web_name"]
        blade_layer_names = rotorse_options["layer_name"]
        blade_layer_materials = rotorse_options["layer_mat"]

        # WEIS does not have tower web and it's also empty in OWENS input yaml
        # tower_web_names = towerse_options["web_name"]
        # WEIS does not store tower names and materials in options
        # They are inputs and outputs
        tower_layer_names = towerse_options["layer_name"]
        tower_layer_materials = towerse_options["layer_material"]
        n_height_tower = towerse_options["n_height"] # or towerse_options["n_height_tower"]
        n_layers_tower = towerse_options["n_layers"]
        # n_webs_tower = towerse_options["n_webs"]

        strut_web_names = strut_options["web_name"]
        strut_layer_names = strut_options["layer_name"]
        strut_layer_materials = strut_options["layer_mat"] # Get these from inputs

        material_name = self.modopt["materials"]["mat_name"]

        number_of_blades = int(inputs["Nbld"][0])


        rho = inputs["rho"][0]
        Vinf = inputs["Vinf"][0]

        # ---------- blade structure --------------
        nd_blade_grid = inputs["blade_structure_grid"] # WEIS gives non-dimensional grid
        blade_accum_distances = arc_length(inputs["blade_ref_axis"])
        # OWENS wants dimensional grid
        blade_grid = blade_accum_distances # it should be the same too nd_blade_grid* np.maximum(blade_accum_distances)

        blade_geo_dict = {}
        blade_geo_dict["outer_shape_bem"] = {}
        blade_geo_dict["outer_shape_bem"]["blade_mountpoint"] = self.modopt["OWENS"]["blade"]["blade_mountpoint"]
        blade_geo_dict["outer_shape_bem"]["airfoil_position"] = {}
        blade_geo_dict["outer_shape_bem"]["airfoil_position"]["grid"] = nd_blade_grid # not sure how those numbers from
        # The airfoils in options are guaranteed to have the same number as the airfoil positions, so if not, just copy to the same length
        if len(rotorse_options["airfoil_labels"]) == len(nd_blade_grid):
            blade_geo_dict["outer_shape_bem"]["airfoil_position"]["labels"] = rotorse_options["airfoil_labels"]
        else:
            blade_geo_dict["outer_shape_bem"]["airfoil_position"]["labels"] = [rotorse_options["airfoil_labels"][0]]*len(nd_blade_grid)
        blade_geo_dict["outer_shape_bem"]["chord"] = {}
        blade_geo_dict["outer_shape_bem"]["chord"]["grid"] = nd_blade_grid
        blade_geo_dict["outer_shape_bem"]["chord"]["values"] = inputs["blade_chord_values"]
        blade_geo_dict["outer_shape_bem"]["twist"] = {}
        blade_geo_dict["outer_shape_bem"]["twist"]["grid"] = nd_blade_grid
        blade_geo_dict["outer_shape_bem"]["twist"]["values"] = inputs["blade_twist_values"]
        blade_geo_dict["outer_shape_bem"]["pitch_axis"] = {}
        blade_geo_dict["outer_shape_bem"]["pitch_axis"]["grid"] = nd_blade_grid
        blade_geo_dict["outer_shape_bem"]["pitch_axis"]["values"] = inputs["blade_pitch_axis_values"]
        # put the outer shape bem here but OWENS does not use outer shape bem reference axis
        # OWENS uses that from the internal structure 2d fem
        blade_geo_dict["outer_shape_bem"]["reference_axis"] = {}
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["x"] = {}
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["y"] = {}
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["z"] = {}
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["x"]["grid"] = nd_blade_grid
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["y"]["grid"] = nd_blade_grid
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["z"]["grid"] = nd_blade_grid
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["x"]["values"] = inputs["blade_ref_axis"][:,0]
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["y"]["values"] = inputs["blade_ref_axis"][:,1]
        blade_geo_dict["outer_shape_bem"]["reference_axis"]["z"]["values"] = inputs["blade_ref_axis"][:,2]
 
        # The reference axis is usually the same between outer shape bem and internal sturctgure fem       blade_geo_dict["outer_shape_bem"]["reference_axis"] = inputs["blade_ref_axis"] # Use the same as structural reference axis
        blade_geo_dict["internal_structure_2d_fem"] = {}
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"] = {}
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"] = {}
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"] = {}
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"] = {}
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"]["grid"] = nd_blade_grid
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"]["grid"] = nd_blade_grid
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"]["grid"] = nd_blade_grid
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"]["values"] = inputs["blade_ref_axis"][:,0]
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"]["values"] = inputs["blade_ref_axis"][:,1]
        blade_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"]["values"] = inputs["blade_ref_axis"][:,2]
        # The reference axis is usually the same between outer shape bem and internal sturctgure fems"] = inputs["blade_ref_axis"]
        blade_geo_dict["internal_structure_2d_fem"]["webs"] = []
        blade_geo_dict["internal_structure_2d_fem"]["layers"] = []

        # The grid for the internal structure 2d gem are all the same, they all the same as internal_structuure_2d_fem.s
        for i in range(n_webs):
            blade_i = {}
            blade_i["name"] = blade_web_names[i]
            blade_i["start_nd_arc"] = {}
            blade_i["start_nd_arc"]["grid"] = nd_blade_grid
            blade_i["start_nd_arc"]["values"] = inputs["blade_web_start_nd_arc"][i,:]
            blade_i["end_nd_arc"] = {}
            blade_i["end_nd_arc"]["grid"] = nd_blade_grid
            blade_i["end_nd_arc"]["values"] = inputs["blade_web_end_nd_arc"][i,:]

            blade_geo_dict["internal_structure_2d_fem"]["webs"].append(blade_i)

        for i in range(n_layers):
            blade_i = {}
            blade_i["name"] = blade_layer_names[i]
            blade_i["material"] = blade_layer_materials[i]
            if "web" in blade_i["name"]:
                blade_i["web"] = "web0"
            else:
                blade_i["start_nd_arc"] = {}
                blade_i["start_nd_arc"]["values"] = inputs["blade_layer_start_nd_arc"][i,:]
                blade_i["start_nd_arc"]["grid"] = nd_blade_grid
                blade_i["end_nd_arc"] = {}
                blade_i["end_nd_arc"]["values"] = inputs["blade_layer_end_nd_arc"][i,:]
                blade_i["end_nd_arc"]["grid"] = nd_blade_grid

            # loop to find the material ply thickness and compute the n_plies for OWENS
            for m, name in enumerate(material_name):
                if blade_layer_materials[i] == name:
                    n_plies = inputs["blade_layer_thickness"][i,:]/inputs["ply_t"][m]

            blade_i["n_plies"] = {}
            blade_i["n_plies"]["values"] = n_plies

            blade_i["n_plies"]["grid"] = nd_blade_grid
            blade_i["fiber_orientation"] = {}
            blade_i["fiber_orientation"]["values"] = inputs["blade_layer_fiber_orientation"][i,:]
            blade_i["fiber_orientation"]["grid"] = nd_blade_grid

            blade_geo_dict["internal_structure_2d_fem"]["layers"].append(blade_i)


        # ---------- tower structure --------------
        nd_tower_grid = inputs["tower_grid"] # WEIS gives non-dimensional grid
        tower_diameter = inputs["tower_diameter"]
        tower_accum_distances = arc_length(inputs["tower_ref_axis"])
        # OWENS wants dimensional grid
        tower_grid = tower_accum_distances # it should be equivalent nd_tower_grid*np.maximum(tower_accum_distances)
        
        
        tower_geo_dict = {}
        tower_geo_dict["outer_shape_bem"] = {}
        tower_geo_dict["outer_shape_bem"]["airfoil_position"] = {}
        tower_geo_dict["outer_shape_bem"]["airfoil_position"]["grid"] = nd_tower_grid
        tower_geo_dict["outer_shape_bem"]["airfoil_position"]["labels"] = n_height_tower *["Circular"]
        tower_geo_dict["outer_shape_bem"]["chord"] = {}
        tower_geo_dict["outer_shape_bem"]["chord"]["grid"] = nd_tower_grid
        tower_geo_dict["outer_shape_bem"]["chord"]["values"] = tower_diameter
        tower_geo_dict["outer_shape_bem"]["twist"] = {}
        tower_geo_dict["outer_shape_bem"]["twist"]["grid"] = nd_tower_grid
        tower_geo_dict["outer_shape_bem"]["twist"]["values"] = np.zeros(len(nd_tower_grid))
        tower_geo_dict["outer_shape_bem"]["pitch_axis"] = {}
        tower_geo_dict["outer_shape_bem"]["pitch_axis"]["grid"] = nd_tower_grid
        tower_geo_dict["outer_shape_bem"]["pitch_axis"]["values"] = 0.5*np.ones(len(nd_tower_grid))
        tower_geo_dict["outer_shape_bem"]["reference_axis"] = {}
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["x"] = {}
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["y"] = {}
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["z"] = {}
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["x"]["grid"] = nd_tower_grid
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["y"]["grid"] = nd_tower_grid
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["z"]["grid"] = nd_tower_grid
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["x"]["values"] = inputs["tower_ref_axis"][:,0]
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["y"]["values"] = inputs["tower_ref_axis"][:,1]
        tower_geo_dict["outer_shape_bem"]["reference_axis"]["z"]["values"] = inputs["tower_ref_axis"][:,2]
        
        tower_geo_dict["internal_structure_2d_fem"] = {}
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"] = {}
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"] = {}
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"] = {}
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"] = {}
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"]["grid"] = nd_tower_grid
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"]["grid"] = nd_tower_grid
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"]["grid"] = nd_tower_grid
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"]["values"] = inputs["tower_ref_axis"][:,0]
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"]["values"] = inputs["tower_ref_axis"][:,1]
        tower_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"]["values"] = inputs["tower_ref_axis"][:,2]# Not needed in readNuMad
        tower_geo_dict["internal_structure_2d_fem"]["webs"] = {}
        tower_geo_dict["internal_structure_2d_fem"]["layers"] = []


        for i in range(n_layers_tower):
            tower_i = {}
            tower_i["name"] = tower_layer_names[i]
            tower_i["material"] = tower_layer_materials[i]
            tower_i["start_nd_arc"] = {}
            tower_i["start_nd_arc"]["values"] = np.zeros(n_height_tower)
            tower_i["start_nd_arc"]["grid"] = nd_tower_grid
            tower_i["end_nd_arc"] = {}
            tower_i["end_nd_arc"]["values"] = np.ones(n_height_tower)
            tower_i["end_nd_arc"]["grid"] = nd_tower_grid

            # loop to find the material ply thickness and compute the n_plies for OWENS
            for m, name in enumerate(material_name):
                if tower_layer_materials[i] == name:
                    n_plies = inputs["tower_layer_thickness"][i,:]/inputs["ply_t"][m]


            tower_i["n_plies"] = {}
            tower_i["n_plies"]["values"] = n_plies
            tower_i["n_plies"]["grid"] = nd_tower_grid
            tower_i["fiber_orientation"] = {}
            tower_i["fiber_orientation"]["values"] = np.zeros(n_height_tower)
            tower_i["fiber_orientation"]["grid"] = nd_tower_grid

            tower_geo_dict["internal_structure_2d_fem"]["layers"].append(tower_i)

        # ---------- strut structure --------------

        nd_strut_grid = inputs["strut_grid"] # WEIS gives non-dimensional grid
        strut_accum_distances = arc_length(inputs["strut_ref_axis"])
        # OWENS wants dimensional grid
        strut_grid = strut_accum_distances
        strut_grid_rescale = 19.822*nd_strut_grid

    
        strut_geo_dict = {}
        strut_geo_dict["mountfraction_blade"] = {}
        strut_geo_dict["mountfraction_blade"] = strut_options["mountfraction_blade"]
        strut_geo_dict["mountfraction_tower"] = strut_options["mountfraction_tower"]
        strut_geo_dict["outer_shape_bem"] = {}
        strut_geo_dict["outer_shape_bem"]["airfoil_position"] = {}
        # strut grid in OWENS yaml is dimensional but it doesn't matter
        # the strut dimension is determined by the strut mountpoint to tower and blade
        strut_geo_dict["outer_shape_bem"]["airfoil_position"]["grid"] = nd_strut_grid
        strut_geo_dict["outer_shape_bem"]["airfoil_position"]["labels"] = strut_options["airfoils"]
        strut_geo_dict["outer_shape_bem"]["chord"] = {}
        strut_geo_dict["outer_shape_bem"]["chord"]["grid"] = nd_strut_grid
        strut_geo_dict["outer_shape_bem"]["chord"]["values"] = inputs["strut_chord"]
        strut_geo_dict["outer_shape_bem"]["twist"] = {}
        strut_geo_dict["outer_shape_bem"]["twist"]["grid"] = nd_strut_grid
        strut_geo_dict["outer_shape_bem"]["twist"]["values"] = inputs["strut_twist"]
        strut_geo_dict["outer_shape_bem"]["pitch_axis"] = {}
        strut_geo_dict["outer_shape_bem"]["pitch_axis"]["grid"] = nd_strut_grid
        strut_geo_dict["outer_shape_bem"]["pitch_axis"]["values"] = inputs["strut_pitch_axis"]
        strut_geo_dict["outer_shape_bem"]["reference_axis"] = {}
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["x"] = {}
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["y"] = {}
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["z"] = {}
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["x"]["grid"] = nd_strut_grid
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["y"]["grid"] = nd_strut_grid
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["z"]["grid"] = nd_strut_grid
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["x"]["values"] = inputs["strut_ref_axis"][:,0]
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["y"]["values"] = inputs["strut_ref_axis"][:,1]
        strut_geo_dict["outer_shape_bem"]["reference_axis"]["z"]["values"] = inputs["strut_ref_axis"][:,2]

        
        strut_geo_dict["internal_structure_2d_fem"] = {}
        # structure refenrence axis is not actuallly needed in Numad, still put values
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"] = {}
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"] = {}
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"] = {}
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"] = {}
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"]["grid"] = nd_strut_grid
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"]["grid"] = nd_strut_grid
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"]["grid"] = nd_strut_grid
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["x"]["values"] = inputs["strut_ref_axis"][:,0]
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["y"]["values"] = inputs["strut_ref_axis"][:,1]
        strut_geo_dict["internal_structure_2d_fem"]["reference_axis"]["z"]["values"] = inputs["strut_ref_axis"][:,2]
        strut_geo_dict["internal_structure_2d_fem"]["webs"] = []
        strut_geo_dict["internal_structure_2d_fem"]["layers"] = []

        # The grid for the internal structure 2d gem are all the same, they all the same as internal_structure_2d_fem.s
        for i in range(n_webs):

            strut_i = {}

            strut_i["name"] = strut_web_names[i]
            strut_i["start_nd_arc"] = {}
            strut_i["start_nd_arc"]["grid"] = nd_strut_grid
            strut_i["start_nd_arc"]["values"] = inputs["strut_web_start_nd_arc"][i,:]
            strut_i["end_nd_arc"] = {}
            strut_i["end_nd_arc"]["grid"] = nd_strut_grid
            strut_i["end_nd_arc"]["values"] = inputs["strut_web_end_nd_arc"][i,:]
            strut_geo_dict["internal_structure_2d_fem"]["webs"].append(strut_i)

        for i in range(n_layers):
            
            strut_i = {}
            strut_i["name"] = strut_layer_names[i]
            if "web" in strut_i["name"]:
                strut_i["web"] = "web0"
            strut_i["material"] = strut_layer_materials[i]
            strut_i["start_nd_arc"] = {}
            strut_i["start_nd_arc"]["values"] = inputs["strut_layer_start_nd_arc"][i,:]
            strut_i["start_nd_arc"]["grid"] = nd_strut_grid
            strut_i["end_nd_arc"] = {}
            strut_i["end_nd_arc"]["values"] = inputs["strut_layer_end_nd_arc"][i,:]
            strut_i["end_nd_arc"]["grid"] = nd_strut_grid
            # strut_i["n_plies"]["values"] = inputs["strut_layer_nplies"]
            # loop to find the material ply thickness and compute the n_plies for OWENS
            for m, name in enumerate(material_name):
                if strut_layer_materials[i] == name:
                    n_plies = inputs["strut_layer_thickness"][i,:]/inputs["ply_t"][m]
            strut_i["n_plies"] = {}
            strut_i["n_plies"]["values"] = n_plies
            strut_i["n_plies"]["grid"] = nd_strut_grid
            strut_i["fiber_orientation"] = {}
            strut_i["fiber_orientation"]["values"] = inputs["strut_layer_fiber_orientation"][i,:]
            strut_i["fiber_orientation"]["grid"] = nd_strut_grid
            strut_geo_dict["internal_structure_2d_fem"]["layers"].append(strut_i)

        # materials dict
        material_dict = []
        for i in range(n_mat):
            material_i = {}
            material_i["name"] = material_name[i]
            material_i["ply_t"] = inputs["ply_t"][i]
            material_i["E"] = inputs["E"][i,:]
            material_i["G"] = inputs["G"][i,:]
            material_i["nu"] = inputs["nu"][i,:]
            material_i["rho"] = inputs["mat_rho"][i]
            material_i["Xt"] = inputs["Xt"][i,:]
            material_i["Xc"] = inputs["Xc"][i,:]
            material_i["unit_cost"] = inputs["unit_cost"][i]
            material_i["m"] = [0.001,1.0,2.0,4.0,6.0,20.0] # hard coded cycle numbers
            material_i["A"] = (inputs["wohler_A_mat"][i] *(10**np.array(material_i["m"]))** (-1/inputs["wohler_m_mat"][i]))
            material_i["A"][-1] = 0 # hard coded the last one to be zero cause the interpolation did not go to zero
            material_i["S"] = inputs["S"][i]

            material_dict.append(material_i)

        # Put all dicts together
        yaml_dict = {}
        yaml_dict["name"] = "WINDIO"
        # Append all other options to the dict
        yaml_dict["assembly"] = self.modopt["assembly"]
        yaml_dict["assembly"]["hub_height"] = inputs["hub_height"][0]
        # {'turbine_class': 'I', 'turbulence_class': 'B', 'drivetrain': 'geared', 'rotor_orientation': 'upwind', 'number_of_blades': 3, 'hub_height': -25.2, 'rotor_diameter': 20.0, 'rated_power': 500000, 'lifetime': 25.0, 'marine_hydro': True, 'turbine_type': 'vertical'}
        yaml_dict["components"] = {}
        yaml_dict["components"]["tower"] = tower_geo_dict
        yaml_dict["components"]["blade"] = blade_geo_dict
        # save blade radius
        self.Blade_Radius = np.max(blade_geo_dict["outer_shape_bem"]["reference_axis"]["x"]["values"])
        strut_dict = []
        strut_geo_dict["name"] = "strut1"
        strut_dict.append(strut_geo_dict)
        yaml_dict["components"]["struts"] = strut_dict
        yaml_dict["materials"] = material_dict

        yaml_dict["environment"] = {}
        yaml_dict["environment"]["air_density"] = inputs["rho"][0]
        yaml_dict["environment"]["air_dyn_viscosity"] = inputs["mu"][0]
        yaml_dict["environment"]["gravity"] = np.array([0,0,-9.81])


        # Write out geometry info to yaml file
        FileTools.save_yaml(outdir=self.OWENS_run_dir, fname="data.yml", data_out=yaml_dict)

        # Get speeds
        DLCs = modopt['DLC_driver']['DLCs']
        fix_wind_seeds = modopt['DLC_driver']['fix_wind_seeds']
        fix_wave_seeds = modopt['DLC_driver']['fix_wave_seeds']
        metocean = modopt['DLC_driver']['metocean_conditions']

        cut_in = float(inputs['V_cutin'])
        cut_out = float(inputs['V_cutout'])

        dlc_generator = DLCGenerator(
            metocean,
            **{
                'ws_cut_in': cut_in, 
                'ws_cut_out':cut_out, 
                'MHK': modopt['flags']['marine_hydro'],
                'OWENS': True,
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
        URef = np.zeros(dlc_generator.n_cases)

        # fix hub height if MHK
        # if modopt['flags']['marine_hydro']:
            # make grid span whole water depth for now, ref height will be actual hub height to get speed right
            # in deeper water, will need something better than this
            # grid_height = 2. * np.abs(hub_height) - 1.e-3
        #     hub_height = grid_height/ 2
        #     ref_height = float(inputs['water_depth'] - np.abs(inputs['hub_height']))

        #     # Inflow wind wants these relative to sea bed
        #     fst_vt['InflowWind']['WindVziList'] = [ref_height]

        # else:
        #     ref_height = hub_height

        # The turbulence class setup is not currently used. This needs to be completed.
        wt_class = discrete_inputs['turbulence_class']
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
                # if not dlc_generator.cases[i_case].RefHt:   # default RefHt is 0, use hub_height if not set
                #     dlc_generator.cases[i_case].RefHt = ref_height
                # # Center of wind grid (TurbSim confusingly calls it HubHt)
                # if not dlc_generator.cases[i_case].HubHt:   # default HubHt is 0, use hub_height if not set
                #     dlc_generator.cases[i_case].HubHt = np.abs(hub_height)

                # if not dlc_generator.cases[i_case].GridHeight:   # default GridHeight is 0, use hub_height if not set
                #     dlc_generator.cases[i_case].GridHeight =  2. * np.abs(hub_height) - 1.e-3

                # if not dlc_generator.cases[i_case].GridWidth:   # default GridWidth is 0, use hub_height if not set
                #     dlc_generator.cases[i_case].GridWidth =  2. * hub_height - 1.e-3

                # # Power law exponent of wind shear
                # if dlc_generator.cases[i_case].PLExp < 0:    # use PLExp based on environment options (shear_exp), otherwise use custom DLC PLExp
                #     dlc_generator.cases[i_case].PLExp = PLExp
                # Length of wind grids
                dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].total_time
        
        # Only run serial right now 
        # The turbulence class setup is not currently used. OWENS seems to generate the turbsim files on its own. Might not need this.

        # for i_case in range(dlc_generator.n_cases):
        #     WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
        #         dlc_generator, self.FAST_namingOut, self.wind_directory, rotorD, hub_height, self.turbsim_exe, i_case)

        # Parameteric inputs
        # case_name = []
        # case_list = []
        # for i_case, case_inputs in enumerate(dlc_generator.owens_case_inputs):
        #     # Generate case list for DLC i
        #     dlc_label = DLCs[i_case]['DLC']
        #     case_list_i, case_name_i = CaseGen_General(case_inputs, self.OWENS_run_dir, "data.yaml", filename_ext=f'_DLC{dlc_label}_{i_case}')
        #     # Add DLC to case names
        #     case_name_i = [f'DLC{dlc_label}_{i_case}_{cni}' for cni in case_name_i]
            
        #     # Extend lists of cases
        #     case_list.extend(case_list_i)
        #     case_name.extend(case_name_i)

        # # Apply wind files to case_list (this info will be in combined case matrix, but not individual DLCs)
        # for case_i, wt, wf in zip(case_list,WindFile_type,WindFile_name):
        #     case_i[('InflowWind','WindType')] = wt
        #     case_i[('InflowWind','FileName_Uni')] = wf
        #     case_i[('InflowWind','FileName_BTS')] = wf

        # Save some case info
        self.TMax = [c.total_time for c in dlc_generator.cases]
        self.TStart = [c.transient_time for c in dlc_generator.cases]
        dlc_label = [c.label for c in dlc_generator.cases]

        # Common modeling options
        # Initialize OWENS modeling options
        owens_modeling_options = {}
        owens_modeling_options["OWENS_Options"] = {}
        owens_modeling_options["OWENS_Options"]["analysisType"] = modopt["OWENS"]["general"]["analysisType"]
        owens_modeling_options["OWENS_Options"]["AeroModel"] = modopt["OWENS"]["general"]["AeroModel"]
        owens_modeling_options["OWENS_Options"]["structuralModel"] = modopt["OWENS"]["general"]["structuralModel"]
        owens_modeling_options["OWENS_Options"]["controlStrategy"] = modopt["OWENS"]["general"]["controlStrategy"]
        owens_modeling_options["OWENS_Options"]["delta_t"] = self.modopt["OWENS"]["general"]["delta_t"]
        owens_modeling_options["OWENS_Options"]["dataOutputFilename"] = os.path.join(self.OWENS_run_dir,self.modopt["OWENS"]["general"]["dataOutputFilename"])
        owens_modeling_options["OWENS_Options"]["MAXITER"] = self.modopt["OWENS"]["general"]["MAXITER"]
        owens_modeling_options["OWENS_Options"]["TOL"] = self.modopt["OWENS"]["general"]["TOL"]
        owens_modeling_options["OWENS_Options"]["verbosity"] = self.modopt["OWENS"]["general"]["verbosity"]
        owens_modeling_options["OWENS_Options"]["VTKsaveName"] = self.modopt["OWENS"]["general"]["VTKsaveName"]
        owens_modeling_options["OWENS_Options"]["aeroLoadsOn"] = self.modopt["OWENS"]["general"]["aeroLoadsOn"]

        # Below are hard coded for now. They should be modified when the turbulence DLCs are finalized.
        owens_modeling_options["DLC_Options"] = {}
        owens_modeling_options["DLC_Options"]["IEC_std"] = r"'\"1-ED3\"'"
        owens_modeling_options["DLC_Options"]["WindChar"] = r"'\"A\"'"
        owens_modeling_options["DLC_Options"]["DLCs"] = ["1.1"]
        owens_modeling_options["DLC_Options"]["WindClass"] = 1
        owens_modeling_options["DLC_Options"]["turbsimsavepath"] = "./turbsimfiles"
        owens_modeling_options["DLC_Options"]["pathtoturbsim"] = None
        owens_modeling_options["DLC_Options"]["NumGrid_Z"] = 38
        owens_modeling_options["DLC_Options"]["NumGrid_Y"] = 26
        owens_modeling_options["DLC_Options"]["grid_oversize"] = 1.1
        owens_modeling_options["DLC_Options"]["regenWindFiles"] = False
        owens_modeling_options["DLC_Options"]["delta_t_turbsim"] = 0.05
        owens_modeling_options["DLC_Options"]["simtime_turbsim"] = 60.0
        
        owens_modeling_options["OWENSAero_Options"] = self.modopt["OWENS"]["OWENSAero_Options"]
        owens_modeling_options["OWENSFEA_Options"] = self.modopt["OWENS"]["OWENSFEA_Options"]
        owens_modeling_options["Mesh_Options"] = self.modopt["OWENS"]["Mesh_Options"]
        owens_modeling_options["OWENSOpenFASTWrappers_Options"] = self.modopt["OWENS"]["OWENSOpenFASTWrappers"]

        # Call run OWENS
        # Only running DLC 1.1 for now
        multipoint_power = np.zeros(dlc_generator.n_cases)
        multipoint_fatigue_blade_U = np.zeros(dlc_generator.n_cases)
        multipoint_fatigue_blade_L = np.zeros(dlc_generator.n_cases)
        multipoint_fatigue_tower_U = np.zeros(dlc_generator.n_cases)
        multipoint_fatigue_tower_L = np.zeros(dlc_generator.n_cases)
        multipoint_safety_factor = np.zeros(dlc_generator.n_cases)

        for i_case, case_inputs in enumerate(dlc_generator.cases):
            if case_inputs.label == "1.1":
                URef[i_case] = case_inputs.URef
                owens_modeling_options["DLC_Options"]["Vref"] = case_inputs.URef
                owens_modeling_options["DLC_Options"]["Vdesign"] = case_inputs.URef
                owens_modeling_options["DLC_Options"]["Vinf_range"] = [case_inputs.URef]

                self.run_owens_steady(inputs, owens_modeling_options, case_inputs)

                # Parse outputs using h5 files
                # Not sure why OWENS replace the last 3 characters of the output file name with .h5
                output_path = os.path.join(self.OWENS_run_dir,self.modopt["OWENS"]["general"]["dataOutputFilename"][:-3]+"h5")
                output_h5 = OWENSOutput(output_path, output_channels=["t", "FReactionHist", "OmegaHist", "massOwens", "topDamage_blade_U", "topDamage_blade_L", "topDamage_tower_U", "topDamage_tower_L" , "SF_ult_U", "SF_ult_L", "SF_ult_TU", "SF_ult_TL", "stress_U", "stress_L", "stress_TU", "stress_TL"]) 
                # A full list of output channels can be found in the fileio.jl file in OWENS
                # FReactionHist is reaction force history
                # omegaHist is rotor speed history
                # massOwens is the mass of the system
                # topDamage_blade_U is the fatigue damage of the blade upper surface
                # topDamage_blade_L is the fatigue damage of the blade lower surface
                # topDamage_tower_U is the fatigue damage of the tower upper surface
                # topDamage_tower_L is the fatigue damage of the tower lower surface
                # SF_ult_U is the ultimate safety factor of the blade upper surface
                # SF_ult_L is the ultimate safety factor of the blade lower surface
                # SF_ult_TU is the ultimate safety factor of the tower upper surface
                # SF_ult_TL is the ultimate safety factor of the tower lower surface
                # stress_U is the stress of the blade upper surface
                # stress_L is the stress of the blade lower surface
                # stress_TU is the stress of the tower upper surface
                # stress_TL is the stress of the tower lower surface
                # Although genTorque and genPower are in the output channels, they required the implementation and use of a generator function to have valid output. That's why in this example the power is computed from the reaction force history and rotor speed history.
                massOwens = output_h5["massOwens"]
                omegaHist = output_h5["OmegaHist"]
                FReactionHist = output_h5["FReactionHist"]
                # print("shape of FReactionHist: ", FReactionHist.shape)
                topDamage_blade_U = output_h5["topDamage_blade_U"]
                topDamage_blade_L = output_h5["topDamage_blade_L"]
                topDamage_tower_U = output_h5["topDamage_tower_U"]
                topDamage_tower_L = output_h5["topDamage_tower_L"]
                SF_ult_U= output_h5["SF_ult_U"]
                SF_ult_L= output_h5["SF_ult_L"]
                SF_ult_TU= output_h5["SF_ult_TU"]
                SF_ult_TL= output_h5["SF_ult_TL"]
                stress_L= output_h5["stress_L"]
                stress_U= output_h5["stress_U"]
                stress_TU= output_h5["stress_TU"]
                stress_TL= output_h5["stress_TL"]
                t = output_h5["t"]

                # find the index of transient time 
                t_start = np.where(t > self.TStart[i_case])[0][0]
                multipoint_power[i_case] = np.mean(-FReactionHist[t_start:,5]*omegaHist[t_start:])

                # Get fatigue and safety factors
                multipoint_fatigue_blade_U[i_case] = np.max(topDamage_blade_U/t[-1])
                multipoint_fatigue_blade_L[i_case] = np.max(topDamage_blade_L/t[-1])
                multipoint_fatigue_tower_U[i_case] = np.max(topDamage_tower_U/t[-1])
                multipoint_fatigue_tower_L[i_case] = np.max(topDamage_tower_L/t[-1])
                # maxFatigue = np.max([maxFatiguePer20yr_blade_L, maxFatiguePer20yr_blade_U, maxFatiguePer20yr_tower_L, maxFatiguePer20yr_tower_U])

                # Using pcrunch
                # fatigue_param_U = FatigueParams(load2stress=1.0,
                #                                     lifetime=t[-1],
                #                                     slope=inputs[f'blade_sparU_wohlerexp'],
                #                                     ult_stress=inputs[f'blade_sparU_ultstress'],
                #                                     S_intercept=inputs[f'blade_sparU_wohlerA']*1e6,
                #                                     rainflow_bins=20)
                # fatigue_param_L = FatigueParams(load2stress=1.0,
                #                                     lifetime=t[-1],
                #                                     slope=inputs[f'blade_sparL_wohlerexp'],
                #                                     ult_stress=inputs[f'blade_sparL_ultstress'],
                #                                     S_intercept=inputs[f'blade_sparL_wohlerA']*1e6,
                #                                     rainflow_bins=20)
                # stress_L_reordered = {}
                # stress_U_reordered = {}
                # # for layer in range(n_layers):
                # for i in range(np.shape(stress_L)[1]):
                #     for j in range(np.shape(stress_L)[2]):
                        
                #         stress_L_reordered[str(i)+str(j)] = stress_L[:, i, j, 0].tolist()
                # for i in range(np.shape(stress_U)[1]):
                #     for j in range(np.shape(stress_U)[2]):
                        
                #         stress_U_reordered[str(i)+str(j)] = stress_U[:, i, j, 0].tolist()
                # stress_L_reordered["Time"] = t.tolist()
                # stress_U_reordered["Time"] = t.tolist()
                # chan_labels_U = [str(i)+str(j) for j in range(np.shape(stress_U)[2]) for i in range(np.shape(stress_U)[1]) ]
                # chan_labels_L = [str(i)+str(j) for j in range(np.shape(stress_L)[2]) for i in range(np.shape(stress_L)[1]) ]

                # myfatiguesU = {}
                # myfatiguesL = {}
                # for i in chan_labels_U:
                #     myfatiguesU[i] = fatigue_param_U
                # for i in chan_labels_L:
                #     myfatiguesL[i] = fatigue_param_L
                
                # myobj_L = AeroelasticOutput(stress_L_reordered, fatigue_channels=myfatiguesL)
                # myobj_U = AeroelasticOutput(stress_U_reordered, fatigue_channels=myfatiguesU)

                # delsU, damsU = myobj_U.get_DELs(return_damage=True)
                # delsL, damsL = myobj_L.get_DELs(return_damage=True)
                
                # topDamage_blade_U_pcrunch = np.zeros_like(topDamage_blade_U)
                # topDamage_blade_L_pcrunch = np.zeros_like(topDamage_blade_L)
                # for i in range(np.shape(stress_U)[1]):
                #     for j in range(np.shape(stress_U)[2]):
                #         topDamage_blade_U_pcrunch[i,j] = damsU[str(i)+str(j)]
                # for i in range(np.shape(stress_L)[1]):
                #     for j in range(np.shape(stress_L)[2]):
                #         topDamage_blade_L_pcrunch[i,j] = damsL[str(i)+str(j)]


                minSF_U = np.min(SF_ult_U)
                minSF_L = np.min(SF_ult_L)
                minSF_TU = np.min(SF_ult_TU)
                minSF_TL = np.min(SF_ult_TL)

                multipoint_safety_factor[i_case] = np.min([minSF_L, minSF_TL, minSF_U, minSF_TU])

            else:
                print("Only support DLC 1.1 for now")


        self.design_counter += 1

        outputs["mass"] = massOwens
        if 'user_probability' in self.options['modeling_options']['DLC_driver']['metocean_conditions']:
            v_prob = modopt['DLC_driver']['metocean_conditions']['user_probability']['speed']
            user_probability = modopt['DLC_driver']['metocean_conditions']['user_probability']['probability']
            # compute the actual probability based on the speeds in DLC 1.1
            prob = np.interp(URef,v_prob,user_probability)
            # self._reset_probabilities() # What does this do?
            prob = prob / prob.sum()
          

        multipoint_power = np.average(multipoint_power, weights=prob)       
        outputs["power"] = multipoint_power
        outputs["lcoe"] = massOwens/outputs["power"]
        if outputs["lcoe"] < 0: # avoid negative lcoe. It might still want to produce negative lcoe even when positive power constraint is used.
            outputs["lcoe"] = 100


        # # Other outputs for constraints
        outputs["SF"] = np.min(multipoint_safety_factor)
        # Computed the fatigue damage per 20 years with the probability
        fatigue_blade_U = np.average(multipoint_fatigue_blade_U, weights=prob)
        fatigue_blade_L = np.average(multipoint_fatigue_blade_L, weights=prob)
        fatigue_tower_U = np.average(multipoint_fatigue_tower_U, weights=prob)
        fatigue_tower_L = np.average(multipoint_fatigue_tower_L, weights=prob)
        maxFatigue = np.max([fatigue_blade_U, fatigue_blade_L, fatigue_tower_U, fatigue_tower_L])
        outputs["fatigue_damage"] = maxFatigue*60*60*24*365*20 # convert to 20 years


    def run_owens_steady(self, inputs, owens_modeling_options, case):
        # Run OWENS

        owens_modeling_options["OWENS_Options"]["numTS"] = int(case.total_time/self.modopt["OWENS"]["general"]["delta_t"])
    
        # These will be determined by DLC generator
        if case.transient_time > 0:
            owens_modeling_options["OWENS_Options"]["Prescribed_Vinf_time_controlpoints"] = [0.0, case.transient_time,  case.total_time]
            owens_modeling_options["OWENS_Options"]["Prescribed_RPM_time_controlpoints"] = [0.0,  case.transient_time,  case.total_time]
            owens_modeling_options["OWENS_Options"]["Prescribed_Vinf_Vinf_controlpoints"] = [case.URef, case.URef, case.URef]
        else:
            owens_modeling_options["OWENS_Options"]["Prescribed_Vinf_time_controlpoints"] = [0.0, case.total_time]
            owens_modeling_options["OWENS_Options"]["Prescribed_RPM_time_controlpoints"] = [0.0, case.total_time]
            owens_modeling_options["OWENS_Options"]["Prescribed_Vinf_Vinf_controlpoints"] = [case.URef, case.URef]
        # Write out modeling option file

        FileTools.save_yaml(outdir=self.OWENS_run_dir, fname=f"OWENS_Opt_U{case.URef}.yml", data_out=owens_modeling_options)
        
        # update vtk output dir
        if self.modopt["OWENS"]["general"]["write_intermediate_design"]:
            owens_modeling_options["OWENS_Options"]["VTKsaveName"] = os.path.join(self.OWENS_run_dir,f"vtk_{self.design_counter:03d}_U{case.URef}","windio")

        
        if self.modopt["OWENS"]["general"]["controlStrategy"] == "tsrTracking":
            TSR = inputs["tsr"][0]
            owens_modeling_options["OWENS_Options"]["Prescribed_RPM_RPM_controlpoints"] = np.array(owens_modeling_options["OWENS_Options"]["Prescribed_Vinf_Vinf_controlpoints"])*TSR*30/np.pi/self.Blade_Radius
            owens_modeling_options["OWENS_Options"]["controlStrategy"] = "prescribedRPM" # still use prescribedRPM for OWENS
            FileTools.save_yaml(outdir=self.OWENS_run_dir, fname=f"OWENS_Opt_U{case.URef}.yml", data_out=owens_modeling_options)
            # print("TSR is: ", TSR)


        jl.OWENS.runOWENSWINDIO(os.path.join(self.OWENS_run_dir,f"OWENS_Opt_U{case.URef}.yml"), os.path.join(self.OWENS_run_dir,"data.yml"),self.data_path)







            
            