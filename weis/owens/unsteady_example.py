import numpy as np
import openmdao.api as om
from openmdao_owens import *
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator

class discretization_x(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_control_pts", default=5, types=int)
        self.options.declare("number_of_grid_pts", default=5, types=int)
    def setup(self):
        number_of_control_pts = self.options["number_of_control_pts"]
        number_of_grid_pts = self.options["number_of_grid_pts"]
        self.add_input("quarter_span", val=1.0, units="m")
        self.add_input("max_radius", val=1.0, units="m")
        self.add_input("x_control_pts_grid", val=np.linspace(0,1,number_of_control_pts, endpoint=True))
        self.add_input("x_grid",val=np.zeros(number_of_grid_pts))
        self.add_output("blade_x", val=np.zeros(number_of_grid_pts), units="m")

    def setup_partials(self):
        # This can be set analytically from julia AD
        self.declare_partials("blade_x", ["quarter_span", "max_radius"], method="fd")

    def compute(self, inputs, outputs):
        x_control_pts_grid = inputs["x_control_pts_grid"]
        x_grid = inputs["x_grid"]
        # print("x_grid: ", x_grid)
        quarter_span = inputs["quarter_span"][0]
        max_radius = inputs["max_radius"][0]

        x_control_values = np.array([0, quarter_span, max_radius, quarter_span, 0])
        blade_x  = Akima1DInterpolator(x_control_pts_grid, x_control_values)(x_grid)

        outputs["blade_x"] = blade_x


# Unsteady coupled analysis example
model = om.Group()
x =  np.array([0.0, 0.03333333333333333, 0.06666666666666667, 0.1, 0.13333333333333333, 0.16666666666666666, 0.2, 0.23333333333333334, 0.26666666666666666, 0.3, 0.3333333333333333, 0.36666666666666664, 0.4, 0.43333333333333335, 0.4666666666666667, 0.5, 0.5333333333333333, 0.5666666666666667, 0.6, 0.6333333333333333, 0.6666666666666666, 0.7, 0.7333333333333333, 0.7666666666666667, 0.8, 0.8333333333333334, 0.8666666666666667, 0.9, 0.9333333333333333, 0.9666666666666667, 1.0])
options = {
    "OWENS_directory": "/Users/yliao/Documents/Projects/CT-OPT/OWENS",
    "master_input": "/Users/yliao/repos/OWENS.jl/docs/src/literate/sampleOWENS.yml",
    "analysis_type": "unsteady",
    "turbine_type": "Darrieus",
    "control_strategy": "constantRPM",
    "aeroModel":"DMS",
    "adi_lib": None, #"./../../../openfast/build/modules/aerodyn/libaerodyn_inflow_c_binding",
    "adi_rootname": "./ExampleB",
    "eta": 0.5,
    "NumadSpec":{
        "NuMad_geom_xlscsv_file_twr": "/Users/yliao/repos/OWENS.jl/examples/Optimization/data/NuMAD_Geom_SNL_5MW_D_TaperedTower.csv",
        "NuMad_mat_xlscsv_file_twr": "/Users/yliao/repos/OWENS.jl/examples/Optimization/data/NuMAD_Materials_SNL_5MW.csv",
        "NuMad_geom_xlscsv_file_bld": "/Users/yliao/repos/OWENS.jl/examples/Optimization/data/NuMAD_Geom_SNL_5MW_D_Carbon_LCDT_ThickFoils_ThinSkin.csv",
        "NuMad_mat_xlscsv_file_bld": "/Users/yliao/repos/OWENS.jl/examples/Optimization/data/NuMAD_Materials_SNL_5MW.csv",
        "NuMad_geom_xlscsv_file_strut": "/Users/yliao/repos/OWENS.jl/examples/Optimization/data/NuMAD_Geom_SNL_5MW_Struts.csv",
        "NuMad_mat_xlscsv_file_strut": "/Users/yliao/repos/OWENS.jl/examples/Optimization/data/NuMAD_Materials_SNL_5MW.csv",

    },
    "TurbulenceSpec":{
        "ifw": False,
        "WindType": 3,
        "windINPfilename": "/Users/yliao/repos/OWENS.jl/examples/Optimization/data/turbsim/115mx115m_30x30_20.0msETM.bts",
        "ifw_libfile": None, #"./../../../openfast/build/modules/inflowwind/libifw_c_binding"
    },
    "number_of_blades": 3,
    "number_of_grid_pts": len(x),
    "structuralModel": "GX",
    "structuralNonlinear": False,
    "run_path": "../../../OWENS.jl/docs/src/literate/",
}

model.add_subsystem("discretization", discretization_x(number_of_control_pts=5, number_of_grid_pts=len(x)), promotes=["*"])
model.add_subsystem("crossflow", OWENSUnsteadySetup(modeling_options=options), promotes=["*"])

# Note: Change the return line in topRunDLC in OWENS to "return mass_breakout_twr, genPower, massOwens" for this example to work


# Working on more realistic problem2
prob = om.Problem(model)
prob.model.set_input_defaults("quarter_span", 24.0)
prob.model.set_input_defaults("max_radius", 42.0)
prob.model.set_input_defaults("blade_y", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
prob.model.set_input_defaults("blade_z", np.array([0.0, 3.67276364, 7.34552728, 11.01829092, 14.69105456, 18.363818199999997, 22.03658184, 25.70934548, 29.38210912, 33.054872759999995, 36.727636399999994, 40.400400039999994, 44.07316368, 47.74592732, 51.41869096, 55.0914546, 58.76421824, 62.43698188, 66.10974551999999, 69.78250915999999, 73.45527279999999, 77.12803643999999, 80.80080007999999, 84.47356372, 88.14632736, 91.819091, 95.49185464, 99.16461828, 102.83738192, 106.51014556, 110.1829092]))
prob.model.set_input_defaults("x_grid", np.linspace(0, 1, len(x), endpoint=True))
prob.model.set_input_defaults("RPM", 15)
prob.setup()
prob.set_val("numTS", 200)
prob.set_val("delta_t", 0.05)
prob.run_model()

print("quarter_span of 24 m and max radius of 42 m")
print("Mean power is : "+str(prob.get_val("power")))
print("lcoe is : "+str(prob.get_val("lcoe")))
# exit()
# run opt
prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.recording_options["includes"] = ["*"]
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_constraints'] = True
prob.driver.recording_options['record_desvars'] = True
prob.driver.recording_options['record_inputs'] = True
prob.driver.recording_options['record_outputs'] = True
prob.driver.recording_options['record_residuals'] = True
prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
prob.model.add_design_var('quarter_span', lower=20, upper=30, ref=10)
prob.model.add_design_var('max_radius', lower=35, upper=50, ref=10)
prob.model.add_design_var('RPM', lower=10, upper=20, ref=10)
prob.model.add_objective('lcoe')
prob.model.add_constraint('fatigue_damage', upper=1.0)
prob.model.add_constraint('SF', lower=1.0)
prob.model.add_constraint('power', lower=0.0)
prob.model.approx_totals(method="fd", step=1e-4, step_calc="rel_avg")
prob.setup()
prob.set_val("numTS", 200)
prob.set_val("delta_t", 0.05)
prob.run_driver()
print("quarter_span: ", prob.get_val("quarter_span"))
print("max_radius: ", prob.get_val("max_radius"))
opt_radius_dist = np.array([0, prob.get_val("quarter_span")[0], prob.get_val("max_radius")[0], prob.get_val("quarter_span")[0], 0])
print("Optimal radius distribution: "+str(opt_radius_dist))






