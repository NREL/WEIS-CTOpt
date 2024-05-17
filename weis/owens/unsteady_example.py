import numpy as np
import openmdao.api as om
from openmdao_owens import *

# Unsteady coupled analysis example
model = om.Group()

options = {
    "OWENS_directory": "/Users/yliao/Documents/Projects/CT-OPT/OWENS",
    "master_input": "/Users/yliao/repos/OWENS.jl/docs/src/literate/sampleOWENS.yml",
    "analysis_type": "unsteady",
    "turbine_type": "Darrieus",
    "control_strategy": "constantRPM",
    "aeroModel":"DMS",
    "adi_lib":"./../../../openfast/build/modules/aerodyn/libaerodyn_inflow_c_binding",
    "adi_rootname": "./ExampleB",
    "NumadSpec":{
        "NuMad_geom_xlscsv_file_twr": "data/NuMAD_Geom_SNL_5MW_D_TaperedTower.csv",
        "NuMad_mat_xlscsv_file_twr": "data/NuMAD_Materials_SNL_5MW.csv",
        "NuMad_geom_xlscsv_file_bld": "data/NuMAD_Geom_SNL_5MW_D_Carbon_LCDT_ThickFoils_ThinSkin.csv",
        "NuMad_mat_xlscsv_file_bld": "data/NuMAD_Materials_SNL_5MW.csv",
        "NuMad_geom_xlscsv_file_strut": "data/NuMAD_Geom_SNL_5MW_Struts.csv",
        "NuMad_mat_xlscsv_file_strut": "data/NuMAD_Materials_SNL_5MW.csv",

    },
    "TurbulenceSpec":{
        "ifw": False,
        "WindType": 3,
        "windINPfilename": "data/turbsim/115mx115m_30x30_20.0msETM.bts",
        "ifw_libfile": "./../../../openfast/build/modules/inflowwind/libifw_c_binding"
    },
    "number_of_blades": 3,
    "structuralModel": "GX",
    "structuralNonlinear": False,
    "run_path": "../../../OWENS.jl/docs/src/literate/",
}
model.add_subsystem("crossflow", OWENSUnsteadySetup(modeling_options=options))

# Note: Change the return line in topRunDLC in OWENS to "return mass_breakout_twr, genPower, massOwens" for this example to work

# Tower sanity check problem
tower_prob = om.Problem(model)
tower_prob.setup()
tower_prob.set_val("crossflow.towerHeight", 5.0)
tower_prob.run_model()
print("Tower height of 5 m: "+str(tower_prob.get_val("crossflow.tower_mass")))
tower_prob.set_val("crossflow.towerHeight", 3.0)
tower_prob.run_model()
print("Tower height of 3 m: "+str(tower_prob.get_val("crossflow.tower_mass")))

# run opt
tower_prob.driver = om.pyOptSparseDriver()
tower_prob.driver.options["optimizer"] = "SLSQP"

tower_prob.model.add_design_var('crossflow.towerHeight', lower=3.0, upper=5.0)
tower_prob.model.add_objective('crossflow.first_tower_mass')
tower_prob.setup()
tower_prob.run_driver()

print("Optimal height: "+str(tower_prob.get_val("crossflow.towerHeight")))
print("This should be 3 m")

# Working on more realistic problem2
prob = om.Problem(model)
prob.setup()
prob.set_val("crossflow.Blade_Height", 54.0)
prob.set_val("crossflow.Blade_Radius", 110.0)
prob.run_model()

# Right now, this isn't working because the generator power from OWENS are zero
print("Blade height of 54 m and blade radius of 110 m")
print("Mean power is : "+str(prob.get_val("crossflow.GenPower")))
print("Turbine mass is : "+str(prob.get_val("crossflow.GenPower")))
prob.set_val("crossflow.Blade_Height", 56.0)
prob.set_val("crossflow.Blade_Radius", 120.0)
prob.run_model()
print("Blade height of 56 m and blade radius of 120 m")
print("Mean power is : "+str(prob.get_val("crossflow.GenPower")))
print("Turbine mass is : "+str(prob.get_val("crossflow.GenPower")))




