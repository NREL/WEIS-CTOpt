import numpy as np
import openmdao.api as om
from openmdao_owens_in_progress import *

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


