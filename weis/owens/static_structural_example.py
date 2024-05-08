import numpy as np
import openmdao.api as om
from openmdao_owens_in_progress import *

model = om.Group()

options = {
    "OWENS_directory": "/Users/yliao/Documents/Projects/CT-OPT/OWENS",
    "master_input": "/Users/yliao/repos/OWENS.jl/docs/src/literate/sampleOWENS.yml",
    "analysis_type": "unsteady",
    "turbine_type": "Darrieus",
    "control_strategy": "constantRPM",
    "aeroModel":"DMS",
    "adi_lib":"./../../openfast/build/modules/aerodyn/libaerodyn_inflow_c_binding",
    "adi_rootname": "./SNL34m",
    "NumadSpec":{
        "NuMad_geom_xlscsv_file_twr": "/Users/yliao/repos/OWENS.jl/test/data/NuMAD_34m_TowerGeom.csv",
        "NuMad_mat_xlscsv_file_twr": "/Users/yliao/repos/OWENS.jl/test/data/NuMAD_34m_TowerMaterials.csv",
        "NuMad_geom_xlscsv_file_bld": "/Users/yliao/repos/OWENS.jl/test/data/NuMAD_SNL34mGeomBlades.csv",
        "NuMad_mat_xlscsv_file_bld": "/Users/yliao/repos/OWENS.jl/test/data/NuMAD_SNL34mMaterials.csv",
        "NuMad_geom_xlscsv_file_strut": "/Users/yliao/repos/OWENS.jl/test/data/NuMAD_SNL34mGeomStruts.csv",
        "NuMad_mat_xlscsv_file_strut": "/Users/yliao/repos/OWENS.jl/test/data/NuMAD_SNL34mMaterials.csv",

    },
    "TurbulenceSpec":{
        "ifw": False,
        "WindType": 3,
        "windINPfilename": "/Users/yliao/repos/OWENS.jl/test/data/turbsim/115mx115m_30x30_20.0msETM.bts",
        "ifw_libfile": "./../../openfast/build/modules/inflowwind/libifw_c_binding"
    },
    "number_of_blades": 2,
    "structuralModel": "GX",
    "structuralNonlinear": False,
    "run_path": "/Users/yliao/repos/OWENS.jl/examples",
}
model.add_subsystem("crossflow_structure", OWENSStructSetup(modeling_options=options))

prob = om.Problem(model)
prob.setup()
prob.set_val("crossflow_structure.Blade_Height", 40.0)
prob.run_model()
print("Blade height of 40 m: "+str(prob.get_val("crossflow_structure.blade_mass")))
prob.set_val("crossflow_structure.Blade_Height", 40.0)
prob.run_model()
print("Blade height of 40 m: "+str(prob.get_val("crossflow_structure.blade_mass")))

# run opt
# prob.driver = om.pyOptSparseDriver()
# prob.driver.options["optimizer"] = "SLSQP"

# prob.model.add_design_var('crossflow_structure.Blade_Height', lower=30.0, upper=50.0)
# prob.model.add_objective('crossflow_structure.blade_mass')
# prob.setup()
# prob.run_driver()

# print("Optimal Blade height: "+str(prob.get_val("crossflow_structure.Blade_Height")))
# print("This should be 30 m")