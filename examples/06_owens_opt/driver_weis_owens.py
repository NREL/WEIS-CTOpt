#!/usr/bin/env python3

import os
from weis import weis_main


## File management
run_dir                 = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input          = os.path.join(run_dir,'..','00_setup','WINDIO_RM2.yaml')
fname_modeling_options  = run_dir + os.sep + 'modeling_options_OWENS_RM2.yml'
fname_analysis_options  = run_dir + os.sep + 'analysis_options_owens_DVs.yaml'


wt_opt, modeling_options, opt_options = weis_main(fname_wt_input, 
                                                  fname_modeling_options, 
                                                  fname_analysis_options)

## File management
# mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

# fname_wt_input  = mydir + os.sep + 'WINDIO_RM2.yaml'
# fname_modeling_options = mydir + os.sep + "modeling_options_OWENS_RM2.yml"
# fname_analysis_options = mydir + os.sep + "analysis_options_owens_DVs.yaml"


# wt_opt, modeling_options, analysis_options = run_weis(
#     fname_wt_input, 
#     fname_modeling_options, 
#     fname_analysis_options)


# if MPI:
#     rank = MPI.COMM_WORLD.Get_rank()
# else:
#     rank = 0
# if rank == 0:
#     # shutil.copyfile(os.path.join(analysis_options['general']['folder_output'],analysis_options['general']['fname_output']+'.yaml'), fname_wt_input)
#     print("Mass (kg) =", wt_opt["owens.mass"])
#     print("Power =", wt_opt["owens.power"])
#     # print("Floating platform mass (kg) =", wt_opt["floatingse.platform_mass"])

