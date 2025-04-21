
import os
from weis import weis_main

'''
This example uses WEIS to generate the platform model and run RAFT simulations, 
optimizing the platform
'''

## File management
run_dir                 = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input          = run_dir + os.sep + 'RM1.yaml'
fname_modeling_options  = run_dir + os.sep + 'modeling_options_RAFT.yaml'
fname_analysis_options  = run_dir + os.sep + 'analysis_options_ptfm_opt.yaml'


wt_opt, modeling_options, opt_options = weis_main(fname_wt_input, 
                                                  fname_modeling_options, 
                                                  fname_analysis_options)