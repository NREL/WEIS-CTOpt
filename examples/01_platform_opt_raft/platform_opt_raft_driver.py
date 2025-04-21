
import os
from weis import weis_main

'''
This example uses WEIS to generate the platform model and run RAFT simulations, 
optimizing the platform
'''

## File management
run_dir                 = os.path.dirname( os.path.realpath(__file__) )
fname_wt_input          = os.path.join(run_dir,'..','00_setup','RM1.yaml')
fname_modeling_options  = run_dir + os.sep + 'raft_modeling.yaml'
fname_analysis_options  = run_dir + os.sep + 'platform_opt_raft_analysis.yaml'


wt_opt, modeling_options, opt_options = weis_main(fname_wt_input, 
                                                  fname_modeling_options, 
                                                  fname_analysis_options)