#!/usr/bin/env python3
import os
import shutil
from weis.glue_code.runWEIS import run_weis
import wisdem.inputs as sch
import numpy as np
from wisdem.commonse.mpi_tools import MPI
import sys


## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

fname_wt_input  = mydir + os.sep + 'owens_land.yaml'
fname_modeling_options = mydir + os.sep + "modeling_options_OWENS_windioExample.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_owens_DVs.yaml"

# Change optimizer and output folder

analysis_override = {}
analysis_override['general'] = {}
analysis_override['general']['folder_output'] = os.path.join('outputs/18_OWENS_OptStudies/Opt_chord_radius_TSR/',"slsqp")
analysis_override['driver'] = {}
analysis_override['driver']['optimization'] = {}
analysis_override['driver']['optimization']['solver'] = "SLSQP"

wt_opt, modeling_options, analysis_options = run_weis(
    fname_wt_input, 
    fname_modeling_options, 
    fname_analysis_options, 
    analysis_override=analysis_override
    )


if MPI:
    rank = MPI.COMM_WORLD.Get_rank()
else:
    rank = 0
if rank == 0:
    # shutil.copyfile(os.path.join(analysis_options['general']['folder_output'],analysis_options['general']['fname_output']+'.yaml'), fname_wt_input)
    print("Mass (kg) =", wt_opt["owens.mass"])
    print("Power =", wt_opt["owens.power"])
    # print("Floating platform mass (kg) =", wt_opt["floatingse.platform_mass"])

