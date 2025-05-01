import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time as timer
from mat4py import loadmat
import pickle
import shutil

from rosco.toolbox import control_interface as ROSCO_ci
from weis.aeroelasticse.CaseGen_General import case_naming
from weis.aeroelasticse.FAST_writer         import InputWriter_OpenFAST
from pCrunch.io import OpenFASTOutput
from pCrunch import LoadsAnalysis, PowerProduction, FatigueParams
from weis.dlc_driver.dlc_generator    import DLCGenerator
from weis.aeroelasticse.turbsim_file   import TurbSimFile
from weis.aeroelasticse.turbsim_util import generate_wind_files
from weis.aeroelasticse.CaseGen_General import CaseGen_General

from weis.dfsm.simulation_details import SimulationDetails
from weis.dfsm.dfsm_utilities import valid_extension,calculate_time,valid_extension_DISCON,compile_dfsm_results
from weis.dfsm.construct_dfsm import DFSM
from weis.dfsm.ode_algorithms import RK4,ABM4
from weis.dfsm.generate_wave_elev import generate_wave_elev

from scipy.interpolate import CubicSpline, interp1d

def evaluate_multi(case_data):

    # initialize controller interface
    case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])

    # run simulation
    if case_data['ode_method'] == 'RK4':

        output = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])

    elif case_data['ode_method'] == 'ABM4':

        output = ABM4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])


    # shut down controller
    case_data['param']['controller_interface'].kill_discon()


    return output


def run_serial(case_data_all):

    # initialize storage array for results
    outputs = []

    # loop through and evaluate simulations    
    for case_data in case_data_all:
        
        # initialize controller interface
        case_data['param']['controller_interface'] = ROSCO_ci.ControllerInterface(case_data['param']['lib_name'],param_filename=case_data['param']['param_filename'],**case_data['param']['args'])
        
        # run simulation
        t1 = timer.time()

        if case_data['ode_method'] == 'RK4':
            t,x,u,y = RK4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])

        elif case_data['ode_method'] == 'ABM4':
            t,x,u,y = ABM4(case_data['x0'],case_data['dt'],case_data['tspan'],case_data['dfsm'],case_data['param'])
        t2 = timer.time()

        print(t2-t1)
        # shut down controller
        case_data['param']['controller_interface'].kill_discon()

        # initialize
        sim_results = {}
        sim_results['case_no'] = case_data['case']
        sim_results['T_dfsm'] = t 
        sim_results['states_dfsm'] = x 
        sim_results['controls_dfsm'] = u 
        sim_results['outputs_dfsm'] = y

        outputs.append(sim_results)
    
    return outputs

def run_mpi(case_data_all,mpi_options):

    from mpi4py import MPI

    # mpi comm management
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sub_ranks = mpi_options['mpi_comm_map_down'][rank]

    size = len(sub_ranks)

    N_cases = len(case_data_all)
    N_loops = int(np.ceil(float(N_cases)/float(size)))

    sim_results = []
    for i in range(N_loops):
        idx_s    = i*size
        idx_e    = min((i+1)*size, N_cases)

        for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
            data   = [evaluate_multi, case_data]
            rank_j = sub_ranks[j]
            comm.send(data, dest=rank_j, tag=0)

        # for rank_j in sub_ranks:
        for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
            rank_j = sub_ranks[j]
            data_out = comm.recv(source=rank_j, tag=1)
            sim_results.append(data_out)

    # compile and store results
    outputs = []
    
    for icase,result_ in enumerate(sim_results):
        t = result_[0]
        x = result_[1]
        u = result_[2]
        y = result_[3]

        # initialize
        sim_results = {}

        # store
        sim_results['case_no'] = icase
        sim_results['T_dfsm'] = t 
        sim_results['states_dfsm'] = x 
        sim_results['controls_dfsm'] = u 
        sim_results['outputs_dfsm'] = y

        outputs.append(sim_results)



    return outputs




def  generate_wind_files_local(fst_vt, modopt, inputs, discrete_inputs, FAST_runDirectory, FAST_namingOut, wind_directory):

    DLCs = modopt['DLC_driver']['DLCs']
    # Initialize the DLC generator
    cut_in = float(inputs['V_cutin'])
    cut_out = float(inputs['V_cutout'])
    rated = float(inputs['Vrated'])
    ws_class = discrete_inputs['turbine_class']
    wt_class = discrete_inputs['turbulence_class']
    hub_height = float(inputs['hub_height'])
    rotorD = float(inputs['Rtip'])*2.
    PLExp = float(inputs['shearExp'])
    fix_wind_seeds = modopt['DLC_driver']['fix_wind_seeds']
    fix_wave_seeds = modopt['DLC_driver']['fix_wave_seeds']
    metocean = modopt['DLC_driver']['metocean_conditions']
    dlc_generator = DLCGenerator(
        metocean = metocean,
        **{
                'ws_cut_in': cut_in, 
                'ws_cut_out':cut_out, 
                'MHK': True,
            }
        )
    # dlc_generator = DLCGenerator(cut_in, cut_out, rated, ws_class, wt_class, fix_wind_seeds, fix_wave_seeds, metocean)
    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)

    # Initialize parametric inputs
    WindFile_type = np.zeros(dlc_generator.n_cases, dtype=int)
    WindFile_name = [''] * dlc_generator.n_cases
    rot_speed_initial = np.zeros(dlc_generator.n_cases)
    pitch_initial = np.zeros(dlc_generator.n_cases)
    shutdown_time = np.full(dlc_generator.n_cases, fill_value = 9999)
    azimuth_init = np.full(dlc_generator.n_cases, fill_value = 0)
    WindHd = np.zeros(dlc_generator.n_cases)
    WaveHs = np.zeros(dlc_generator.n_cases)
    WaveTp = np.zeros(dlc_generator.n_cases)
    WaveHd = np.zeros(dlc_generator.n_cases)
    WaveGamma = np.zeros(dlc_generator.n_cases)
    WaveSeed1 = np.zeros(dlc_generator.n_cases, dtype=int)
    TMax = np.zeros(dlc_generator.n_cases)
    TStart = np.zeros(dlc_generator.n_cases)
    dlc_label = [''] * dlc_generator.n_cases
    wind_seed = np.zeros(dlc_generator.n_cases, dtype=int)
    mean_wind_speed = np.zeros(dlc_generator.n_cases)
    yaw_misalignment = np.zeros(dlc_generator.n_cases)
    DT = np.full(dlc_generator.n_cases, fill_value = fst_vt['Fst']['DT'])
    aero_mod = np.full(dlc_generator.n_cases, fill_value = fst_vt['AeroDyn15']['AFAeroMod'])
    wake_mod = np.full(dlc_generator.n_cases, fill_value = fst_vt['AeroDyn15']['WakeMod'])
    dt_fvw = np.zeros(dlc_generator.n_cases)
    tMin = np.zeros(dlc_generator.n_cases)
    nNWPanels = np.zeros(dlc_generator.n_cases, dtype=int)
    nNWPanelsFree = np.zeros(dlc_generator.n_cases, dtype=int)
    nFWPanels = np.zeros(dlc_generator.n_cases, dtype=int)
    nFWPanelsFree = np.zeros(dlc_generator.n_cases, dtype=int)

    # fix hub height if MHK
    if modopt['flags']['marine_hydro']:
        # make grid span whole water depth for now, ref height will be actual hub height to get speed right
        # in deeper water, will need something better than this
        grid_height = 2. * np.abs(hub_height) - 1.e-3
        hub_height = grid_height/ 2
        ref_height = float(inputs['water_depth'] - np.abs(inputs['hub_height']))

        # Inflow wind wants these relative to sea bed
        fst_vt['InflowWind']['WindVziList'] = ref_height

    else:
        ref_height = hub_height


    for i_case in range(dlc_generator.n_cases):
        if dlc_generator.cases[i_case].turbulent:
            # Assign values common to all DLCs
            # Wind turbulence class
            if dlc_generator.cases[i_case].IECturbc > 0:    # use custom TI for DLC case
                dlc_generator.cases[i_case].IECturbc = str(dlc_generator.cases[i_case].IECturbc)
                dlc_generator.cases[i_case].IEC_WindType = 'NTM'
            else:
                dlc_generator.cases[i_case].IECturbc = wt_class
            # Reference height for wind speed
            if not dlc_generator.cases[i_case].RefHt:   # default RefHt is 0, use hub_height if not set
                dlc_generator.cases[i_case].RefHt = ref_height
            # Center of wind grid (TurbSim confusingly calls it HubHt)
            dlc_generator.cases[i_case].HubHt = np.abs(hub_height)
            # Height of wind grid, it stops 1 mm above the ground
            dlc_generator.cases[i_case].GridHeight = 2. * np.abs(hub_height) - 1.e-3
            # If OLAF is called, make wind grid 3x higher, taller, and wider
            if fst_vt['AeroDyn15']['WakeMod'] == 3:
                dlc_generator.cases[i_case].HubHt *= 3.
                dlc_generator.cases[i_case].GridHeight *= 3.
            # Width of wind grid, same of height
            dlc_generator.cases[i_case].GridWidth = dlc_generator.cases[i_case].GridHeight
            # Power law exponent of wind shear
            if dlc_generator.cases[i_case].PLExp < 0:    # use PLExp based on environment options (shear_exp), otherwise use custom DLC PLExp
                dlc_generator.cases[i_case].PLExp = PLExp
            # Length of wind grids
            dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
    turbsim_exe = shutil.which('turbsim')
    print('generating wind files')
    for i_case in range(dlc_generator.n_cases):
            WindFile_type[i_case] , WindFile_name[i_case] = generate_wind_files(
                dlc_generator, FAST_namingOut, wind_directory, rotorD, hub_height, turbsim_exe,i_case)
            
    # Set initial rotor speed and pitch if the WT operates in this DLC and available,
    # otherwise set pitch to 90 deg and rotor speed to 0 rpm when not operating
    # set rotor speed to rated and pitch to 15 deg if operating
    for i_case in range(dlc_generator.n_cases):
        if 'operating' in dlc_generator.cases[i_case].turbine_status:
            # We have initial conditions from WISDEM
            if ('U' in inputs) and ('Omega' in inputs) and ('pitch' in inputs):
                rot_speed_initial[i_case] = np.interp(dlc_generator.cases[i_case].URef, inputs['U'], inputs['Omega'])
                pitch_initial[i_case] = np.interp(dlc_generator.cases[i_case].URef, inputs['U'], inputs['pitch'])
            else:
                rot_speed_initial[i_case]   = fst_vt['DISCON_in']['PC_RefSpd'] * 30 / np.pi / fst_vt['ElastoDyn']['GBRatio']
                pitch_initial[i_case]       = 15

            if dlc_generator.cases[i_case].turbine_status == 'operating-shutdown':
                shutdown_time[i_case] = dlc_generator.cases[i_case].shutdown_time
        else:
            rot_speed_initial[i_case]   = 0.
            pitch_initial[i_case]       = 90.
            shutdown_time[i_case]      = 0
            aero_mod[i_case]            = 1
            wake_mod[i_case]            = 0

        # Wave inputs to HydroDyn
        WindHd[i_case] = dlc_generator.cases[i_case].wind_heading
        WaveHs[i_case] = dlc_generator.cases[i_case].wave_height
        WaveTp[i_case] = dlc_generator.cases[i_case].wave_period
        WaveHd[i_case] = dlc_generator.cases[i_case].wave_heading
        WaveGamma[i_case] = dlc_generator.cases[i_case].wave_gamma
        WaveSeed1[i_case] = dlc_generator.cases[i_case].wave_seed1

        # Other case info
        TMax[i_case] = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time
        TStart[i_case] = dlc_generator.cases[i_case].transient_time
        dlc_label[i_case] = dlc_generator.cases[i_case].label
        wind_seed[i_case] = dlc_generator.cases[i_case].RandSeed1
        mean_wind_speed[i_case] = dlc_generator.cases[i_case].URef
        yaw_misalignment[i_case] = dlc_generator.cases[i_case].yaw_misalign
        azimuth_init[i_case] = dlc_generator.cases[i_case].azimuth_init

        # Current
        # CurrMod = np.zeros(dlc_generator.n_cases,dtype=int)
        # CurrDIV = np.array([c.current for c in dlc_generator.cases])
        # CurrMod[CurrDIV > 0] = 1

        # Parameteric inputs
        case_inputs = {}
        # Main fst
        case_inputs[("Fst","DT")] = {'vals':DT, 'group':1}
        case_inputs[("Fst","TMax")] = {'vals':TMax, 'group':1}
        case_inputs[("Fst","TStart")] = {'vals':TStart, 'group':1}
       
        # Inflow wind
        case_inputs[("InflowWind","WindType")] = {'vals':WindFile_type, 'group':1}
        case_inputs[("InflowWind","HWindSpeed")] = {'vals':mean_wind_speed, 'group':1}
        case_inputs[("InflowWind","FileName_BTS")] = {'vals':WindFile_name, 'group':1}
        case_inputs[("InflowWind","Filename_Uni")] = {'vals':WindFile_name, 'group':1}
        case_inputs[("InflowWind","RefLength")] = {'vals':[rotorD], 'group':0}
        case_inputs[("InflowWind","PropagationDir")] = {'vals':WindHd, 'group':1}
        case_inputs[("InflowWind","RefHt_Uni")] = {'vals':[np.abs(hub_height)], 'group':0}   #TODO: check if this is the distance from sea level or bottom
        
        # Initial conditions for rotor speed, pitch, and azimuth
        case_inputs[("ElastoDyn","RotSpeed")] = {'vals':rot_speed_initial, 'group':1}
        case_inputs[("ElastoDyn","BlPitch1")] = {'vals':pitch_initial, 'group':1}
        case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
        case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]
        case_inputs[("ElastoDyn","Azimuth")] = {'vals':azimuth_init, 'group':1}
        
        # Yaw offset
        case_inputs[("ElastoDyn","NacYaw")] = {'vals':yaw_misalignment, 'group':1}
       
        # Inputs to HydroDyn
        case_inputs[("HydroDyn","WaveHs")] = {'vals':WaveHs, 'group':1}
        case_inputs[("HydroDyn","WaveTp")] = {'vals':WaveTp, 'group':1}
        case_inputs[("HydroDyn","WaveDir")] = {'vals':WaveHd, 'group':1}
        case_inputs[("HydroDyn","WavePkShp")] = {'vals':WaveGamma, 'group':1}
        case_inputs[("HydroDyn","WaveSeed1")] = {'vals':WaveSeed1, 'group':1}
       
        # Inputs to ServoDyn (parking), PitManRat and BlPitchF are ServoDyn modeling_options
        case_inputs[("ServoDyn","TPitManS1")] = {'vals':shutdown_time, 'group':1}
        case_inputs[("ServoDyn","TPitManS2")] = {'vals':shutdown_time, 'group':1}
        case_inputs[("ServoDyn","TPitManS3")] = {'vals':shutdown_time, 'group':1}

        # case_inputs[("HydroDyn","CurrMod")] = {'vals':CurrMod, 'group':1} 
        # case_inputs[("HydroDyn","CurrDIV")] = {'vals':CurrDIV, 'group':1} 

        # Inputs to AeroDyn (parking)
        case_inputs[("AeroDyn15","AFAeroMod")] = {'vals':aero_mod, 'group':1}
        case_inputs[("AeroDyn15","WakeMod")] = {'vals':wake_mod, 'group':1}

        # Inputs to OLAF
        case_inputs[("AeroDyn15","OLAF","DTfvw")] = {'vals':dt_fvw, 'group':1} 
        case_inputs[("AeroDyn15","OLAF","nNWPanels")] = {'vals':nNWPanels, 'group':1} 
        case_inputs[("AeroDyn15","OLAF","nNWPanelsFree")] = {'vals':nNWPanelsFree, 'group':1} 
        case_inputs[("AeroDyn15","OLAF","nFWPanels")] = {'vals':nFWPanels, 'group':1}
        case_inputs[("AeroDyn15","OLAF","nFWPanelsFree")] = {'vals':nFWPanelsFree, 'group':1}

        # DLC Label add these for the case matrix and delete from the case_list
        case_inputs[("DLC","Label")] = {'vals':dlc_label, 'group':1}
        case_inputs[("DLC","WindSeed")] = {'vals':wind_seed, 'group':1}
        case_inputs[("DLC","MeanWS")] = {'vals':mean_wind_speed, 'group':1}
        fst_vt['DLC'] = []

        # Append current DLC to full list of cases
        FAST_InputFile = modopt['General']['openfast_configuration']['OF_run_fst']
        case_list, case_name = CaseGen_General(case_inputs, FAST_runDirectory, FAST_InputFile)


        # Now delete the DLC-based case_inputs because they don't play nicely with aeroelasticse
        for case in case_list:
            for key in list(case):
                if key[0] == 'DLC':
                    del case[key]
            
    return dlc_generator, case_list, case_name,TMax, TStart



def write_FAST_files(fst_vt, FAST_runDirectory, FAST_namingOut):
    writer = InputWriter_OpenFAST()
    writer.fst_vt = fst_vt 
    writer.FAST_runDirectory = FAST_runDirectory
    writer.FAST_namingOut    = FAST_namingOut

    writer.execute()



def extract_simulation_results(modeling_data_path, model_options):

    # Get the path to the outb files in the folder, and arrange them
    outfiles = [os.path.join(modeling_data_path,f) for f in os.listdir(modeling_data_path) if valid_extension(f)]
    outfiles = sorted(outfiles)

    # Get the states, controls and outputs of the DFSM model
    reqd_states = model_options['reqd_states']
    reqd_controls = model_options['reqd_controls']
    reqd_outputs = model_options['reqd_outputs']


    # Get scaling and filtering arguments
    # --these arguments are used to scale and filter specific signals before constructing the DFSM model
    scale_args = model_options['scale_args']
    filter_args = model_options['filter_args']

    # Get additional options

    # 1. file name: Referes to file that has the linear models are available
    filename = model_options['linear_model_file']
    if filename == 'none':
        filename = None

    # 2. region: The region for which the linear model stored in the file is constructed
    region = model_options['region']

    # 3. add_dx2: Flag to add the first time/second time state derivatives to the model
    add_dx2 = model_options['add_dx2']

    # 4. tmin and tmax: If specified, the simulation data corresponding to t \in [tmin,tmax] is used to construct the DFSM
    # -- tmin is by default 0, and tmax is none
    tmin = model_options['tmin']; tmax = model_options['tmax']

    # Extract the simulation
    # instantiate class
    sim_detail = SimulationDetails(outfiles,
                                reqd_states,reqd_controls,reqd_outputs,
                                scale_args, filter_args,
                                tmin = tmin, add_dx2 = add_dx2,
                                linear_model_file = filename, region = region)

    # load and process data
    sim_detail.load_openfast_sim()

    return sim_detail

def construct_dfsm_helper(sim_detail, construction_options):

    # Extract options
    n_samples = construction_options['n_samples']

    # Sampling method
    sampling_method = construction_options['sampling_method']

    # Linear model type
    L_type = construction_options['L_type']

    # Nonlinear model type 
    N_type = construction_options['N_type']

    if N_type == 'None':
        N_type = None

    # Train split
    train_split = construction_options['train_split']


    # instantiate DFSM class
    dfsm = DFSM(sim_detail,
                n_samples = n_samples,
                sampling_method = sampling_method,
                L_type = L_type,
                N_type = N_type,
                train_split = train_split)
    
    # construct
    dfsm.construct_surrogate()

    return dfsm



def dfsm_wrapper(fst_vt, modopt, inputs, discrete_inputs, FAST_runDirectory = None, FAST_namingOut = None,mpi_options = None):

    print('----------------------------------------------------')
    print('Running DFSM')
    print('Derivative Function Surrogate Model')
    print('This program is licensed under Apache License Version 2.0 and comes with ABSOLUTELY NO WARRANTY.')
    print('----------------------------------------------------')

    # Load the stored DFSM model and run simulations with it
    general_options = modopt['DFSM']['general_options']
    model_options = modopt['DFSM']['model_options']

    reqd_states = model_options['reqd_states']
    reqd_controls = model_options['reqd_controls']
    reqd_outputs = model_options['reqd_outputs']
    scale_args = model_options['scale_args']
    
    # set run dir. THis is the directory where OpenFAST files are stored
    general_options['run_dir'] = modopt['General']['openfast_configuration']['OF_run_dir']

    # load DFSM model
    dfsm_file = general_options['dfsm_file']

    with open(dfsm_file,'rb') as handle:
        dfsm = pickle.load(handle)

    interp_type = model_options['interp_type']

    ode_method = model_options['ode_method']
    
    # setup interpolation method for LPV model
    dfsm.setup_LPV(interp_type)

    # If the test data is online, this implies, the DFSM will be used to run simulations
    # and save them in the current openfast run directory 
    testing_data_path = FAST_runDirectory

    # folder to save dfsm results
    dfsm_save_folder = testing_data_path + os.sep + 'dfsm_results'


    #------------------------------------------------------
    # Use case
    #------------------------------------------------------

    # Get usecase
    usecase = general_options['usecase']

    if usecase == 'closed-loop-simulation' :

        # generate wind files
        wind_directory = FAST_runDirectory + os.sep + 'wind'
        dlc_generator, case_list, case_name, TMax, TStart = generate_wind_files_local(fst_vt, modopt, inputs, discrete_inputs, FAST_runDirectory, FAST_namingOut, wind_directory)
        fst_vt['Fst']['TMax'] = TMax[0]
        fst_vt['Fst']['TStart'] = TStart[0]
        # Extract disturbance(s)
        test_dataset = []

        # initialize random number generator
        rng = np.random.default_rng(12345)

        for i_case,case in enumerate(case_list):
            ts_file     = TurbSimFile(case[('InflowWind','FileName_BTS')])
            ts_file.compute_rot_avg(fst_vt['ElastoDyn']['TipRad'])
            u_h         = ts_file['rot_avg'][0,:]
            tt          = ts_file['t']
            dt = case[('Fst','DT')]
            
            eta = generate_wave_elev(tt,dlc_generator.cases[i_case].wave_height,dlc_generator.cases[i_case].wave_period,rng)

            test_dataset.append({'time':tt, 'wind_speed': u_h,'wave_elev':eta})

        # number of test cases
        test_ind = np.arange(len(case_list))

        # generate OpenFAST files
        # This step generates the DISCON.IN and cp-ct-cq.txt files which are need to run closed-loop simulations
        
        for case in case_name:
            write_FAST_files(fst_vt, FAST_runDirectory, case)
        
        case_names = case_naming(len(case_name),'dfsm')

        
        # initialize storage lists
        output_list = []
        ct = []
        
        # required maginitude and fatigue channels
        #-- note: Some of these quantities are not always modeled using the DFSM. The corresponding outputs for these quantities will be 'Nan'
        magnitude_channels = {'LSShftF': ['RotThrust', 'LSShftFys', 'LSShftFzs'], 
                'LSShftM': ['RotTorq', 'LSSTipMys', 'LSSTipMzs'],
                'RootMc1': ['RootMxc1', 'RootMyc1', 'RootMzc1'],
                'RootMc2': ['RootMxc2', 'RootMyc2', 'RootMzc2'],
                'RootMc3': ['RootMxc3', 'RootMyc3', 'RootMzc3'],
                'TipDc1': ['TipDxc1', 'TipDyc1', 'TipDzc1'],
                'TipDc2': ['TipDxc2', 'TipDyc2', 'TipDzc2'],
                'TipDc3': ['TipDxc3', 'TipDyc3', 'TipDzc3'], 
                'TwrBsM': [ 'TwrBsMyt'],
                'NcIMUTA': ['NcIMUTAxs', 'NcIMUTAys', 'NcIMUTAzs']}

        fatigue_channels = {
                    'RootMc1': FatigueParams(slope=10),
                    'RootMc2': FatigueParams(slope=10),
                    'RootMc3': FatigueParams(slope=10),
                    'RootMyb1': FatigueParams(slope=10),
                    'RootMyb2': FatigueParams(slope=10),
                    'RootMyb3': FatigueParams(slope=10),
                    'TwrBsM': FatigueParams(slope=4),
                    'LSShftM': FatigueParams(slope=4),
                                        }

        DISCON_file = [os.path.join(testing_data_path,f) for f in os.listdir(testing_data_path) if valid_extension_DISCON(f)]
        GB_ratio = fst_vt['DISCON_in']['WE_GearboxRatio']
        case_data_all = []

        for idx,ind in enumerate(test_ind):

            case_data = {}
            case_data['case'] = idx
                
            test_data = test_dataset[ind]

            time = test_data['time']
            wind_speed = test_data['wind_speed']
            wind_fun = CubicSpline(time, wind_speed)

            x0 = np.zeros((dfsm.n_deriv,))
            # x0[0] = 20
            # x0[dfsm.gen_speed_ind] = 600

            t0 = time[0]; 
            tf = time[-1]
            tspan = [t0,tf]

            args = {'DT':dt,
                        'num_blade':2,'pitch':0}


            if 'Wave1Elev' in model_options['reqd_controls']:
                wave_elev = test_data['wave_elev']
                wave_fun = CubicSpline(time,wave_elev)

            # initialize param dict
            param = {}

            # populate dictonary with relevant info
            param['VS_GenEff'] = fst_vt['DISCON_in']['VS_GenEff']
            param['WE_GearboxRatio'] = fst_vt['DISCON_in']['WE_GearboxRatio']
            param['VS_RtPwr'] = fst_vt['DISCON_in']['VS_RtPwr']
            param['time'] = [t0]
            param['dt']= dt
            param['blade_pitch'] = [0]
            param['gen_torque'] = [7]
            param['t0'] = t0
            param['tf'] = tf
            param['w_fun'] = wind_fun 
            param['gen_speed_scaling'] = scale_args['state_scaling_factor'][-1]
            param['lib_name'] = fst_vt['ServoDyn']['DLL_FileName']
            if 'Wave1Elev' in model_options['reqd_controls']:
                param['wave_fun'] = wave_fun
            else:
                param['wave_fun'] = None

            param['num_blade'] = 2
            param['ny'] = dfsm.n_outputs
            param['args'] = args
            param['param_filename'] = DISCON_file[idx]
            



            # # start timer and solve for the states and controls
            # t1 = timer.time()
            # T_dfsm, states_dfsm, controls_dfsm,outputs_dfsm = RK4(x0, dt, tspan, dfsm, param)
            # t2 = timer.time()
            # dfsm.simulation_time = (t2-t1)

            case_data = {}
            case_data['case'] = idx
            case_data['param'] = param 
            case_data['dt'] = dt 
            case_data['x0'] = x0 
            case_data['tspan'] = tspan 
            case_data['dfsm'] = dfsm
            case_data['ode_method'] = ode_method

            case_data_all.append(case_data)

        if mpi_options['mpi_run']:

            # evaluate the closed loop simulations in parallel using MPI
            sim_outputs = run_mpi(case_data_all,mpi_options)

        else:

            # evaluate the closed loop simulations serially
            sim_outputs = run_serial(case_data_all)

        # plot properties
        markersize = 10
        linewidth = 1.5
        fontsize_legend = 16
        fontsize_axlabel = 18
        fontsize_tick = 12

        # save results
        if not os.path.exists(dfsm_save_folder):
            os.makedirs(dfsm_save_folder)

        for icase,sim_result in enumerate(sim_outputs):
            
            # extract results
            T_dfsm = sim_result['T_dfsm']
            states_dfsm = sim_result['states_dfsm']
            controls_dfsm = sim_result['controls_dfsm']
            outputs_dfsm = sim_result['outputs_dfsm']

        

            for iu,control in enumerate(reqd_controls):
        
                fig,ax = plt.subplots(1)
                
                ax.plot(T_dfsm,controls_dfsm[:,iu],label = 'DFSM')
                
                ax.set_title(control,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if general_options['save_results']:       
                    fig.savefig(dfsm_save_folder +os.sep+ control + '_' + str(icase) +'_comp.pdf')

                plt.close(fig)

            

            #------------------------------------------------------
            # Plot States
            #------------------------------------------------------
            for ix,state in enumerate(reqd_states):
                    
                fig,ax = plt.subplots(1)
                
                ax.plot(T_dfsm,states_dfsm[:,ix],label = 'DFSM')
                
                ax.set_title(state,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if general_options['save_results']:
                        
                    fig.savefig(dfsm_save_folder +os.sep+ state+ '_'+str(icase) +'_comp.pdf')

                plt.close(fig)

            #------------------------------------------
            # Plot Outputs
            #------------------------------------------
            for iy,output_ in enumerate(reqd_outputs):

                fig,ax = plt.subplots(1)
                
                ax.plot(T_dfsm,outputs_dfsm[:,iy],label = 'DFSM')
                
                
                ax.set_title(output_,fontsize = fontsize_axlabel)
                ax.set_xlim(tspan)
                ax.tick_params(labelsize=fontsize_tick)
                ax.legend(ncol = 2,fontsize = fontsize_legend)
                ax.set_xlabel('Time [s]',fontsize = fontsize_axlabel)
                
                if general_options['save_results']:
                        
                    fig.savefig(dfsm_save_folder +os.sep+ output_+ '_'+str(icase) + '_comp.pdf')

                plt.close(fig)

            # post processing
            case_name = case_names[icase]

            # compile results from DFSM
            OutData = compile_dfsm_results(T_dfsm,states_dfsm,controls_dfsm,outputs_dfsm,model_options['reqd_states'],
                                            model_options['reqd_controls'],model_options['reqd_outputs'],GB_ratio,TStart[icase])

            ct.append(OutData)

            # get output
            output = OpenFASTOutput.from_dict(OutData,case_name,magnitude_channels = magnitude_channels)

            output_list.append(output)
        
        # Collect outputs
        ss = {}
        et = {}
        dl = {}
        dam = {}
        

        loads_analysis = LoadsAnalysis(
            outputs = [],
            magnitude_channels = magnitude_channels,
            fatigue_channels = fatigue_channels
        )   

        for output in output_list:
            _name, _ss, _et, _dl, _dam = loads_analysis._process_output(output)
            ss[_name] = _ss
            et[_name] = _et
            dl[_name] = _dl
            dam[_name] = _dam

        summary_stats, extreme_table, DELs, Damage = loads_analysis.post_process(ss, et, dl, dam)
    
    return summary_stats,extreme_table,DELs,Damage,case_list, case_name, ct, dlc_generator, TMax, TStart






