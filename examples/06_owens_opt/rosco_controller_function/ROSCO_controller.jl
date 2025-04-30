module ROSCO_controller
# Note that this module is not currently used in WEIS-CTOpt.
# Futher development is needed to make this module work within the WEIS-CTOpt framework.

__precompile__()

export rosco_controller_init, rosco_controller_calc, rosco_controller_end
import Libdl
global rosco_controller_lib
global sym_time_step
global avrSWAP = zeros(Cfloat, 150)
global aviFAIL = [0]
global accINFILE = "<path to DISCON.IN>"
global avcOUTNAME = "outfile.txt"
global avcMSG = "nothing"
 

"""
    rosco_controller(rosco_controller_lib_filename ;HWindSpeed=6.87,turbsim_filename="path/test.bts")
 
calls inflow wind init
 
# Inputs
* `rosco_controller_lib_filename::string`: path and name of inflow-wind dynamic library
 
# Outputs:
* `none`:
 
"""
function rosco_controller(status,time,dT,genRPM,Vinf_hub,GenTq,Azimuth,rosco_controller_lib_filename)
 
    global avrSWAP
 
    avrSWAP[1] = status # 1 --> Status flag set as follows: 0 if this is the first call, 1 for all subsequent time steps, -1 if this is the final call at the end of the simulation (-)
    avrSWAP[2] = time # 2   --> Current time    (sec)   [t in single precision]
    avrSWAP[3] = dT # 3 --> Communication interval  (sec)   
    # 4 --> Blade 1 pitch angle (rad)   [SrvD input]
    # 5 --> Below-rated pitch angle set-point   (rad)   [SrvD Ptch_SetPnt parameter]
    # 6 --> Minimum pitch angle (rad)   [SrvD Ptch_Min parameter]
    # 7 --> Maximum pitch angle (rad)   [SrvD Ptch_Max parameter]
    # 8 --> Minimum pitch rate (most negative value allowed)    (rad/s) [SrvD PtchRate_Min parameter]
    # 9 --> Maximum pitch rate                                  (rad/s) [SrvD PtchRate_Max parameter]
    # 10    --> 0 = pitch position actuator, 1 = pitch rate actuator    (-) [must be 0 for ServoDyn]
    # 11    --> Current demanded pitch angle    (rad)   [I am sending the previous value for blade 1 from the DLL, in the absence of any more information provided in Bladed documentation]
    # 12    --> Current demanded pitch rate     (rad/s) [always zero for ServoDyn]
    # 13    --> Demanded power  (W) [SrvD GenPwr_Dem parameter from input file]
    # 14    --> Measured shaft power    (W) [SrvD input]
    # 15    --> Measured electrical power output    (W) [SrvD calculation from previous step; should technically be a state]
    # 16    --> Optimal mode gain   (Nm/(rad/s)^2)  [if torque-speed table look-up not selected in input file, use SrvD Gain_OM parameter, otherwise use 0 (already overwritten in Init routine)]
    # 17    --> Minimum generator speed (rad/s) [SrvD GenSpd_MinOM parameter]
    # 18    --> Optimal mode maximum speed  (rad/s) [SrvD GenSpd_MaxOMp arameter]
    # 19    --> Demanded generator speed above rated    (rad/s) [SrvD GenSpd_Dem parameter]
    avrSWAP[20] = genRPM/60.0*2*pi # 20 --> Measured generator speed    (rad/s) [SrvD input]
    # 21    --> Measured rotor speed    (rad/s) [SrvD input]
    # 22    --> Demanded generator torque above rated   (Nm)    [SrvD GenTrq_Dem parameter from input file]
    avrSWAP[23] = GenTq # 23    --> Measured generator torque   (Nm)    [SrvD calculation from previous step; should technically be a state]
    # 24    --> Measured yaw error  (rad)   [SrvD input]
    # 25    --> Start of below-rated torque-speed look-up table     (Lookup table not in use)
    # 26    --> No. of points in torque-speed look-up table (-) [SrvD DLL_NumTrq parameter]:
    avrSWAP[27] = Vinf_hub # 27 --> Hub wind speed  (m/s)   [SrvD input]
    # 28    --> Pitch control: 0 = collective, 1 = individual   (-) [SrvD Ptch_Cntrl parameter]
    # 29    --> Yaw control: 0 = yaw rate control, 1 = yaw torque control   (-) [must be 0 for ServoDyn]
    # 30    --> Blade 1 root out-of-plane bending moment    (Nm)    [SrvD input]
    # 31    --> Blade 2 root out-of-plane bending moment    (Nm)    [SrvD input]
    # 32    --> Blade 3 root out-of-plane bending moment    (Nm)    [SrvD input]
    # 33    --> Blade 2 pitch angle (rad)   [SrvD input]
    # 34    --> Blade 3 pitch angle (rad)   [SrvD input]
    # 35    <-> Generator contactor (-) [GenState from previous call to DLL (initialized to 1)]
    # 36    <-> Shaft brake status  (-) [sent to DLL at the next call; anything other than 0 or 1 is an error]
    # 37    --> Nacelle yaw angle from North    (rad)   
    # 41    <-- demanded yaw actuator torque        [this output is ignored since record 29 is set to 0 by ServoDyn indicating yaw rate control]
    # avrSWAP[42] = PitchCom1  # Use the command angles of all blades if using individual pitch
    # avrSWAP[43] = PitchCom2  # Use the command angles of all blades if using individual pitch
    # avrSWAP[44] = PitchCom3  # Use the command angles of all blades if using individual pitch
    # 45    <-- Demanded pitch angle (Collective pitch) (rad)   
    # avrSWAP[46] = DemandedPitchRate  # Demanded pitch rate (Collective pitch)
    # 47    <-- Demanded generator torque   (Nm)    
    # 48    <-- Demanded nacelle yaw rate   (rad/s)
    avrSWAP[49] = 7 # 49    --> Maximum number of characters in the "MESSAGE" argument  (-) [size of ErrMsg argument plus 1 (we add one for the C NULL CHARACTER)]
    avrSWAP[50] = 100 # 50   --> Number of characters in the "INFILE"  argument  (-) [trimmed length of DLL_InFile parameter plus 1 (we add one for the C NULL CHARACTER)]
    avrSWAP[51] = 11 # 51   --> Number of characters in the "OUTNAME" argument  (-) [trimmed length of RootName parameter plus 1 (we add one for the C NULL CHARACTER)]
    # 53    --> Tower top fore-aft     acceleration (m/s^2) [SrvD input]
    # 54    --> Tower top side-to-side acceleration (m/s^2) [SrvD input]
    # 55    <-- UNUSED: Pitch override      [anything other than 0 is an error in ServoDyn]
    # 56    <-- UNUSED: Torque override     [anything other than 0 is an error in ServoDyn]
    avrSWAP[60] = Azimuth # 60    --> Rotor azimuth angle (rad)   [SrvD input]
    # 61    --> Number of blades    (-) [SrvD NumBl parameter]
    # 62    --> Maximum number of values which can be returned for logging  (-) [set to 300]
    # 63    <-- Number logging channels     
    # 64    --> Maximum number of characters which can be returned in "OUTNAME" (-) [set to 12601 (including the C NULL CHARACTER)]
    # 65    <-- Number of variables returned for logging        [anything greater than MaxLoggingChannels is an error]
    # 66    --> Start of Platform motion -- 1001        
    # 69    --> Blade 1 root in-plane bending moment    (Nm)    [SrvD input]
    # 70    --> Blade 2 root in-plane bending moment    (Nm)    [SrvD input]
    # 71    --> Blade 3 root in-plane bending moment    (Nm)    [SrvD input]
    # avrSWAP[72] = GeneratorStartResistance  # Generator start-up resistance
    # 73    --> Rotating hub My (GL co-ords)    (Nm)    [SrvD input]
    # 74    --> Rotating hub Mz (GL co-ords)    (Nm)    [SrvD input]
    # 75    --> Fixed    hub My (GL co-ords)    (Nm)    [SrvD input]
    # 76    --> Fixed    hub Mz (GL co-ords)    (Nm)    [SrvD input]
    # 77    --> Yaw bearing  My (GL co-ords)    (Nm)    [SrvD input]
    # 78    --> Yaw bearing  Mz (GL co-ords)    (Nm)    [SrvD input]
    # avrSWAP[79] = LoadsReq  # Request for loads: 0=none
    # avrSWAP[80] = VariableSlipStatus  # Variable slip current status
    # avrSWAP[81] = VariableSlipDemand  # Variable slip current demand
    # 82    --> Nacelle roll    acceleration    (rad/s^2)   [SrvD input] -- this is in the shaft (tilted) coordinate system, instead of the nacelle (nontilted) coordinate system
    # 83    --> Nacelle nodding acceleration    (rad/s^2)   [SrvD input]
    # 84    --> Nacelle yaw     acceleration    (rad/s^2)   [SrvD input] -- this is in the shaft (tilted) coordinate system, instead of the nacelle (nontilted) coordinate system
    # 95    --> Reserved        (SrvD customization: set to SrvD AirDens parameter)
    # 96    --> Reserved        (SrvD customization: set to SrvD AvgWindSpeed parameter)
    # 109   --> Shaft torque (=hub Mx for clockwise rotor)  (Nm)    [SrvD input]
    # 110   --> Thrust - Rotating low-speed shaft force x (GL co-ords)  (N) [SrvD input]
    # 111   --> Nonrotating low-speed shaft force y (GL co-ords)    (N) [SrvD input]
    # 112   --> Nonrotating low-speed shaft force z (GL co-ords)    (N) [SrvD input]
    # 117   --> Controller state        [always set to 0]
    # 120   <-- Airfoil command, blade 1        
    # 121   <-- Airfoil command, blade 2        
    # 122   <-- Airfoil command, blade 3        
    # 129   --> Maximum extent of the avrSWAP array: 3300       
 
    if status == 0
        # println("Attempting to access controller at: $rosco_controller_lib_filename")
        global rosco_controller_lib = Libdl.dlopen(rosco_controller_lib_filename) # Open the library explicitly.
        global sym_time_step = Libdl.dlsym(rosco_controller_lib, :DISCON)   # Get a symbol for the function to call.
    end
 
    # time_step(float avrSWAP[], int aviFAIL, char accINFILE[], char avcOUTNAME[], char avcMSG[])
    ccall(sym_time_step,Cint,
        (Ref{Cfloat}, # float avrSWAP[]
        Ptr{Cint}, # int aviFAIL
        Ptr{Cchar}, # char accINFILE[]
        Ptr{Cchar}, # char avcOUTNAME[]
        Ptr{Cchar}), # char avcMSG[]
        avrSWAP, #InputFileStrings_C
        aviFAIL,
        accINFILE,
        avcOUTNAME,
        avcMSG)
 
 
    genTorque = avrSWAP[47]# 47 <-- Demanded generator torque   (Nm)    
 
    return -genTorque # Return the low speed shaft torque applied to the rotor
 
end
 
# Convenience functions
rosco_controller_init(rosco_controller_lib_filename) = rosco_controller(0,0.0,0.0,0.0,0.0,0.0, 0.0, rosco_controller_lib_filename)
rosco_controller_calc(time,dT,genRPM,Vinf_hub,GenTq,Azimuth) = rosco_controller(1,time,dT,genRPM,Vinf_hub, GenTq, Azimuth, nothing)
rosco_controller_end() = rosco_controller(-1,0.0,0.0,0.0,0.0,0.0,0.0,nothing)
 
end