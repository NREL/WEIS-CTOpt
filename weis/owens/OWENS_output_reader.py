import os
import h5py
import numpy as np



def OWENSOutput(filepath, output_channels):
    
    read_file = h5py.File(filepath, "r")

    requested_output = {}

    for output_var in output_channels:
        try:
            requested_output[output_var] = np.array(read_file[output_var]).T
        except:
            print("The requested output {} is not avaiable from OWENS time series output".format(output_var))

    
    return requested_output

