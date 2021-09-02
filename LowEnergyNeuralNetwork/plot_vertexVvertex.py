import h5py
import argparse
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
from PlottingFunctions import plot_2D_prediction, plot_single_resolution, plot_bin_slices
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("--input2", type=str, default=None,
                    dest="input2_file", help="path and name of the second input file")
parser.add_argument("--input1_name", type=str, default=None,
                    dest = "input1_name", help="Name of first input")
parser.add_argument("--input1_index", type=int, default=None,
                    dest = "input1_index", help="Index of first input")
parser.add_argument("--input2_name", type=str, default=None,
                    dest = "input2_name", help="Name of second input")
parser.add_argument("--input2_index", type=int, default=None,
                    dest = "input2_index", help="Index of second input")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--transformation", default=1, type=int,
                    dest='transformation',help="use to adjust the max abs factor for transformed data")
parser.add_argument("--variable", type=str, default=None,
                    dest = "variable", help = "Name of variable being compared")
args = parser.parse_args()

transformation = args.transformation
input_file = args.input_file
input2_file = args.input2_file
input1_name = args.input1_name
input2_name = args.input2_name
input1_index= args.input1_index
input2_index = args.input2_index
save_folder_name = args.output_dir
variable = args.variable

save = True
bins = 100


f = h5py.File(input2_file, "r")
Y_test_use2 = f["Y_test_use"][:]
Y_test_predicted2 = f["Y_predicted"][:]
reco_test2 = f["reco_test"][:]
#weights = f["weights_test"][:]
try:
    info = f["additional_info"][:]
except:
    info = None
f.close()
del f

f = h5py.File(input_file, "r")
Y_test_use1 = f["Y_test_use"][:]
Y_test_predicted1 = f["Y_predicted"][:]
reco_test1 = f["reco_test"][:]
#weights = f["weights_test"][:]
try:
    info = f["additional_info"][:]
except:
    info = None
f.close()
del f

plot_2D_prediction(Y_test_use1[:,input1_index], Y_test_use2[:,input2_index],
                    save = True, savefolder = save_folder_name, bins=bins,
                    variable = variable, units = None)
plot_single_resolution(Y_test_use1[:,input1_index], Y_test_use2[:,input2_index],
                    save = True, savefolder = save_folder_name, bins=bins,
                    variable=variable,units = None)
