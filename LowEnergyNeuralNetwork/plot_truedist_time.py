import h5py
import argparse
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("--input2",type=str,default=None,
                    dest="input_file2", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
args = parser.parse_args()

input_file = args.input_file
input_file2 = args.input_file2
save_folder_name = args.output_dir

f = h5py.File(input_file, "r")
truth = f["Y_test_use"][:]
predict = f["Y_predicted"][:]
#reco = f["reco_test"][:]
weights = None #f["weights_test"][:]
try:
    info = f["additional_info"][:]
except:
    info = None
f.close()
del f

cnn_time = np.array(predict[:,0])*100 #change indices for x, y, and z, and get rid of *100
true_time = np.array(truth[:,3]) #change indices for x, y, and z

#hits8 = info[:,9]


#Plot
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_2D_prediction

save=True
if save ==True:
    print("Saving to %s"%save_folder_name)

plot_name = "Time"
plot_units = "(ns)"
maxabs_factors = 100.

save_base_name = save_folder_name
bins = 100
syst_bin = 20

dist_title = "NuMu CC"

plot_distributions(cnn_time,log=False,
                   save=save, savefolder=save_folder_name,
                   variable='Time', units= '(ns)',
                   bins=bins,
                   title="Predicted Time Distribution for %s"%dist_title)

plot_distributions(true_time,log=False,
                   save=save, savefolder=save_folder_name,
                   variable='Time', units= '(ns)',
                   bins=bins,
                   title="True Time Testing Distribution for %s"%dist_title)
