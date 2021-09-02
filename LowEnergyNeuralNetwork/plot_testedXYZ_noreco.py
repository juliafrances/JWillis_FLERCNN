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
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
args = parser.parse_args()

input_file = args.input_file
save_folder_name = args.output_dir

save = True 
bins = 100

f = h5py.File(input_file, "r")
Y_test_use = f["Y_test_use"][:]
Y_test_predicted = f["Y_predicted"][:]
#reco_test = f["reco_test"][:]
#weights = f["weights_test"][:]
try:
    info = f["additional_info"][:]
except: 
    info = None
f.close()
del f

#true_x = np.array(truth[:,4])
#true_y = np.array(truth[:,5])
#true_z = np.array(truth[:,6])
#reco_x = np.array(reco[:,4])
#reco_y = np.array(reco[:,5])
#reco_z = np.array(reco[:,6])

#Vertex Position
#x_origin = np.ones((len(true_x)))*46.290000915527344
#y_origin = np.ones((len(true_y)))*-34.880001068115234
#true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
#reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )

title = ["X", "Y", "Z"]
minrange = [-600,-600,-1000]
maxrange = [600,600,200] 
for i in range(3):

  #Plot X
  true_index = 4+i
  NN_index= i
  recoindex= 4+i 
  maxabs_factor = 1 #1 if not transformed, 200 if transformed
  plot_name = "%s Position"%title[i] 
  plot_units = "(m)"
  plot_2D_prediction(Y_test_use[:,true_index]*maxabs_factor,\
                          Y_test_predicted[:,NN_index]*maxabs_factor,\
                          save,save_folder_name,bins=bins,\
                          variable=plot_name,units=plot_units)
  plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
                     Y_test_predicted[:,NN_index]*maxabs_factor,\
                     use_fraction=True,
                     minaxis=-2.,maxaxis=2,bins=bins,
                     save=save,savefolder=save_folder_name,\
                     variable=plot_name,units=plot_units)
  plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
                     Y_test_predicted[:,NN_index]*maxabs_factor,\
                     bins=bins,
                     save=save,savefolder=save_folder_name,\
                     variable=plot_name,units=plot_units)
  plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
                      use_fraction = False,\
                      bins=20,min_val=minrange[i],max_val=maxrange[i],\
                      save=True,savefolder=save_folder_name,\
                      variable=plot_name,units=plot_units)
  plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
                      use_fraction = True,\
                      bins=20,min_val=minrange[i],max_val=maxrange[i],\
                      save=True,savefolder=save_folder_name,\
                      variable=plot_name,units=plot_units)
#  plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
#                      use_fraction = False,\
#                      bins=20,min_val=minrange[i],max_val=maxrange[i],\
#                      save=True,savefolder=save_folder_name,\
#                      variable=plot_name,units=plot_units, vs_predict=True)