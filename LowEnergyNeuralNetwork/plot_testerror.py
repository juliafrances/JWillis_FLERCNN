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
parser.add_argument("--transformation", default=1, type=int,
                    dest='transformation',help="use to adjust the max abs factor for transformed data")
args = parser.parse_args()

transformation = args.transformation
input_file = args.input_file
save_folder_name = args.output_dir

save = True
bins = 100

f = h5py.File(input_file, "r")
Y_test_use = f["Y_test_use"][:]
Y_test_predicted = f["Y_predicted"][:]
reco_test = f["reco_test"][:]
#weights = f["weights_test"][:]
try:
    info = f["additional_info"][:]
except:
    info = None
f.close()
del f

pred_xerror = Y_test_predicted[:,3]
pred_yerror = Y_test_predicted[:,4]
pred_zerror = Y_test_predicted[:,5]
print(type(pred_xerror))
true_xerror = np.abs(Y_test_use[:,4]-Y_test_predicted[:,0])
true_yerror = np.abs(Y_test_use[:,5]-Y_test_predicted[:,1])
true_zerror = np.abs(Y_test_use[:,6]-Y_test_predicted[:,2])

cut_xmask = true_xerror < 100
cut_xpred = pred_xerror[cut_xmask]
cut_xtruth = true_xerror[cut_xmask]

cut_ymask = true_yerror < 100
cut_ypred = pred_yerror[cut_ymask]
cut_ytruth = true_yerror[cut_ymask]

cut_zmask = true_zerror < 100
cut_zpred = pred_zerror[cut_zmask]
cut_ztruth = true_zerror[cut_zmask]

true_error = [true_xerror,true_yerror,true_zerror]

cut_truth = [cut_xtruth,cut_ytruth,cut_ztruth]
cut_pred = [cut_xpred,cut_ypred,cut_zpred]

true_energy = Y_test_use[:,0]

pred_x = Y_test_predicted[:,0]
pred_y = Y_test_predicted[:,1]
pred_z = Y_test_predicted[:,2]

title = ["X", "Y", "Z", "X Error", "Y Error", "Z Error"]
minrange = [-600,-600,-1000, 1,0,0]
maxrange = [600,600,200,6,1,10]

for i in range(3,6):
    print(i)
    print("Plottng %s"%title[i])
    #Plot
    true_index = 4+i
    NN_index= i
    NN_pos_index = i-3
    true_pos_index = i+1
    maxabs_factor = transformation #1 if not transformed, 200 if transformed
    plot_name = title[i]
    plot_units = " "

    plot_2D_prediction(cut_truth[i-3], cut_pred[i-3],
                    save=True,savefolder=save_folder_name,variable = title[i],
                    units=plot_units)

    plot_2D_prediction(Y_test_predicted[:,NN_pos_index],Y_test_predicted[:,NN_index],
                    save=True,savefolder=save_folder_name,variable = title[i],
                    new_labels = ["Predicted Vertex Error","Reconstructed Vertex"],
                    new_units = [" ", "(m)"])

    plot_2D_prediction(Y_test_use[:,0],Y_test_predicted[:,NN_index],
                    save=True,savefolder=save_folder_name,variable=title[i],
                    new_labels=["Predicted Vertex Error", "True Energy"],
                    new_units = [" ","GeV"])
