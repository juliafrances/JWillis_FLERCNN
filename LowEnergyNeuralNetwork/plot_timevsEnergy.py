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
parser.add_argument("--cuts", nargs=3, type=int, default = None, dest="cuts",
                    help="set the max bounds for the plots of [maxr, minz, maxz]. Default is none.")
parser.add_argument("--axis_square", default=True,dest="axis_square", help="Cut Truth in 2D predictions to make axis square")
#parser.add_argument("--minr", default=-200, type=int,
#                        dest='minr',help="use to cut the data")
#parser.add_argument("--maxr", default=200, type=int,
#                        dest='maxr',help="use to cut the data")
#parser.add_argument("--minz", default=-200, type=int,
#                        dest='minz',help="use to cut the data")
#parser.add_argument("--maxz", default=200, type=int,
#                        dest='maxz',help="use to cut the data")
args = parser.parse_args()

input_file = args.input_file
save_folder_name = args.output_dir
transformation = args.transformation
cuts = args.cuts
axis_square=args.axis_square
#minr=args.minr
#maxr=args.maxr
#minz=args.minz
#maxz=args.maxz
print("Max abs factor (1 if not transformed)", transformation)

save = True
bins = 100
maxabs_factor = transformation

print('Importing data...')

f = h5py.File(input_file, "r")
Y_test_use = f["Y_test_use"][:]
Y_test_predicted = f["Y_predicted"][:]
#reco_test = f["reco_test"][:]
#weights = f["weights_test"][:]
#additional_info=f["additional_info"][:]
#try:
#    info = f["additional_info"][:]
#except:
#    info = None
#f.close()
#del f
#hit_8 = []
#for i in additional_info[:,9]:
#  if i ==1:
#    hit_8.append(True)
#  else:
#    hit_8.append(False)
#
#fit_success=additional_info[:,3]

numu_files = 1518
nue_files = 602
#if weights is not None:
#    weights = np.array(weights[:,-2])
    #modify by number of files
#    mask_numu = np.array(Y_test_use[:,9]) == 14
#    mask_nue = np.array(Y_test_use[:,9]) == 12
#    if sum(mask_numu) > 1:
#        weights[mask_numu] = weights[mask_numu]/numu_files
#    if sum(mask_nue) > 1:
#        weights[mask_nue] = weights[mask_nue]/nue_files

#true_x = np.array(truth[:,4])
#true_y = np.array(truth[:,5])
#true_z = np.array(truth[:,6])
#reco_x = np.array(reco[:,4])
#reco_y = np.array(reco[:,5])
#reco_z = np.array(reco[:,6])

#print('Calculating R from X & Y...')

x_origin = np.ones((len(Y_test_use[:,2])))*46.290000915527344
y_origin = np.ones((len(Y_test_use[:,3])))*-34.880001068115234
#true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
#reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )

#Set time
true_time = Y_test_use[:,3]
cnn_time = Y_test_predicted[:,0]

#Calculating R from X & Y
R_test_use_nolim = np.sqrt((Y_test_use[:,4]-x_origin)**2+(Y_test_use[:,5]-y_origin)**2)
#R_test_predicted_nolim = np.sqrt(((Y_test_predicted[:,2]*maxabs_factor)-x_origin)**2+((Y_test_predicted[:,3]*maxabs_factor)-y_origin)**2)
#R_reco_test_nolim = np.sqrt((reco_test[:,4]-x_origin)**2+(reco_test[:,5]-y_origin)**2)

#Set energy and zenith
Energy_test_use_nolim = Y_test_use[:,0]
#Energy_test_predicted_nolim = Y_test_predicted[:,0]*100
#Energy_reco_test_nolim = reco_test[:,0]

Zenith_test_use_nolim = Y_test_use[:,1]
#Zenith_test_predicted_nolim = Y_test_predicted[:,1]
#Zenith_reco_test_nolim = reco_test[:,1]

#Set Z to make cuts easier

Z_test_use_nolim = Y_test_use[:,6]
#Z_test_predicted_nolim = Y_test_predicted[:,4]
#Z_reco_test_nolim = reco_test[:,6]

minrange = [0,-500]
minrangefrac = [0,-500]
maxrangefrac = [300,-200]
maxrangevsPredict = [300,-200]
maxrange = [300,-200]
print("Max Energy:", Energy_test_use_nolim.max(), "Min Energy:", Energy_test_use_nolim.min())
#Plot R
plot_name = "Time"
plot_units = "(ns)"
i = 0
plot_bin_slices(true_time, cnn_time,\
                    use_fraction = False,\
                    bins=20,\
                    save=True,savefolder=save_folder_name,\
                    variable=plot_name,units=plot_units,\
                    energy_truth = Energy_test_use_nolim*100, xunits="GeV", max_val = 1000)
