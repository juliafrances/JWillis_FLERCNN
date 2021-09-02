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
parser.add_argument("--cuts", nargs=3, type=int, default = [150,-500,-200], dest="cuts",
                    help="set the cuts for the data in order of [maxr, minz, maxz]. Default is Reco cuts")
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
print("Cutting outside %d for R and %d, %d for z" % (cuts[0], cuts[1], cuts[2]))

save = True
bins = 100
maxabs_factor = transformation

print('Importing data...')

f = h5py.File(input_file, "r")
Y_test_use = f["Y_test_use"][:]
Y_test_predicted = f["Y_predicted"][:]
reco_test = f["reco_test"][:]
weights = f["weights_test"][:]
additional_info=f["additional_info"][:]
try:
    info = f["additional_info"][:]
except:
    info = None
f.close()
del f

fit_success=additional_info[:,9]
fit_success_retro=additional_info[:,3]

numu_files = 1518
nue_files = 602
if weights is not None:
    weights = np.array(weights[:,-2])
    #modify by number of files
    mask_numu = np.array(Y_test_use[:,9]) == 14
    mask_nue = np.array(Y_test_use[:,9]) == 12
    if sum(mask_numu) > 1:
        weights[mask_numu] = weights[mask_numu]/numu_files
    if sum(mask_nue) > 1:
        weights[mask_nue] = weights[mask_nue]/nue_files

#true_x = np.array(truth[:,4])
#true_y = np.array(truth[:,5])
#true_z = np.array(truth[:,6])
#reco_x = np.array(reco[:,4])
#reco_y = np.array(reco[:,5])
#reco_z = np.array(reco[:,6])

print('Calculating R from X & Y...')

x_origin = np.ones((len(Y_test_use[:,2])))*46.290000915527344
y_origin = np.ones((len(Y_test_use[:,3])))*-34.880001068115234
#true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
#reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )

#Calculating R from X & Y
R_test_use_nolim = np.sqrt((Y_test_use[:,4]-x_origin)**2+(Y_test_use[:,5]-y_origin)**2)
R_test_predicted_nolim = np.sqrt(((Y_test_predicted[:,2]*maxabs_factor)-x_origin)**2+((Y_test_predicted[:,3]*maxabs_factor)-y_origin)**2)
R_reco_test_nolim = np.sqrt((reco_test[:,4]-x_origin)**2+(reco_test[:,5]-y_origin)**2)

#Set Z to make cuts easier

Z_test_use_nolim = Y_test_use[:,6]
Z_test_predicted_nolim = Y_test_predicted[:,4]
Z_reco_test_nolim = reco_test[:,6]

#Set energy and zenith
Energy_test_use_nolim = Y_test_use[:,0]
Energy_test_predicted_nolim = Y_test_predicted[:,0]*100
Energy_reco_test_nolim = reco_test[:,0]

Zenith_test_use_nolim = Y_test_use[:,1]
Zenith_test_predicted_nolim = Y_test_predicted[:,1]
Zenith_reco_test_nolim = reco_test[:,1]

#Make the cuts

print('Creating masks...')

#Mask Predict on Predict
R_test_predictedmask = np.logical_and(R_test_predicted_nolim < cuts[0], fit_success)
Z_test_predictedmask = np.logical_and(np.logical_and(Z_test_predicted_nolim > cuts[1], Z_test_predicted_nolim<cuts[2]), fit_success)
#Mask True on True
R_test_usemask = np.logical_and(R_test_use_nolim < cuts[0], fit_success)
Z_test_usemask = np.logical_and(np.logical_and(Z_test_use_nolim > cuts[1], Z_test_use_nolim<cuts[2]), fit_success)
#Mask Reco on Reco
R_test_recomask = np.logical_and(np.logical_and(R_reco_test_nolim < cuts[0], fit_success), fit_success_retro)
Z_test_recomask = np.logical_and(np.logical_and(np.logical_and(Z_reco_test_nolim > cuts[1], Z_reco_test_nolim<cuts[2]),fit_success), fit_success_retro)

R_test_usepredicted = R_test_use_nolim[R_test_predictedmask] #Mask True on Predict
R_test_predicted = R_test_predicted_nolim[R_test_predictedmask]#Mask Predict on Predict
weights_Rpredicted = weights[R_test_predictedmask]

R_test_usereco = R_test_use_nolim[R_test_recomask] #Mask True on Reco
R_reco_test = R_reco_test_nolim[R_test_recomask]#Mask Reco on Reco
weights_Rreco = weights[R_test_recomask]

R_test_use = R_test_use_nolim[R_test_usemask] #Mask True on True
weights_Ruse = weights[R_test_usemask]

Z_test_usepredicted = Z_test_use_nolim[Z_test_predictedmask] #Mask True on Predict
Z_test_predicted = Z_test_predicted_nolim[Z_test_predictedmask] #Mask Predict on Predict
weights_Zpredicted = weights[Z_test_predictedmask]

Z_test_use = Z_test_use_nolim[Z_test_usemask] #Mask True on True
weights_Zuse = weights[Z_test_usemask]

Z_test_usereco = Z_test_use_nolim[Z_test_recomask] #Mask True on Reco
Z_reco_test = Z_reco_test_nolim[Z_test_recomask] #Mask Reco on Reco
weights_Zreco = weights[Z_test_recomask]

Energy_test_usepredictedR = Energy_test_use_nolim[R_test_predictedmask]
Energy_test_predictedR = Energy_test_predicted_nolim[R_test_predictedmask]

Energy_test_userecoR = Energy_test_use_nolim[R_test_recomask]
Energy_reco_testR = Energy_reco_test_nolim[R_test_recomask]

Energy_test_useR = Energy_test_use_nolim[R_test_usemask]


Energy_test_usepredictedZ = Energy_test_use_nolim[Z_test_predictedmask]
Energy_test_predictedZ = Energy_test_predicted_nolim[Z_test_predictedmask]

Energy_test_useZ = Energy_test_use_nolim[Z_test_usemask]

Energy_test_userecoZ = Energy_test_use_nolim[Z_test_recomask]
Energy_reco_testZ = Energy_reco_test_nolim[Z_test_recomask]

Zenith_test_usepredictedR = Zenith_test_use_nolim[R_test_predictedmask]
Zenith_test_predictedR = Zenith_test_predicted_nolim[R_test_predictedmask]

Zenith_test_userecoR = Zenith_test_use_nolim[R_test_recomask]
Zenith_reco_testR = Zenith_reco_test_nolim[R_test_recomask]

Zenith_test_useR = Zenith_test_use_nolim[R_test_usemask]


Zenith_test_usepredictedZ = Zenith_test_use_nolim[Z_test_predictedmask]
Zenith_test_predictedZ = Zenith_test_predicted_nolim[Z_test_predictedmask]

Zenith_test_useZ = Zenith_test_use_nolim[Z_test_usemask]

Zenith_test_userecoZ = Zenith_test_use_nolim[Z_test_recomask]
Zenith_reco_testZ = Zenith_reco_test_nolim[Z_test_recomask]

print("True cut R data elements:", np.shape(R_test_use), "Out of", np.shape(R_test_use_nolim))
print("CNN cut R data elements:", np.shape(R_test_predicted), "Out of", np.shape(R_test_predicted_nolim))
print("Reco cut R data elements:", np.shape(R_reco_test), "Out of", np.shape(R_reco_test_nolim))

print("Truth on Predict elements:", np.shape(R_test_usepredicted), "Truth on Reco elements:", np.shape(R_test_usereco))

print("True cut Z data elements:", np.shape(Z_test_use), "Out of", np.shape(Z_test_use_nolim))
print("CNN cut Z data elements:", np.shape(Z_test_predicted), "Out of", np.shape(Z_test_predicted_nolim))
print("Reco cut Z data elements:", np.shape(Z_reco_test), "Out of", np.shape(Z_reco_test_nolim))


plot_distributions(R_reco_test**2,log=False,label = 'Retro',
                                        save=save, savefolder=save_folder_name,
                                        weights=weights_Rreco,
                                        variable='Energy', units= 'GeV',
                                        bins=bins,
                                        title="Testing R^2 Retro Distribution for %s"%dist_title)
plot_distributions(R_reco_test**2,log=False,label = 'Retro',
                                        save=save, savefolder=save_folder_name,
                                        weights=weights_Rreco,
                                        variable='Zenith', units= '(m)',
                                        bins=bins,
                                        title="Testing R^2 Retro Distribution for %s"%dist_title)
