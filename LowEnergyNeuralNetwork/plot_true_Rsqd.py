
import h5py
import argparse
import os, sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

from PlottingFunctions import plot_distributions

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--transformation", default=1, type=int,
                        dest='transformation',help="use to adjust the max abs factor for transformed data")
parser.add_argument("--cuts", nargs=3, type=int, default = None, dest="cuts",
                    help="set the cuts for the data in order of [maxr, minz, maxz]. Default is Reco cuts")
parser.add_argument("--axis_square", default=True,dest="axis_square", help="Cut Truth in 2D predictions to make axis square")
parser.add_argument("--max_val", default=None, type=int, dest="max_val", help="Set a limit for the x-axis")#parser.add_argument("--minr", default=-200, type=int,
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
max_val=args.max_val
axis_square=args.axis_square
#minr=args.minr
#maxr=args.maxr
#minz=args.minz
#maxz=args.maxz
print("Max abs factor (1 if not transformed)", transformation)
if cuts ==None:
    print("No cuts being made")
else:
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

hit_8 = []
for i in additional_info[:,9]:
  if i ==1:
    hit_8.append(True)
  else:
    hit_8.append(False)

fit_success=additional_info[:,3]

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

#Make the cuts
print('Creating masks...')

#Mask True with hits8 and fit success
if cuts == None:
    recomask = np.logical_and(hit_8, fit_success)

    R_test_usepredicted = R_test_use_nolim[hit_8] #Mask True on Predict
    R_test_predicted = R_test_predicted_nolim[hit_8]
    weights_Rpredicted = weights[hit_8]

    R_test_usereco = R_test_use_nolim[recomask] #Mask True on Reco
    R_reco_test = R_reco_test_nolim[recomask]
    weights_Rreco = weights[recomask]


    Z_test_usepredicted = Z_test_use_nolim[hit_8] #Mask True on Predict
    weights_Zpredicted = weights[hit_8]

    Z_test_usereco = Z_test_use_nolim[recomask] #Mask True on Reco
    weights_Zreco = weights[recomask]
else:
    R_test_usemask = np.logical_and(R_test_use_nolim < cuts[0], hit_8)
    Z_test_usemask = np.logical_and(np.logical_and(Z_test_use_nolim > cuts[1], Z_test_use_nolim<cuts[2]), hit_8)

    R_test_usemask2 = np.logical_and(np.logical_and(R_test_use_nolim < cuts[0], hit_8), fit_success)
    Z_test_usemask2 = np.logical_and(np.logical_and(np.logical_and(Z_test_use_nolim > cuts[1], Z_test_use_nolim<cuts[2]), hit_8), fit_success)


    R_test_usepredicted = R_test_use_nolim[R_test_usemask] #Mask True on Predict
    R_test_predicted = R_test_predicted_nolim[R_test_usemask]
    weights_Rpredicted = weights[R_test_usemask]

    R_test_usereco = R_test_use_nolim[R_test_usemask2] #Mask True on Reco
    R_reco_test = R_reco_test_nolim[R_test_usemask2]
    weights_Rreco = weights[R_test_usemask2]


    Z_test_usepredicted = Z_test_use_nolim[Z_test_usemask] #Mask True on Predict
    weights_Zpredicted = weights[Z_test_usemask]

    Z_test_usereco = Z_test_use_nolim[Z_test_usemask2] #Mask True on Reco
    weights_Zreco = weights[Z_test_usemask2]

dist_title='NuMu NuE'

if max_val == None:
    plot_distributions(R_reco_test**2,log=False,label = 'Retro',
                                        save=save, savefolder=save_folder_name,
                                        weights=weights_Rreco,
                                        variable='R^2', units= '(m)',
                                        bins=bins,
                                        title="Testing R^2 Retro Distribution for %s"%dist_title)
    plot_distributions(R_test_predicted**2,log=False,label = 'CNN',
                                        save=save, savefolder=save_folder_name,
                                        weights=weights_Rpredicted,
                                        variable='R^2', units= '(m)',
                                        bins=bins,
                                        title="Testing R^2 CNN Distribution for %s"%dist_title)
    plot_distributions(R_test_usepredicted**2,log=False,label = 'Truth',
                                        save=save, savefolder=save_folder_name,
                                        weights=weights_Rpredicted,
                                        variable='R^2', units= '(m)',
                                        bins=bins,
                                        title="Testing R^2 Truth Distribution for %s"%dist_title)
else:
    plot_distributions(R_reco_test**2,log=False,label = 'Retro', minval=0, maxval=max_val,
                                        save=save, savefolder=save_folder_name,
                                        weights=weights_Rreco,
                                        variable='R^2', units= '(m)',
                                        bins=bins,
                                        title="Testing R^2 Retro Distribution for %s"%dist_title)
    plot_distributions(R_test_predicted**2,log=False,label = 'CNN',minval=0, maxval=max_val,
                                        save=save, savefolder=save_folder_name,
                                        weights=weights_Rpredicted,
                                        variable='R^2', units= '(m)',
                                        bins=bins,
                                        title="Testing R^2 CNN Distribution for %s"%dist_title)
    plot_distributions(R_test_usepredicted**2,log=False,label = 'Truth',minval=0, maxval=max_val,
                                        save=save, savefolder=save_folder_name,
                                        weights=weights_Rpredicted,
                                        variable='R^2', units= '(m)',
                                        bins=bins,
                                        title="Testing R^2 Truth Distribution for %s"%dist_title)

#plot_distributions(Z_test_usereco,log=False,
#                                    save=save, savefolder=save_folder_name,
#                                    weights=weights_Zreco,reco_name = 'Retro',
#                                    variable='Z', units= '(m)',
#                                    bins=bins,
#                                    title="Testing Z Distribution for %s"%dist_title)
