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
parser.add_argument("--switch_axis", default=False, dest="switch_axis", help="Switch the X and Y axes in the 2D predictions")
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
switch_axis=args.switch_axis
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


recomask = np.logical_and(hit_8, fit_success)

R_test_usepredicted = R_test_use_nolim[hit_8] #Mask True on Predict
R_test_predicted = R_test_predicted_nolim[hit_8]#Mask Predict on Predict
weights_Rpredicted = weights[hit_8]

R_test_usereco = R_test_use_nolim[recomask] #Mask True on Reco
R_reco_test = R_reco_test_nolim[recomask]#Mask Reco on Reco
weights_Rreco = weights[recomask]

R_test_use = R_test_use_nolim[hit_8] #Mask True on True
weights_Ruse = weights[hit_8]

Z_test_usepredicted = Z_test_use_nolim[hit_8] #Mask True on Predict
Z_test_predicted = Z_test_predicted_nolim[hit_8] #Mask Predict on Predict
weights_Zpredicted = weights[hit_8]

Z_test_use = Z_test_use_nolim[hit_8] #Mask True on True
weights_Zuse = weights[hit_8]

Z_test_usereco = Z_test_use_nolim[recomask] #Mask True on Reco
Z_reco_test = Z_reco_test_nolim[recomask] #Mask Reco on Reco
weights_Zreco = weights[recomask]

minrange = [0,-500]
minrangefrac = [0,-500]
maxrangefrac = [300,-200]
maxrangevsPredict = [300,-200]
maxrange = [300,-200]

#Plot R
plot_name = "R Position"
plot_units = "(m)"
print(min(R_reco_test), max(R_reco_test))
print(min(R_test_use), max(R_test_use))
print(min(R_test_predicted), max(R_test_predicted))

i = 0

if cuts == None:

   plot_2D_prediction(R_test_usepredicted,\
                      R_test_predicted,\
                      save,save_folder_name,bins=bins,\
                      variable=plot_name,units=plot_units,\
                      weights=weights_Rpredicted,
                      switch_axis=switch_axis)
   #plot_single_resolution(R_test_usepredicted,\
   #                   R_test_predicted,\
   #                   bins=bins,
   #                   save=save,savefolder=save_folder_name,\
   #                   variable=plot_name,units=plot_units,use_old_reco = True,\
   #                   old_reco = R_reco_test, reco_name="Retro",\
   #                   old_reco_truth=R_test_usereco,\
   #                   weights=weights_Rpredicted,\
   #                   old_reco_weights = weights_Rreco)
   #plot_bin_slices(R_test_usepredicted, R_test_predicted,\
   #                    use_fraction = False,\
   #                    bins=20,\
   #                    save=True,savefolder=save_folder_name,\
   #                    variable=plot_name,units=plot_units,\
   #                    old_reco=R_reco_test, reco_name="Retro", old_reco_truth=R_test_usereco,\
   #                    weights=weights_Rpredicted, old_reco_weights = weights_Rreco)
   #plot_bin_slices(R_test_usepredicted, R_test_predicted,
   #                    use_fraction = False,\
   #                    bins=20,\
   #                    save=True,savefolder=save_folder_name,\
   #                    variable=plot_name,units=plot_units, vs_predict=True,\
   #                    old_reco=R_reco_test, reco_name='Retro', old_reco_truth=R_test_usereco,\
   #                    weights = weights_Rpredicted, old_reco_weights = weights_Rreco)
   #plot_2D_prediction(R_test_usereco,\
   #                    R_reco_test,\
   #                    save,save_folder_name,bins=bins,\
   #                    variable=plot_name,units=plot_units, reco_name="Retro",\
   #                    weights=weights_Rreco)



   #Plot Z

   i = 1

   #true_index = 6
   #NN_index = 2
   #recoindex = 6



   #plot_name = "Z Position"
   #plot_units = "(m)"
   #print(min(Z_reco_test), max(Z_reco_test))
   #print(min(Z_test_use), max(Z_test_use))
   #print(min(Z_test_predicted), max(Z_test_predicted))
   #
   #plot_2D_prediction(Z_test_usepredicted,\
   #                   Z_test_predicted*maxabs_factor,\
   #                   save,save_folder_name,bins=bins,\
   #                   variable=plot_name,units=plot_units,
   #                   weights = weights_Zpredicted,minval=cuts[1], maxval=cuts[2],\
   #                   axis_square=axis_square)
   #plot_single_resolution(Z_test_usepredicted,\
   #                   Z_test_predicted*maxabs_factor,\
   #                   bins=bins,
   #                   save=save,savefolder=save_folder_name,\
   #                   variable=plot_name,units=plot_units,\
   #                   use_old_reco = True,old_reco = Z_reco_test,\
   #                   reco_name="Retro", old_reco_truth=Z_test_usereco,\
   #                   weights = weights_Zpredicted, old_reco_weights = weights_Zreco)
   #plot_bin_slices(Z_test_usepredicted, Z_test_predicted*maxabs_factor,\
   #                    use_fraction = False,\
   #                    bins=20,min_val=minrangefrac[i],max_val=maxrangefrac[i],\
   #                    save=True,savefolder=save_folder_name,\
   #                    variable=plot_name,units=plot_units,\
   #                    old_reco=Z_reco_test, reco_name="Retro",\
   #                    old_reco_truth=Z_test_usereco,\
   #                    weights = weights_Zpredicted, old_reco_weights = weights_Zreco)
   #plot_bin_slices(Z_test_usepredicted, Z_test_predicted*maxabs_factor,
   #                    use_fraction = False,\
   #                    bins=20,min_val=minrangefrac[i],max_val=maxrangevsPredict[i],\
   #                    save=True,savefolder=save_folder_name,\
   #                    variable=plot_name,units=plot_units,\
   #                    vs_predict=True,old_reco=Z_reco_test,\
   #                    reco_name='Retro', old_reco_truth=Z_test_usereco,\
   #                    weights = weights_Zpredicted, old_reco_weights = weights_Zreco)
   #plot_2D_prediction(Z_test_usereco,\
   #                    Z_reco_test,\
   #                    save,save_folder_name,bins=bins,\
   #                    variable=plot_name,units=plot_units, reco_name="Retro",\
   #                    weights = weights_Zreco,minval=cuts[1], maxval=cuts[2],\
   #                    axis_square=axis_square)
else:
   plot_2D_prediction(R_test_usepredicted,\
                      R_test_predicted,\
                      save,save_folder_name,bins=bins,\
                      variable=plot_name,units=plot_units, minval=0, maxval=cuts[0],\
                      weights=weights_Rpredicted,\
                      axis_square=axis_square, switch_axis=switch_axis)
   #plot_single_resolution(R_test_usepredicted,\
   #                   R_test_predicted,\
   #                   bins=bins,
   #                   save=save,savefolder=save_folder_name,\
   #                   variable=plot_name,units=plot_units,use_old_reco = True,\
   #                   old_reco = R_reco_test, reco_name="Retro",\
   #                   old_reco_truth=R_test_usereco,\
   #                   weights=weights_Rpredicted,\
   #                   old_reco_weights = weights_Rreco)
   #plot_bin_slices(R_test_usepredicted, R_test_predicted,\
   #                    use_fraction = False,\
   #                    bins=20,min_val=0,max_val=cuts[0],\
   #                    save=True,savefolder=save_folder_name,\
   #                    variable=plot_name,units=plot_units,\
   #                    old_reco=R_reco_test, reco_name="Retro", old_reco_truth=R_test_usereco,\
   #                    weights=weights_Rpredicted, old_reco_weights = weights_Rreco)
   #plot_bin_slices(R_test_usepredicted, R_test_predicted,
   #                    use_fraction = False,\
   #                    bins=20,min_val=0,max_val=cuts[0],\
   #                    save=True,savefolder=save_folder_name,\
   #                    variable=plot_name,units=plot_units, vs_predict=True,\
   #                    old_reco=R_reco_test, reco_name='Retro', old_reco_truth=R_test_usereco,\
   #                    weights = weights_Rpredicted, old_reco_weights = weights_Rreco)
   #plot_2D_prediction(R_test_usereco,\
   #                    R_reco_test,\
   #                    save,save_folder_name,bins=bins,\
   #                    variable=plot_name,units=plot_units, reco_name="Retro",\
   #                    weights=weights_Rreco,minval=0, maxval=cuts[0],\
   #                    axis_square=axis_square)



   #Plot Z

   i = 1

   #true_index = 6
   #NN_index = 2
   #recoindex = 6



   #plot_name = "Z Position"
   #plot_units = "(m)"
   #print(min(Z_reco_test), max(Z_reco_test))
   #print(min(Z_test_use), max(Z_test_use))
   #print(min(Z_test_predicted), max(Z_test_predicted))
   #
   #plot_2D_prediction(Z_test_usepredicted,\
   #                   Z_test_predicted*maxabs_factor,\
   #                   save,save_folder_name,bins=bins,\
   #                   variable=plot_name,units=plot_units,
   #                   weights = weights_Zpredicted,minval=cuts[1], maxval=cuts[2],\
   #                   axis_square=axis_square)
   #plot_single_resolution(Z_test_usepredicted,\
   #                   Z_test_predicted*maxabs_factor,\
   #                   bins=bins,
   #                   save=save,savefolder=save_folder_name,\
   #                   variable=plot_name,units=plot_units,\
   #                   use_old_reco = True,old_reco = Z_reco_test,\
   #                   reco_name="Retro", old_reco_truth=Z_test_usereco,\
   #                   weights = weights_Zpredicted, old_reco_weights = weights_Zreco)
   #plot_bin_slices(Z_test_usepredicted, Z_test_predicted*maxabs_factor,\
   #                    use_fraction = False,\
   #                    bins=20,min_val=minrangefrac[i],max_val=maxrangefrac[i],\
   #                    save=True,savefolder=save_folder_name,\
   #                    variable=plot_name,units=plot_units,\
   #                    old_reco=Z_reco_test, reco_name="Retro",\
   #                    old_reco_truth=Z_test_usereco,\
   #                    weights = weights_Zpredicted, old_reco_weights = weights_Zreco)
   #plot_bin_slices(Z_test_usepredicted, Z_test_predicted*maxabs_factor,
   #                    use_fraction = False,\
   #                    bins=20,min_val=minrangefrac[i],max_val=maxrangevsPredict[i],\
   #                    save=True,savefolder=save_folder_name,\
   #                    variable=plot_name,units=plot_units,\
   #                    vs_predict=True,old_reco=Z_reco_test,\
   #                    reco_name='Retro', old_reco_truth=Z_test_usereco,\
   #                    weights = weights_Zpredicted, old_reco_weights = weights_Zreco)
   #plot_2D_prediction(Z_test_usereco,\
   #                    Z_reco_test,\
   #                    save,save_folder_name,bins=bins,\
   #                    variable=plot_name,units=plot_units, reco_name="Retro",\
   #                    weights = weights_Zreco,minval=cuts[1], maxval=cuts[2],\
   #                    axis_square=axis_square)


#
#
#title = ["X", "Y", "Z"]
#minrange = [-600,-1000]
#minrangefrac = [-300,-600]
#maxrangefrac = [300,0]
#maxrangevsPredict = [300,-200]
#maxrange = [600,200]
#for i in range(2):

#  true_index = 4+i
#  NN_index= i
#  recoindex= 4+i
#  maxabs_factor = transformation #1 if not transformed, 200 if transformed
#  plot_name = "%s Position"%title[i]
#  plot_units = "(m)"
#  print(min(reco_test[:,recoindex]), max(reco_test[:,recoindex]))
#  print(min(Y_test_use[:,true_index]), max(Y_test_use[:,true_index]))
#  print(min(Y_test_predicted[:,NN_index]), max(Y_test_predicted[:,NN_index]))
#
#  plot_2D_prediction(Y_test_use[:,true_index],\
#                     Y_test_predicted[:,NN_index]*maxabs_factor,\
#                     save,save_folder_name,bins=bins,\
#                     variable=plot_name,units=plot_units)
#  plot_single_resolution(Y_test_use[:,true_index],\
#                     Y_test_predicted[:,NN_index]*maxabs_factor,\
#                     bins=bins,
#                     save=save,savefolder=save_folder_name,\
#                     variable=plot_name,units=plot_units,use_old_reco = True,old_reco = reco_test[:,recoindex], reco_name="Retro")
#  plot_bin_slices(Y_test_use[:,true_index], Y_test_predicted[:,NN_index]*maxabs_factor,\
#                      use_fraction = False,\
#                      bins=20,min_val=minrangefrac[i],max_val=maxrangefrac[i],\
#                      save=True,savefolder=save_folder_name,\
#                      variable=plot_name,units=plot_units, old_reco=reco_test[:,recoindex], reco_name="Retro") #add that reco index
#  plot_bin_slices(Y_test_use[:,true_index], Y_test_predicted[:,NN_index]*maxabs_factor,
#                      use_fraction = False,\
#                      bins=20,min_val=minrangefrac[i],max_val=maxrangevsPredict[i],\
#                      save=True,savefolder=save_folder_name,\
#                      variable=plot_name,units=plot_units, vs_predict=True,old_reco=reco_test[:,recoindex], reco_name='Retro')
#  plot_2D_prediction(Y_test_use[:,true_index],\
#                      reco_test[:,recoindex],\
#                      save,save_folder_name,bins=bins,\
#                      variable=plot_name,units=plot_units, reco_name="Retro")
