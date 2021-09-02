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

title = ["X", "Y", "Z", "X Error", "Y Error", "Z Error"]
minrange = [-600,-600,-1000, 1,0,0]
minrangefrac = [-300,-300,-600,1,0,0]
maxrangefrac = [300,300,0,6,1,10]
maxrangevsPredict = [300,300,-200,6,1,10]
maxrange = [600,600,200,6,1,10]

for i in range(3):
  print('Plotting %s'%title[i], 'index %d'%i)
  #Plot
  true_index = 4+i
  NN_index= i
  recoindex= 4+i
  maxabs_factor = transformation #1 if not transformed, 200 if transformed
  if i < 4:
    plot_name = "%s Position"%title[i]
    plot_units = "(m)"
  else:
    plot_name = title[i]
    plot_units = None
  print(min(reco_test[:,recoindex]), max(reco_test[:,recoindex]))
  print(min(Y_test_use[:,true_index]), max(Y_test_use[:,true_index]))
  print(min(Y_test_predicted[:,NN_index]), max(Y_test_predicted[:,NN_index]))

  plot_2D_prediction(Y_test_use[:,true_index],\
                     Y_test_predicted[:,NN_index]*maxabs_factor,\
                     save,save_folder_name,bins=bins,\
                     variable=plot_name,units=plot_units, minval=minrange[i], maxval=maxrange[i], axis_square=True)
  plot_single_resolution(Y_test_use[:,true_index],\
                     Y_test_predicted[:,NN_index]*maxabs_factor,\
                     bins=bins,
                     save=save,savefolder=save_folder_name,\
                     variable=plot_name,units=plot_units,use_old_reco = True,old_reco = reco_test[:,recoindex], reco_name="Retro",\
                     minaxis=minrangefrac[i], maxaxis=maxrangefrac[i])
  plot_bin_slices(Y_test_use[:,true_index], Y_test_predicted[:,NN_index]*maxabs_factor,\
                      use_fraction = False,\
                      bins=20,min_val=minrangefrac[i],max_val=maxrangefrac[i],\
                      save=True,savefolder=save_folder_name,\
                      variable=plot_name,units=plot_units, old_reco=reco_test[:,recoindex], reco_name="Retro") #add that reco index
  plot_bin_slices(Y_test_use[:,true_index], Y_test_predicted[:,NN_index]*maxabs_factor,
                      use_fraction = False,\
                      bins=20,min_val=minrangefrac[i],max_val=maxrangevsPredict[i],\
                      save=True,savefolder=save_folder_name,\
                      variable=plot_name,units=plot_units, vs_predict=True,old_reco=reco_test[:,recoindex], reco_name='Retro')
  plot_2D_prediction(Y_test_use[:,true_index],\
                      reco_test[:,recoindex],\
                      save,save_folder_name,bins=bins,\
                      variable=plot_name,units=plot_units, reco_name="Retro")

for i in range(3,7):
  print("Plottng %s"%title[i])
  #Plot
  true_index = 4+i
  NN_index= i
  maxabs_factor = transformation #1 if not transformed, 200 if transformed
  plot_name = title[i]
  plot_units = None
  print(min(Y_test_use[:,true_index]), max(Y_test_use[:,true_index]))
  print(min(Y_test_predicted[:,NN_index]), max(Y_test_predicted[:,NN_index]))

  plot_2D_prediction(Y_test_use[:,true_index],\
                     Y_test_predicted[:,NN_index]*maxabs_factor,\
                     save,save_folder_name,bins=bins,\
                     variable=plot_name,units=plot_units, minval=minrange[i], maxval=maxrange[i], axis_square=True)
  plot_single_resolution(Y_test_use[:,true_index],\
                     Y_test_predicted[:,NN_index]*maxabs_factor,\
                     bins=bins,
                     save=save,savefolder=save_folder_name,\
                     variable=plot_name,units=plot_units,\
                     minaxis=minrangefrac[i], maxaxis=maxrangefrac[i])
  plot_bin_slices(Y_test_use[:,true_index], Y_test_predicted[:,NN_index]*maxabs_factor,\
                      use_fraction = False,\
                      bins=20,min_val=minrangefrac[i],max_val=maxrangefrac[i],\
                      save=True,savefolder=save_folder_name,\
                      variable=plot_name,units=plot_units) #add that reco index
  plot_bin_slices(Y_test_use[:,true_index], Y_test_predicted[:,NN_index]*maxabs_factor,
                      use_fraction = False,\
                      bins=20,min_val=minrangefrac[i],max_val=maxrangevsPredict[i],\
                      save=True,savefolder=save_folder_name,\
                      variable=plot_name,units=plot_units, vs_predict=True)
