#########################
# Version of CNN on 12 May 2020
# 
# Evaluates net for given model and plots
# Takes in ONE file to Test on, can compare to old reco
# Runs Energy, Zenith, Track length (1 variable energy or zenith, 2 = energy then zenith, 3 = EZT)
#   Inputs:
#       -i input_file:  name of ONE file 
#       -d path:        path to input files
#       -o ouput_dir:   path to output_plots directory
#       -n name:        name for folder in output_plots that has the model you want to load
#       -e epochs:      epoch number of the model you want to load
#       --variables:    Number of variables to train for (1 = energy or zenith, 2 = EZ, 3 = EZT)
#       --first_variable: Which variable to train for, energy or zenith (for num_var = 1 only)
#       --compare_reco: boolean flag, true means you want to compare to a old reco (pegleg, retro, etc.)
#       -t test:        Name of reco to compare against, with "oscnext" used for no reco to compare with
####################################

import numpy as np
import h5py
import time
import os, sys
import random
from collections import OrderedDict
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default=None,
                    dest="input_file", help="names for test only input file")
parser.add_argument("-d", "--path",type=str,default='/data/icecube/jmicallef/processed_CNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output_dir",type=str,default='/home/users/jmicallef/LowEnergyNeuralNetwork/',
                    dest="output_dir", help="path to output_plots directory, do not end in /")
parser.add_argument("-n", "--name",type=str,default=None,
                    dest="name", help="name for output directory and where model file located")
parser.add_argument("-e","--epoch", type=int,default=None,
                    dest="epoch", help="which model number (number of epochs) to load in")
parser.add_argument("--variables", type=int,default=1,
                    dest="output_variables", help="1 for [energy], 2 for [energy, zenith], 3 for [energy, zenith, track]")
parser.add_argument("--first_variable", type=str,default="energy",
                    dest="first_variable", help = "name for first variable (energy, zenith only two supported)")
parser.add_argument("--compare_reco", default=False,action='store_true',
                        dest='compare_reco',help="use flag to compare to old reco vs. NN")
parser.add_argument("-t","--test", type=str,default="oscnext",
                        dest='test',help="name of reco")
parser.add_argument("--mask_zenith", default=False,action='store_true',
                        dest='mask_zenith',help="mask zenith for up and down going")
parser.add_argument("--z_values", type=str,default=None,
                        dest='z_values',help="Options are gt0 or lt0")
parser.add_argument("--emax",type=float,default=100.,
                        dest='emax',help="Max energy for use for plotting")
parser.add_argument("--efactor",type=float,default=100.,
                        dest='efactor',help="ENERGY FACTOR TO MULTIPLY BY!")
args = parser.parse_args()

test_file = args.path + args.input_file
output_variables = args.output_variables
filename = args.name
epoch = args.epoch
compare_reco = args.compare_reco
print("Comparing reco?", compare_reco)

dropout = 0.2
learning_rate = 1e-3
DC_drop_value = dropout
IC_drop_value =dropout
connected_drop_value = dropout
min_energy = 5
max_energy = args.emax
energy_factor = args.efactor

mask_zenith = args.mask_zenith
z_values = args.z_values

save = True
save_folder_name = "%soutput_plots/%s/"%(args.output_dir,filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
load_model_name = "%s%s_%iepochs_model.hdf5"%(save_folder_name,filename,args.epoch) 
use_old_weights = True

if args.first_variable == "Zenith" or args.first_variable == "zenith" or args.first_variable == "Z" or args.first_variable == "z":
    first_var = "zenith"
    first_var_index = 1
    print("Assuming Zenith is the only variable to test for")
    assert output_variables==1,"DOES NOT SUPPORT ZENITH FIRST + additional variables"
elif args.first_variable == "energy" or args.first_variable == "energy" or args.first_variable == "e" or args.first_variable == "E":
    first_var = "energy"
    first_var_index = 0
    print("testing with energy as the first index")
else:
    first_var = "energy"
    first_var_index = 0
    print("only supports energy and zenith right now! Please choose one of those. Defaulting to energy")
    print("testing with energy as the first index")

reco_name = args.test
save_folder_name += "/%s_%sepochs/"%(reco_name.replace(" ",""),epoch)
if os.path.isdir(save_folder_name) != True:
    os.mkdir(save_folder_name)
    
#Load in test data
print("Testing on %s"%test_file)
f = h5py.File(test_file, 'r')
Y_test_use = f['Y_test'][:]
X_test_DC_use = f['X_test_DC'][:]
X_test_IC_use = f['X_test_IC'][:]
if compare_reco:
    reco_test_use = f['reco_test'][:]
try:
    weights = f["weights_test"][:]
except:
    weights = None
    print("File does not have weights, not using...")
    pass
f.close
del f
print(X_test_DC_use.shape,X_test_IC_use.shape,Y_test_use.shape)

#mask_energy_train = np.logical_and(np.array(Y_test_use[:,0])>min_energy/max_energy,np.array(Y_test_use[:,0])<1.0)
#Y_test_use = np.array(Y_test_use)[mask_energy_train]
#X_test_DC_use = np.array(X_test_DC_use)[mask_energy_train]
#X_test_IC_use = np.array(X_test_IC_use)[mask_energy_train]
#if compare_reco:
#    reco_test_use = np.array(reco_test_use)[mask_energy_train]
if compare_reco:
    print("TRANSFORMING ZENITH TO COS(ZENITH)")
    reco_test_use[:,1] = np.cos(reco_test_use[:,1])
if mask_zenith:
    print("MANUALLY GETTING RID OF HALF THE EVENTS (UPGOING/DOWNGOING ONLY)")
    if z_values == "gt0":
        maxvals = [max_energy, 1., 0.]
        minvals = [min_energy, 0., 0.]
        mask_zenith = np.array(Y_test_use[:,1])>0.0
    if z_values == "lt0":
        maxvals = [max_energy, 0., 0.]
        minvals = [min_energy, -1., 0.]
        mask_zenith = np.array(Y_test_use[:,1])<0.0
    Y_test_use = Y_test_use[mask_zenith]
    X_test_DC_use = X_test_DC_use[mask_zenith]
    X_test_IC_use = X_test_IC_use[mask_zenith]
    if compare_reco:
        reco_test_use = reco_test_use[mask_zenith]

#Make network and load model
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from cnn_model import make_network
model_DC = make_network(X_test_DC_use,X_test_IC_use,output_variables,DC_drop_value,IC_drop_value,connected_drop_value)
model_DC.load_weights(load_model_name)
print("Loading model %s"%load_model_name)

# WRITE OWN LOSS FOR MORE THAN ONE REGRESSION OUTPUT
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_percentage_error

if first_var == "zenith":
    def ZenithLoss(y_truth,y_predicted):
        #return logcosh(y_truth[:,1],y_predicted[:,1])
        return mean_squared_error(y_truth[:,1],y_predicted[:,0])

    def CustomLoss(y_truth,y_predicted):
            zenith_loss = ZenithLoss(y_truth,y_predicted)
            return zenith_loss

    model_DC.compile(loss=ZenithLoss,
                optimizer=Adam(lr=learning_rate),
                metrics=[ZenithLoss])
    
    print("zenith first")


else: 

  def TrackLoss(y_truth,y_predicted):
      return mean_squared_error(y_truth[:,2],y_predicted[:,2])

  def XLoss(y_truth,y_predicted):
      return mean_squared_error(y_truth[:,4],y_predicted[:,0])

  def YLoss(y_truth,y_predicted):
      return mean_squared_error(y_truth[:,5],y_predicted[:,1]) #running with three variables at once so Y and Z loss are at y_predicted[:,1] and [:,2] respectively 
    
  def ZLoss(y_truth,y_predicted):
      return mean_squared_error(y_truth[:,6],y_predicted[:,2]) #^^^
    
  if output_variables == 3:
      def CustomLoss(y_truth, y_predicted):
        x = XLoss(y_truth,y_predicted)
        y = YLoss(y_truth,y_predicted)
        z = ZLoss(y_truth,y_predicted)
        return x + y + z
        
        model_DC.compile(loss=CustomLoss,
                  optimizer=Adam(lr=learning_rate),
                  metrics=[XLoss,YLoss,ZLoss])

#    def EnergyLoss(y_truth,y_predicted):
#        return mean_absolute_percentage_error(y_truth[:,0],y_predicted[:,0])
#
#    def ZenithLoss(y_truth,y_predicted):
#        return mean_squared_error(y_truth[:,1],y_predicted[:,1])
#
#    def TrackLoss(y_truth,y_predicted):
#        return mean_squared_logarithmic_error(y_truth[:,2],y_predicted[:,2])
#
#    if output_variables == 3:
#        def CustomLoss(y_truth,y_predicted):
#            energy_loss = EnergyLoss(y_truth,y_predicted)
#            zenith_loss = ZenithLoss(y_truth,y_predicted)
#            track_loss = TrackLoss(y_truth,y_predicted)
#            return energy_loss + zenith_loss + track_loss
#
#        model_DC.compile(loss=CustomLoss,
#                  optimizer=Adam(lr=learning_rate),
#                  metrics=[EnergyLoss,ZenithLoss,TrackLoss])

  elif output_variables == 2:
        def CustomLoss(y_truth,y_predicted):
          energy_loss = EnergyLoss(y_truth,y_predicted)
          zenith_loss = ZenithLoss(y_truth,y_predicted)
          return energy_loss + zenith_loss

        model_DC.compile(loss=CustomLoss,
                  optimizer=Adam(lr=learning_rate),
                  metrics=[EnergyLoss,ZenithLoss])
  else:
        def CustomLoss(y_truth,y_predicted):
            energy_loss = EnergyLoss(y_truth,y_predicted)
            return energy_loss

        model_DC.compile(loss=EnergyLoss,
                    optimizer=Adam(lr=learning_rate),
                    metrics=[EnergyLoss])

# Run prediction
#Y_test_compare = Y_test_use[:,first_var_index]
#score = model_DC.evaluate([X_test_DC_use,X_test_IC_use], Y_test_compare, batch_size=256)
#print("Evaluate:",score)
t0 = time.time()
Y_test_predicted = model_DC.predict([X_test_DC_use,X_test_IC_use])
t1 = time.time()
print("This took me %f seconds for %i events"%(((t1-t0)),Y_test_predicted.shape[0]))
#print(X_test_DC_use.shape,X_test_IC_use.shape,Y_test_predicted.shape,Y_test_use.shape)

print("Saving output file: %s/prediction_values.hdf5"%save_folder_name)
f = h5py.File("%s/prediction_values.hdf5"%save_folder_name, "w")
f.create_dataset("Y_test_use", data=Y_test_use)
f.create_dataset("Y_predicted", data=Y_test_predicted)
if compare_reco:
    f.create_dataset("reco_test", data=reco_test_use)
if weights is not None:
    f.create_dataset("weights_test", data=weights)
f.close()

### MAKE THE PLOTS ###
from PlottingFunctions import plot_single_resolution
from PlottingFunctions import plot_2D_prediction
from PlottingFunctions import plot_2D_prediction_fraction
from PlottingFunctions import plot_bin_slices
from PlottingFunctions import plot_distributions
from PlottingFunctions import plot_length_energy

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
#title = ["X", "Y", "Z"]
#minrange = [-3,-3,-5]
#maxrange = [3,3,1] 
#for i in range(3):

  #Plot X
#  true_index = 4+i
#  NN_index= i
#  recoindex= 4+i 
#  maxabs_factor = 200 #1 if not transformed, 200 if transformed
#  plot_name = "%s Position"%title[i] 
#  plot_units = "(m)"
#  plot_2D_prediction(Y_test_use[:,true_index]*maxabs_factor,\
#    Y_test_predicted[:,NN_index]*maxabs_factor,\
#    save,save_folder_name,bins=bins,\
#    variable=plot_name,units=plot_units)
#  plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
#                     Y_test_predicted[:,NN_index]*maxabs_factor,\
#                     use_fraction=True,
#                     minaxis=-2.,maxaxis=2,bins=bins,
#                     save=save,savefolder=save_folder_name,\
#                     variable=plot_name,units=plot_units)
#  plot_single_resolution(Y_test_use[:,true_index]*maxabs_factor,\
#                     Y_test_predicted[:,NN_index]*maxabs_factor,\
#                     bins=bins,
#                     save=save,savefolder=save_folder_name,\
#                     variable=plot_name,units=plot_units)
#  plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
#                      use_fraction = False,\
#                      bins=20,min_val=minrange[i],max_val=maxrange[i],\
#                      save=True,savefolder=save_folder_name,\
#                      variable=plot_name,units=plot_units) 
#  plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
#                      use_fraction = True,\
#                      bins=20,min_val=minrange[i],max_val=maxrange[i],\
#                      save=True,savefolder=save_folder_name,\
#                      variable=plot_name,units=plot_units)
#  plot_bin_slices(Y_test_use[:,true_index]*maxabs_factor, Y_test_predicted[:,NN_index]*maxabs_factor,\
#                      use_fraction = False,\
#                      bins=20,min_val=minrange[i],max_val=maxrange[i],\
#                      save=True,savefolder=save_folder_name,\
#                      variable=plot_name,units=plot_units, vs_predict=True)