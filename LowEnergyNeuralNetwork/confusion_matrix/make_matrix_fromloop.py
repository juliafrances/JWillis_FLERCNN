import h5py
import argparse
import os, sys
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/ufs18/home-049/willis51/LowEnergyNeuralNetwork/confusion_matrix/matrices/loops/',
                    dest="output_dir", help="path of ouput file")
parser.add_argument("--cuts", nargs=3, default = [150,-500,-200], dest="cuts", 
                    help="set the cuts for the data in order of [maxr, minz, maxz]. Default is Reco cuts")
parser.add_argument("--transformation", type=int, default=1, dest="transformation")
parser.add_argument
args = parser.parse_args()

input_file = args.input_file
save_folder_name = args.output_dir
cuts = args.cuts
maxabs_factor = args.transformation

print("Cutting outside %d for R and %d, %d for z" % (cuts[0], cuts[1], cuts[2]))

save = True 

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

print('Calculating true vertex...')

#Vertex Position
x_origin = np.ones((len(Y_test_use[:,4])))*46.290000915527344
y_origin = np.ones((len(Y_test_use[:,5])))*-34.880001068115234
#true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
#reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )

print('Calculating R from X & Y...')
#Calculating R from X & Y 
R_test_use_nolim = np.sqrt((Y_test_use[:,4]-x_origin)**2+(Y_test_use[:,5]-y_origin)**2)
R_test_predicted_nolim = np.sqrt(((Y_test_predicted[:,0]*maxabs_factor)-x_origin)**2+((Y_test_predicted[:,1]*maxabs_factor)-y_origin)**2)
R_reco_test_nolim = np.sqrt((reco_test[:,4]-x_origin)**2+(reco_test[:,5]-y_origin)**2)

#Set Z to make cuts easier

Z_test_use_nolim = Y_test_use[:,6]
Z_test_predicted_nolim = Y_test_predicted[:,2]
Z_reco_test_nolim = reco_test[:,6]

#R_test_use = []
#R_test_predicted = []
#R_reco_test = []

#Z_test_use = []
#Z_test_predicted = []
#Z_reco_test = []

#ZR_test_use = []
#ZR_test_predicted = []
#ZR_reco_test = []

#R_test_use_cut = []
#R_test_predicted_cut = []
#R_reco_test_cut = []

#Z_test_use_cut = []
#Z_test_predicted_cut = []
#Z_reco_test_cut = []

#ZR_test_use_cut = []
#ZR_test_predicted_cut = []
ZR_reco_test_cut = []


for i in range(len(R_test_use_nolim)):
  #True R Cuts
  if R_test_use_nolim[i] < cuts[0]:
    R_test_use.append(R_test_use_nolim[i])
  else:
    R_test_use_cuts.append(R_test_use_nolim[i])
  #True Z Cuts
  if Z_test_use_nolim[i] < cuts[2]:
    if Z_test_use_nolim[i] > cuts[1]:
      Z_test_use.append(Z_test_use_nolim[i])
  else:
    Z_test_use_cuts.append(Z_test_use_nolim[i])
  #True R and Z cuts
  if R_test_use_nolim[i] < cuts[0]:
    if Z_test_use_nolim[i] < cuts[2]:
      if Z_test_use_nolim[i] > cuts[1]:
        ZR_test_use.append(R_test_use_nolim[i])
  else:
    R_test_use_cuts.append(R_test_use_nolim[i])
    
