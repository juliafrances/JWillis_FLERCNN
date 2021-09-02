#################################
# Plots input and output features for ONE file
#   Inputs:
#       -i input file:  name of ONE file
#       -d  path:       path to input files
#       -o  outdir:     path to output_plots directory or where final dir will be created
#       -n  name:       Name of directory to create in outdir (associated to filenames)
#       --emax:         Energy max cut, plot events below value, used for UN-TRANSFORM
#       --emin:         Energy min cut, plot events above value
#       --tmax:         Track factor to multiply, use for UN-TRANSFORM
#   Outputs:
#       File with count in each bin
#       Histogram plot with counts in each bin
#################################

import numpy as np
import h5py
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default=None,
                    type=str,dest="input_file", help="name for ONE input file")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/micall12/training_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/micall12/LowEnergyNeuralNetwork/output_plots/',
                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",default=None,
                    dest="name", help="name for output folder")
parser.add_argument("--filenum",default=None,
                    dest="filenum", help="number for file, if multiple with same name")
args = parser.parse_args()

input_file = args.path + args.input_file
output_path = args.outdir
name = args.name
outdir = output_path
if os.path.isdir(outdir) != True:
    os.mkdir(outdir)
print("Saving plots to %s"%outdir)

if args.filenum:
    filenum = str(args.filenum)
else:
    filenum=args.filenum

f = h5py.File(input_file, 'r')
#if file_was_transformed:
#    Y_test = f['Y_test'][:]
#    X_test_DC = f['X_test_DC'][:]
#    X_test_IC = f['X_test_IC'][:]
#else:
#    Y_test = f['labels'][:]
#    X_test_DC = f['features_DC'][:]
#    X_test_IC = f['features_IC'][:]

#try:
Y_train = f['Y_train'][:]
X_train_DC = f['X_train_DC'][:]
X_train_IC = f['X_train_IC'][:]

#    Y_validate = f['Y_validate'][:]
#    X_validate_DC = f['X_validate_DC'][:]
#    X_validate_IC = f['X_validate_IC'][:]
#except:    
#    Y_train = None
#    X_train_DC = None
#    X_train_IC = None

#    Y_validate = None
#    X_validate_DC = None
#    X_validate_IC = None

f.close()
del f

#if Y_train is None: #Test only file
#    Y_labels = Y_test
#    X_DC = X_test_DC
#    X_IC = X_test_IC
#else:
#    Y_labels = np.concatenate((Y_test,Y_train,Y_validate))
#    X_DC = np.concatenate((X_test_DC,X_train_DC,X_validate_DC))
#    X_IC = np.concatenate((X_test_IC,X_train_IC,X_validate_IC))
#print(Y_labels.shape,X_DC.shape,X_IC.shape)
#print(len(output_factors))

#if reco_test is not None:
#    reco_labels = np.concatenate((reco_test,reco_train,reco_validate))

#Vertex Position
x_origin = np.ones((len(Y_train[:,4])))*46.290000915527344
y_origin = np.ones((len(Y_train[:,5])))*-34.880001068115234
#true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
#reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )


#Calculating R from X & Y 
R_values = np.sqrt(((Y_train[:,4])-x_origin)**2+((Y_train[:,5])-y_origin)**2)

Z_values = Y_train[:,6]

plt.figure()
plt.hist(R_values,bins=100);
plt.vlines(x=150, ymin=0,ymax=30000, linestyle='--', color="tab:red", label = "DeepCore")
plt.vlines(x=90, ymin=0,ymax=30000, linestyle='--', color="tab:green", label = "Inner DeepCore")
plt.legend()
plt.title("R Distribution",fontsize=25)
plt.xlabel("R (m)",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

if filenum:
    filenum_name = "_%s"%filenum
else:
    filenum_name = ""
#plt.savefig("%s/Output_%s%s.png"%(outdir,names[0].replace(" ", ""),filenum_name))
plt.savefig("%s/Output_R%s.png"%(outdir,filenum_name))


plt.figure()
plt.hist(Z_values,bins=100);
plt.title("Z Distribution",fontsize=25)
plt.xlabel("Z (m)",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
if filenum:
    filenum_name = "_%s"%filenum
else:
    filenum_name = ""
#plt.savefig("%s/Output_%s%s.png"%(outdir,names[0].replace(" ", ""),filenum_name))
plt.savefig("%s/Output_Z%s.png"%(outdir,filenum_name))