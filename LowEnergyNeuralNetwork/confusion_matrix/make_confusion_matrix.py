import h5py
import argparse
import os, sys
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default=None,
                    dest="input_file", help="path and name of the input file")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/ufs18/home-049/willis51/LowEnergyNeuralNetwork/confusion_matrix/matrices/',
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

#Vertex Position
x_origin = np.ones((len(Y_test_use[:,4])))*46.290000915527344
y_origin = np.ones((len(Y_test_use[:,5])))*-34.880001068115234
#true_r = np.sqrt( (true_x - x_origin)**2 + (true_y - y_origin)**2 )
#reco_r = np.sqrt( (reco_x - x_origin)**2 + (reco_y - y_origin)**2 )

#Calculating R from X & Y 
R_test_use_nolim = np.sqrt((Y_test_use[:,4]-x_origin)**2+(Y_test_use[:,5]-y_origin)**2)
R_test_predicted_nolim = np.sqrt(((Y_test_predicted[:,0]*maxabs_factor)-x_origin)**2+((Y_test_predicted[:,1]*maxabs_factor)-y_origin)**2)
R_reco_test_nolim = np.sqrt((reco_test[:,4]-x_origin)**2+(reco_test[:,5]-y_origin)**2)

#Set Z to make cuts easier

Z_test_use_nolim = Y_test_use[:,6]
Z_test_predicted_nolim = Y_test_predicted[:,2]
Z_reco_test_nolim = reco_test[:,6]

#Make the cuts

#Mask Predict on Predict
R_test_predictedmask = R_test_predicted_nolim < cuts[0]
Z_test_predictedmask = np.logical_and(Z_test_predicted_nolim > cuts[1], Z_test_predicted_nolim<cuts[2])
#Mask True on True
R_test_usemask = R_test_use_nolim < cuts[0]
Z_test_usemask = np.logical_and(Z_test_use_nolim > cuts[1], Z_test_use_nolim<cuts[2])
#Mask Reco on Reco
R_test_recomask = R_reco_test_nolim < cuts[0]
Z_test_recomask = np.logical_and(Z_reco_test_nolim > cuts[1], Z_reco_test_nolim<cuts[2])

print("Masks created, calculating contained and uncontained events...")

B2 = [[sum(np.logical_and(R_test_predictedmask, R_test_usemask)), sum(np.logical_and(R_test_recomask, R_test_usemask))],
[sum(np.logical_and(Z_test_predictedmask, Z_test_usemask)), sum(np.logical_and(Z_test_recomask, Z_test_usemask))],
[sum(np.logical_and(np.logical_and(R_test_predictedmask, R_test_usemask),np.logical_and(Z_test_predictedmask, Z_test_usemask))),
sum(np.logical_and(np.logical_and(R_test_recomask, R_test_usemask),np.logical_and(Z_test_recomask, Z_test_usemask)))]]

B3 = [[sum(np.logical_and(np.logical_not(R_test_predictedmask), R_test_usemask)), sum(np.logical_and(np.logical_not(R_test_recomask), R_test_usemask))],
[sum(np.logical_and(np.logical_not(Z_test_predictedmask), Z_test_usemask)), sum(np.logical_and(np.logical_not(Z_test_recomask), Z_test_usemask))],
[sum(np.logical_and(np.logical_and(np.logical_not(R_test_predictedmask), R_test_usemask),np.logical_and(np.logical_not(Z_test_predictedmask), Z_test_usemask))),
sum(np.logical_and(np.logical_and(np.logical_not(R_test_recomask), R_test_usemask),np.logical_and(np.logical_not(Z_test_recomask), Z_test_usemask)))]]

C2 = [[sum(np.logical_and(R_test_predictedmask, np.logical_not(R_test_usemask))), sum(np.logical_and(R_test_recomask, np.logical_not(R_test_usemask)))],
[sum(np.logical_and(Z_test_predictedmask, np.logical_not(Z_test_usemask))), sum(np.logical_and(Z_test_recomask, np.logical_not(Z_test_usemask)))],
[sum(np.logical_and(np.logical_and(R_test_predictedmask, np.logical_not(R_test_usemask)),np.logical_and(Z_test_predictedmask, np.logical_not(Z_test_usemask)))),
sum(np.logical_and(np.logical_and(R_test_recomask, np.logical_not(R_test_usemask)),np.logical_and(Z_test_recomask, np.logical_not(Z_test_usemask))))]]

C3 = [[sum(np.logical_and(np.logical_not(R_test_predictedmask), np.logical_not(R_test_usemask))), sum(np.logical_and(np.logical_not(R_test_recomask), np.logical_not(R_test_usemask)))],
[sum(np.logical_and(np.logical_not(Z_test_predictedmask), np.logical_not(Z_test_usemask))), sum(np.logical_and(np.logical_not(Z_test_recomask), np.logical_not(Z_test_usemask)))],
[sum(np.logical_and(np.logical_and(np.logical_not(R_test_predictedmask), np.logical_not(R_test_usemask)),np.logical_and(np.logical_not(Z_test_predictedmask), np.logical_not(Z_test_usemask)))),
sum(np.logical_and(np.logical_and(np.logical_not(R_test_recomask), np.logical_not(R_test_usemask)),np.logical_and(np.logical_not(Z_test_recomask), np.logical_not(Z_test_usemask))))]]

labels=["CNN R < %s"%(cuts[0]), "Retro R < %s"%(cuts[0])]
variable=["R","Z", "Z & R"]
wfile = open("%sconfusionmatrix_cutunder%d.txt"%(save_folder_name, cuts[0]),"w")
print("Saving file...")
for i in range(len(labels)):
  for j in range(len(variable)):   
    wfile.write(str(labels[i]) + '/t' +"Variable:" + str(variable[j]))
    wfile.write('\n') #creates new line
    wfile.write("B2:" + '\t' + "B3:"  + '\t' + "C2:" + '\t' + "C3:")
    wfile.write('\n') #creates new line
    wfile.write(str(B2[j][i]) + '\t'  + str(B3[j][i]) + '\t' +str(C2[j][i]) + '\t' + str(C3[j][i])) #\t puts a tab between each variable
    wfile.write('\n') #creates new line
wfile.close()