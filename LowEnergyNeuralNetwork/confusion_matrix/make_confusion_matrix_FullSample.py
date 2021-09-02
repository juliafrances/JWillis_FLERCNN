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
parser.add_argument("--cuts", nargs=3, type=int, default = [300,-500,-200], dest="cuts", 
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

print('Importing data...')

f = h5py.File(input_file, "r")
Y_test_use = f["Y_test_use"][:]
Y_test_predicted = f["Y_predicted"][:]
reco_test = f["reco_test"][:]
#weights = f["weights_test"][:]
try:
    additional_info = f["additional_info"][:]
except: 
    additional_info = None
f.close()
del f


fit_success=additional_info[:,9]
 
print(fit_success)
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

#Mask Predict on Predict
R_test_predictedmask = np.logical_and(R_test_predicted_nolim < cuts[0], fit_success)
Z_test_predictedmask = np.logical_and(np.logical_and(Z_test_predicted_nolim > cuts[1], Z_test_predicted_nolim<cuts[2]), fit_success)
#Mask True on True
R_test_usemask = np.logical_and(R_test_use_nolim < cuts[0], fit_success)
Z_test_usemask = np.logical_and(np.logical_and(Z_test_use_nolim > cuts[1], Z_test_use_nolim<cuts[2]), fit_success)
#Mask Reco on Reco
R_test_recomask = np.logical_and(R_reco_test_nolim < cuts[0], fit_success)
Z_test_recomask = np.logical_and(np.logical_and(Z_reco_test_nolim > cuts[1], Z_reco_test_nolim<cuts[2]),fit_success)

print("Masks created, calculating contained and uncontained events...")

B2 = [[sum(np.logical_and(R_test_predictedmask, R_test_usemask)), sum(np.logical_and(R_test_recomask, R_test_usemask))],
[sum(np.logical_and(Z_test_predictedmask, Z_test_usemask)), sum(np.logical_and(Z_test_recomask, Z_test_usemask))]]
#,
#[sum(np.logical_and(np.logical_and(R_test_predictedmask, R_test_usemask),np.logical_and(Z_test_predictedmask, Z_test_usemask))),
#sum(np.logical_and(np.logical_and(R_test_recomask, R_test_usemask),np.logical_and(Z_test_recomask, Z_test_usemask)))]]

B3 = [[sum(np.logical_and(np.logical_not(R_test_predictedmask), R_test_usemask)), sum(np.logical_and(np.logical_not(R_test_recomask), R_test_usemask))],
[sum(np.logical_and(np.logical_not(Z_test_predictedmask), Z_test_usemask)), sum(np.logical_and(np.logical_not(Z_test_recomask), Z_test_usemask))]]
#[sum(np.logical_and(np.logical_and(np.logical_not(R_test_predictedmask), R_test_usemask),np.logical_and(np.logical_not(Z_test_predictedmask), Z_test_usemask))),
#sum(np.logical_and(np.logical_and(np.logical_not(R_test_recomask), R_test_usemask),np.logical_and(np.logical_not(Z_test_recomask), Z_test_usemask)))]]

C2 = [[sum(np.logical_and(R_test_predictedmask, np.logical_not(R_test_usemask))), sum(np.logical_and(R_test_recomask, np.logical_not(R_test_usemask)))],
[sum(np.logical_and(Z_test_predictedmask, np.logical_not(Z_test_usemask))), sum(np.logical_and(Z_test_recomask, np.logical_not(Z_test_usemask)))]]
#[sum(np.logical_and(np.logical_and(R_test_predictedmask, np.logical_not(R_test_usemask)),np.logical_and(Z_test_predictedmask, np.logical_not(Z_test_usemask)))),
#sum(np.logical_and(np.logical_and(R_test_recomask, np.logical_not(R_test_usemask)),np.logical_and(Z_test_recomask, np.logical_not(Z_test_usemask))))]]

C3 = [[sum(np.logical_and(np.logical_not(R_test_predictedmask), np.logical_not(R_test_usemask))), sum(np.logical_and(np.logical_not(R_test_recomask), np.logical_not(R_test_usemask)))],
[sum(np.logical_and(np.logical_not(Z_test_predictedmask), np.logical_not(Z_test_usemask))), sum(np.logical_and(np.logical_not(Z_test_recomask), np.logical_not(Z_test_usemask)))]]
#[sum(np.logical_and(np.logical_and(np.logical_not(R_test_predictedmask), np.logical_not(R_test_usemask)),np.logical_and(np.logical_not(Z_test_predictedmask), np.logical_not(Z_test_usemask)))),
#sum(np.logical_and(np.logical_and(np.logical_not(R_test_recomask), np.logical_not(R_test_usemask)),np.logical_and(np.logical_not(Z_test_recomask), np.logical_not(Z_test_usemask))))]]

#I set up this code stupidly


B2RZ = [[np.logical_and(R_test_predictedmask, R_test_usemask), np.logical_and(R_test_recomask, R_test_usemask)],
[np.logical_and(Z_test_predictedmask, Z_test_usemask), np.logical_and(Z_test_recomask, Z_test_usemask)]]
#,
#[sum(np.logical_and(np.logical_and(R_test_predictedmask, R_test_usemask),np.logical_and(Z_test_predictedmask, Z_test_usemask))),
#sum(np.logical_and(np.logical_and(R_test_recomask, R_test_usemask),np.logical_and(Z_test_recomask, Z_test_usemask)))]]

B3RZ = [[np.logical_and(np.logical_not(R_test_predictedmask), R_test_usemask), np.logical_and(np.logical_not(R_test_recomask), R_test_usemask)],
[np.logical_and(np.logical_not(Z_test_predictedmask), Z_test_usemask), np.logical_and(np.logical_not(Z_test_recomask), Z_test_usemask)]]
#[sum(np.logical_and(np.logical_and(np.logical_not(R_test_predictedmask), R_test_usemask),np.logical_and(np.logical_not(Z_test_predictedmask), Z_test_usemask))),
#sum(np.logical_and(np.logical_and(np.logical_not(R_test_recomask), R_test_usemask),np.logical_and(np.logical_not(Z_test_recomask), Z_test_usemask)))]]

C2RZ = [[np.logical_and(R_test_predictedmask, np.logical_not(R_test_usemask)), np.logical_and(R_test_recomask, np.logical_not(R_test_usemask))],
[np.logical_and(Z_test_predictedmask, np.logical_not(Z_test_usemask)), np.logical_and(Z_test_recomask, np.logical_not(Z_test_usemask))]]
#[sum(np.logical_and(np.logical_and(R_test_predictedmask, np.logical_not(R_test_usemask)),np.logical_and(Z_test_predictedmask, np.logical_not(Z_test_usemask)))),
#sum(np.logical_and(np.logical_and(R_test_recomask, np.logical_not(R_test_usemask)),np.logical_and(Z_test_recomask, np.logical_not(Z_test_usemask))))]]

C3RZ = [[np.logical_and(np.logical_not(R_test_predictedmask), np.logical_not(R_test_usemask)), np.logical_and(np.logical_not(R_test_recomask), np.logical_not(R_test_usemask))],
[np.logical_and(np.logical_not(Z_test_predictedmask), np.logical_not(Z_test_usemask)), np.logical_and(np.logical_not(Z_test_recomask), np.logical_not(Z_test_usemask))]]
#[sum(np.logical_and(np.logical_and(np.logical_not(R_test_predictedmask), np.logical_not(R_test_usemask)),np.logical_and(np.logical_not(Z_test_predictedmask), np.logical_not(Z_test_usemask)))),
#sum(np.logical_and(np.logical_and(np.logical_not(R_test_recomask), np.logical_not(R_test_usemask)),np.logical_and(np.logical_not(Z_test_recomask), np.logical_not(Z_test_usemask))))]]


labels=["CNN R < %s"%(cuts[0]), "Retro R < %s"%(cuts[0])]
variable=["R","Z"]
wfile = open("%sconfusionmatrix_cutunder%d.txt"%(save_folder_name, cuts[0]),"w")
print("Saving file...")
for i in range(len(labels)):
  for j in range(len(variable)):   
    wfile.write(str(labels[i]) + '\t' +"Variable:" + str(variable[j]))
    wfile.write('\n') #creates new line
    wfile.write("B2:" + '\t' + "B3:"  + '\t' + "C2:" + '\t' + "C3:")
    wfile.write('\n') #creates new line
    wfile.write(str(B2[j][i]) + '\t'  + str(B3[j][i]) + '\t' +str(C2[j][i]) + '\t' + str(C3[j][i])) #\t puts a tab between each variable
    wfile.write('\n') #creates new line
wfile.write('CNN' + '\t' +"Variable: R&Z")
wfile.write('\n') #creates new line
wfile.write("B2:" + '\t' + "B3:"  + '\t' + "C2:" + '\t' + "C3:")
wfile.write('\n') #creates new line
wfile.write(str(sum(np.logical_and(B2RZ[0][0], B2RZ[1][0]))) + '\t'  + str(sum(np.logical_and(B3RZ[0][0], B3RZ[1][0]))) + '\t' +str(sum(np.logical_and(C2RZ[0][0], C2RZ[1][0]))) + '\t' + str(sum(np.logical_and(C3RZ[0][0], C3RZ[1][0])))) #\t puts a tab between each variable
wfile.write('\n')
wfile.write('Retro' + '\t' +"Variable: R&Z")
wfile.write('\n') #creates new line
wfile.write("B2:" + '\t' + "B3:"  + '\t' + "C2:" + '\t' + "C3:")
wfile.write('\n') #creates new line
wfile.write(str(sum(np.logical_and(B2RZ[0][1], B2RZ[1][1]))) + '\t'  + str(sum(np.logical_and(B3RZ[0][1], B3RZ[1][1]))) + '\t' +str(sum(np.logical_and(C2RZ[0][1], C2RZ[1][1]))) + '\t' + str(sum(np.logical_and(C3RZ[0][1], C3RZ[1][1])))) #\t puts a tab between each variable
wfile.write('\n') #creates new line
wfile.write('Total Events:' + '\t' + str(sum(fit_success)))
wfile.write('\n') #creates new line
wfile.close()