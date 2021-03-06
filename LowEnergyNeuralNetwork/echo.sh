#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=119:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --gres=gpu:1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=28G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name CNN_Energytestjob      # you can give your job a name for easier identification (does not have to be unique) (same as -J)
#SBATCH --output /mnt/home/willis51/LowEnergyNeuralNetwork/make_jobs/run_CNN/CNN_energytestjorb.out

########## Command Lines to Run ##########

USERNAME=willis51
INPUT="NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC19__E1to500_CC_all_start_all_end_flat_499bins_20000evtperbin_file??.hdf5"
INDIR="/mnt/research/IceCube/jmicallef/DNN_files/"
OUTDIR="/mnt/home/${USERNAME}/LowEnergyNeuralNetwork"
NUMVAR=1
LR_EPOCH=50
LR_DROP=0.1
LR=0.001
OUTNAME="EnergyTest" #Can name all your outnames differently, it's a folder where the stuff goes
FILENAME=CNN_LoadMultipleFiles.py
FIRSTVARIABLE=energy #when you go to XYZ, it should have no affect, shouldn't need this argument, should be able to leave it as energy

echo "singularity exec -B /mnt/home/$USERNAME:/mnt/home/$USERNAME -B /mnt/research/IceCube:/mnt/research/IceCube --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif python ${OUTDIR}/$FILENAME --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --start $EPOCH --variables $NUMVAR --first_variable $FIRSTVARIABLE --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP --no_test True"
