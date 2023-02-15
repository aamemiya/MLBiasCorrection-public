#!/bin/bash


exptype="optimize"
loc="../DATA/coupled_A13/obs_p8_010/nocorr/assim.nc"
id_exp="test"
recurrent=0
nntype="Dense_single"

python Exp.py -id $id_exp --use_optuna 0 --t "optimize" -osql $id_exp -nt 30 -e 10 -l 1 -ts 5 -ncdf_loc $loc -tb 32768 -nbs 8 -rec $recurrent -vb 8192 -norm 0 -d 1 -afm 0 --network_type "$nntype" -sprd 1

#usage: Exp.py [-h] [--id_exp ID_EXP] [--use_optuna_mlflow {0,1}]
#              [--t {optimize,best,p_best}] [--m_recu {0,1}]
#              [--epochs EPOCHS] [--num_trials NUM_TRIALS]
#              [--netcdf_dataset NETCDF_DATASET] 
#              [--optuna_sql OPTUNA_SQL] [--locality LOCALITY]
#              [--degree DEGREE] [--normalized {0,1}] [--af_mix {0,1}]
#              [--time_splits TIME_SPLITS] [--train_batch TRAIN_BATCH]
#              [--val_batch VAL_BATCH] [--num_batches NUM_BATCHES]
#
#Optuna Experiment Controller
#
#optional arguments:
#  -h, --help            show this help message and exit
#  --id_exp ID_EXP, -id ID_EXP 
#                        Experiment name / Optuna Study name
#  --use_optuna_mlflow {0,1}
#                        Run with optuna for automatic optimization or without them as single trial
#  --t {optimize,best,p_best}
#                        Choose between optimization or training best parameter
#  --recurrent {0,1}, -rec {0,1}
#                        Use recurrent layers or not
#  --epochs EPOCHS, -e EPOCHS
#                        Num of epochs for training
#  --num_trials NUM_TRIALS, -nt NUM_TRIALS
#                        Num of Optuna trials
#  --netcdf_dataset NETCDF_DATASET, -ncdf_loc NETCDF_DATASET
#                        Location of the netCDF dataset
#  --optuna_sql OPTUNA_SQL, -osql OPTUNA_SQL
#                        Optuna Study name
#  --locality LOCALITY, -l LOCALITY
#                        Locality size (including the main variable)
#  --degree DEGREE, -d DEGREE
#                        To make a polynomial input
#  --normalized {0,1}, -norm {0,1}
#                        Use normalized dataset for training.
#  --af_mix {0,1}, -afm {0,1}
#                        Use analysis forecast mixed.
#  --time_splits TIME_SPLITS, -ts TIME_SPLITS
#                        Num of RNN timesteps
#  --train_batch TRAIN_BATCH, -tb TRAIN_BATCH
#                        Training batch size
#  --val_batch VAL_BATCH, -vb VAL_BATCH
#                        Validation batch size
#  --num_batches NUM_BATCHES, -nbs NUM_BATCHES
#                        Number of training batch per epoch
