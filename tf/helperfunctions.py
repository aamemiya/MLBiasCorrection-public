import tensorflow as tf
import numpy as np
from netCDF4 import Dataset
import time
import math
import pandas as pd
import pickle
import sys

#For creating locality for individual state variable
def locality_creator(init_dataset, locality_range, xlocal):
    
    output_dataset = np.zeros((init_dataset.shape[0], init_dataset.shape[1], locality_range))
    radius = int(locality_range / 2)
    
    locality = np.linspace(-radius, radius, locality_range)
    locality = np.true_divide(locality, xlocal)
    locality = np.power(locality, 2)
    locality = np.exp((-1/2) * locality)
    
    for i in range(init_dataset.shape[1]):
        start = i - radius
        stop = i + radius
        index = np.linspace(start,stop,locality_range, dtype='int')
        if stop >= init_dataset.shape[1]:
            stop2 = (stop + 1)%init_dataset.shape[1]
            index[:-stop2] = np.linspace(start,init_dataset.shape[1]-1,init_dataset.shape[1]-start, dtype='int')
            index[-stop2:] = np.arange(0,stop2,1,dtype='int')
        output_dataset[:,i,:] = init_dataset[:,index]

    #return np.multiply(np.transpose(output_dataset,(1,0,2)), locality).astype('float32') 
    return np.transpose(output_dataset,(1,0,2))

def make_poly(init_dataset, degree):
    
    output_dataset = np.zeros((init_dataset.shape[0], init_dataset.shape[1], degree))
    output_dataset[:,:,0] = init_dataset
    for i in range(degree-1):
        output_dataset[:,:,i+1] = np.power(init_dataset, i+2)

    return output_dataset

#For creating the truth label
def truth_label_creator(init_dataset):
    output_dataset = init_dataset[:]
    output_dataset = np.expand_dims(output_dataset, axis=0)
    return np.transpose(output_dataset.astype('float32'))

#Creating time data splits
def split_sequences(sequences, n_steps):
    X = list()
    for i in range(sequences.shape[1]):
        # find the end of this pattern
        end_ix = i*n_steps + n_steps
        # check if we are beyond the dataset
        if end_ix > sequences.shape[1]:
            break
        # gather input and output parts of the pattern
        seq_x = sequences[:,i*n_steps:end_ix, :]
        X.append(seq_x)
    return np.array(X)

#For serializing the tensor to a string for TFRecord
def _serialize_tensor(value):
    return tf.io.serialize_tensor(value)

#For writing data to the TFRecord file
def write_TFRecord(filename, dataset, time_splits):
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(dataset.shape[0]):
            dataset_splits = split_sequences(dataset[i], time_splits)
            for j in range(dataset_splits.shape[0]):
                data = dataset_splits[j]
                serial_string = _serialize_tensor(data)
                writer.write(serial_string.numpy())
    writer.close()

#For reading the TFRecord File
def read_TFRecord(filename):
    return tf.data.TFRecordDataset(filename)

#For parsing the value from string to float32
def _parse_tensor(value):
    return tf.io.parse_tensor(value, out_type=tf.float32)

#For creating Train and Validation datasets
def train_val_creator(dataset, val_size):
    val_dataset = dataset.take(val_size)
    train_dataset = dataset.skip(val_size)
    return train_dataset, val_dataset

def read_pickle(filename):
    pickle_in = open(filename, "rb")
    plist = pickle.load(pickle_in)
    pickle_in.close()
    return plist

def write_pickle(dicty, filename):   
    pickle_out = open(filename, "wb")
    pickle.dump(dicty, pickle_out)
    pickle_out.close()

def tfrecord(plist):
    #Getting the NetCDF files
    root_grp = Dataset(plist['netCDf_loc'], "r", format="NETCDF4")

    #Extrating the datasets
    analysis_init = root_grp["vam"]
    forecast_init = root_grp["vfm"]

    analysis_dataset = truth_label_creator(analysis_init[:plist['num_timesteps']])
    forecast_dataset = locality_creator(forecast_init[:plist['num_timesteps']], plist['locality'], plist['xlocal'])
#    analysis_dataset = truth_label_creator(analysis_init[:])
#    forecast_dataset = locality_creator(forecast_init[:], plist['locality'], plist['xlocal'])

    write_TFRecord(plist['tfrecord_analysis'], analysis_dataset, plist['time_splits'])
    write_TFRecord(plist['tfrecord_forecast'], forecast_dataset, plist['time_splits'])

def write_to_json(loc, model):
    with open(loc, 'w') as json_file:
        json_file.write(model)

def read_json(loc):
    json_file = open(loc, 'r')
    content = json_file.read()
    json_file.close()
    return content

def createdataset(plist):

    #Getting the NetCDF files
    root_grp = Dataset(plist['netCDf_loc'], "r", format="NETCDF4")

    #Extrating the datasets
    analysis_init = np.array(root_grp["vam"])[100:plist['num_timesteps']]
    forecast_init = np.array(root_grp["vfm"])[100:plist['num_timesteps']]
#    analysis_init = np.array(root_grp["vam"])[100:]
#    forecast_init = np.array(root_grp["vfm"])[100:]
    print(np.min(analysis_init), np.max(analysis_init))
    print(np.count_nonzero(analysis_init < 0) / (analysis_init.shape[0] * analysis_init.shape[1]))
    rmse = np.sqrt(np.mean(np.power(analysis_init - forecast_init, 2)))
    print('RMSE of Analysis - Forecast: ', rmse)

    d_analysis_init = analysis_init - forecast_init
    print(np.count_nonzero(d_analysis_init > 0) / (d_analysis_init.shape[0] * d_analysis_init.shape[1]))

### inverse of spread == accuracy weight  ### 
    if plist['use_sprd'] == 1:
        spread_init = np.std(np.array(root_grp["va"])[100:plist['num_timesteps']]-np.array(root_grp["vf"])[100:plist['num_timesteps']],axis=1)
        spread_init = 1.0 / np.clip(spread_init, 1.0e-6, 1.0e6)
        spread_init = spread_init / np.average(spread_init) # normalize
    else:
        spread_init = np.ones(analysis_init.shape)
 
    if plist['normalized'] == 1:
        print('\nUsing Normalized Dataset.\n')
        ave_forecast = np.average(forecast_init,axis=0)
        std_forecast = np.std(forecast_init,axis=0)

    else:
        ave_forecast = np.zeros(forecast_init.shape[1])
        std_forecast = np.ones(forecast_init.shape[1])


    forecast_init_norm = np.true_divide(forecast_init - ave_forecast, std_forecast) 
    print('Forecast normalized min: ', np.min(forecast_init_norm), ' max: ', np.max(forecast_init_norm))

    analysis_dataset = truth_label_creator(d_analysis_init)
    spread_dataset = truth_label_creator(spread_init)
    print(forecast_init_norm.shape)
    forecast_dataset = locality_creator(forecast_init_norm, plist['locality'], plist['xlocal'])
 
    if plist['degree'] > 1:
        if plist['locality'] == 1:
            forecast_dataset = make_poly(np.squeeze(forecast_dataset), plist['degree'])
            print('Poly forecast shape: ', forecast_dataset.shape)
        else:
            print('Cannot implement polynomial version as locality is not 1.')
            sys.exit()
    return forecast_dataset, analysis_dataset, spread_dataset, ave_forecast, std_forecast

#Code for creating Tensorflow Dataset:
def create_tfdataset(initial_dataset):
    tf_dataset = tf.data.Dataset.from_tensor_slices(initial_dataset)
    return tf_dataset
