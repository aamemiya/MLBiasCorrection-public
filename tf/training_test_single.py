import tensorflow as tf
import numpy as np
from netCDF4 import Dataset
import time
import math
import os
import sys
import re

import helperfunctions as helpfunc
import network_arch as net

import datetime

from plotting_single import plot_history

mirrored_strategy = tf.distribute.MirroredStrategy()

def train(plist, model, checkpoint, manager, summary_writer, optimizer, train_dataset_dist, val_dataset_dist):
   
    ename = re.search('DATA/(.+?)/', plist['netCDf_loc'])
    
    rname = plist['experiment_name'] 

    with mirrored_strategy.scope():
            
            loss_func = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM, name='LossMSE')
            #loss_func = tf.keras.losses.MeanAbsoluteError(reduction = tf.keras.losses.Reduction.SUM, name='LossMSE')

            def compute_loss(labels, predictions, accuracy):
                per_example_loss = loss_func(labels, predictions,sample_weight=accuracy)
                return per_example_loss 
          
            def compute_metric(labels, predictions, accuracy):
                per_example_metric = tf.square(tf.subtract(labels*accuracy, predictions*accuracy))
                per_example_metric = tf.reduce_sum(per_example_metric)
                return per_example_metric 

            def train_step(inputs):
                with tf.GradientTape() as tape:
                    
                    local_forecast, analysis, accuracy = inputs

                    pred_analysis, _ = model(local_forecast, stat = [])

                    #Calculating relative loss
                    try:
                        loss = compute_loss(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :]) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size']))
                    except:
                        loss = compute_loss(analysis, pred_analysis, accuracy) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size']))

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                try:
                    metric = (compute_metric(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :])) * (1.0 / (plist['global_batch_size']))
                except:
                    metric = (compute_metric(analysis, pred_analysis, accuracy)) * (1.0 / (plist['global_batch_size']))

                return loss, metric

            def val_step(inputs):
                local_forecast, analysis, accuracy = inputs
                pred_analysis, _ = model(local_forecast, stat = [])
 
                try:
                    loss = compute_loss(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :]) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size_v']))
                except:
                    loss = compute_loss(analysis, pred_analysis, accuracy) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size_v']))

                try:
                    metric = (compute_metric(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :])) * (1.0 / (plist['global_batch_size_v']))
                except:
                    metric = (compute_metric(analysis, pred_analysis, accuracy)) * (1.0 / (plist['global_batch_size_v']))

                return loss, metric 

            @tf.function
            def distributed_train_step(inputs):
                per_replica_losses, per_replica_metric = mirrored_strategy.run(train_step, args=(inputs,))
                loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                met = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_metric, axis=None)
                return loss, tf.sqrt(met)

            @tf.function
            def distributed_val_step(inputs):
                per_replica_losses, per_replica_metric = mirrored_strategy.run(val_step, args=(inputs,))
                loss =  mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                met =  mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_metric, axis=None)
                return loss, tf.sqrt(met)
            
            #Initialing training variables
            global_step = 0
            global_step_val = 0
            val_min = 0
            val_loss_min = plist['val_min']
            timer_tot = time.time()

            epoch_nums=[]
            losses=[]
            t_metrics=[]
            v_metrics=[]

            #Starting training
            with summary_writer.as_default():

                epochs = plist['epochs']

                for epoch in range(epochs):

                    start_time = time.time()

                    plist['global_epoch'] += 1

#                    print('\nStart of epoch %d' %(plist['global_epoch']))

                    # Iterate over the batches of the dataset.
                    for step, inputs in enumerate(train_dataset_dist):
                        global_step += 1

                        # Open a GradientTape to record the operations run
                        # during the forward pass, which enables autodifferentiation.
                        loss, t_metric = distributed_train_step(inputs)

                        #if (step % plist['log_freq']) == 0:
                        #    print('Training loss (for one batch) at step %s: %s' % (step+1, float(loss)))
                            
                    # Display metrics at the end of each epoch.
#                    print('\nTraining loss at epoch end {}'.format(loss))
#                    print('Training acc over epoch: %s ' % (float(t_metric)))
                    #print('Seen so far: %s samples\n' % ((global_step) * plist['global_batch_size']))

                    #Code for validation at the end of each epoch
                    for step_val, val_inputs in enumerate(val_dataset_dist):

                        global_step_val += 1

                        val_loss, v_metric = distributed_val_step(val_inputs)

                        #if (step_val % plist['log_freq']) == 0:
                        #    print('Validation loss (for one batch) at step {}: {}'.format(step_val+1, val_loss))
                            
#                    print('Validation acc over epoch: %s \n' % (float(v_metric)))
                    print('epoch %d : Train loss, acc, Validation acc %s %s %s' % (plist['global_epoch'], format(loss), float(t_metric), float(v_metric)))
                    epoch_nums.append(plist['global_epoch']) 
                    losses.append(loss)                
                    t_metrics.append(t_metric)                
                    v_metrics.append(v_metric)                

                        
                    if val_loss_min > v_metric:

                        val_loss_min = v_metric

                        checkpoint.epoch.assign_add(1)
                        save_path = manager.save()
###                        print("Saved checkpoint for epoch {}: {}".format(checkpoint.epoch.numpy(), save_path))
###                        print("\nRMSE {}\n".format(v_metric.numpy()))
                        plist['val_min'] = val_loss_min

                    if math.isnan(v_metric):
                        print('Breaking out as the validation loss is nan')
                        break                

                    if (epoch > 19):
                        if not (epoch % plist['early_stop_patience']):
                            if not (val_min):
                                val_min = val_loss_min
                            else:
                                if val_min > val_loss_min:
                                    val_min = val_loss_min
                                else:
                                    print('Breaking loop as validation accuracy not improving')
                                    print("loss {}".format(loss.numpy()))
                                    break

#            fig_hist=plot_history(epoch_nums,t_metrics,v_metrics,normalized=True)
#            fig_hist.savefig('history.png')
#            with open('result.txt','w') as tmpf :
#               tmpf.write('epoch %d : Train loss, acc, Validation acc %s %s %s \n' % (plist['global_epoch'], format(loss), float(t_metric), float(v_metric)))
            
#                    print('Time for epoch (seconds): %s' %((time.time() - start_time)))
    
#    print('\n Total training time (in minutes): {}'.format((time.time()-timer_tot)/60))

    helpfunc.write_pickle(plist, plist['pickle_name'])

#    model.summary()

    return val_loss_min
    
def traintest(plist):

    print('\nGPU Available: {}\n'.format(tf.config.list_physical_devices('GPU')))

    #Get dataset
    print("\nProcessing Dataset\n")

    forecast_dataset, analysis_dataset, accuracy_dataset, a_f, s_f = helpfunc.createdataset(plist)
    if plist['make_recurrent']:
        analysis_split = helpfunc.split_sequences(analysis_dataset[:,:,:], plist['time_splits'])
        analysis_split = np.transpose(analysis_split, (1,0,2,3))
        accuracy_split = helpfunc.split_sequences(accuracy_dataset[:,:,:], plist['time_splits'])
        accuracy_split = np.transpose(accuracy_split, (1,0,2,3))
        forecast_split = helpfunc.split_sequences(forecast_dataset[:,:,:], plist['time_splits'])
        forecast_split = np.transpose(forecast_split, (1,0,2,3))

        analysis_dataset = np.reshape(analysis_split, (analysis_split.shape[0]*analysis_split.shape[1], plist['time_splits'], 1))
        accuracy_dataset = np.reshape(accuracy_split, (accuracy_split.shape[0]*accuracy_split.shape[1], plist['time_splits'], 1))
        forecast_dataset = np.reshape(forecast_split, (forecast_split.shape[0]*forecast_split.shape[1], plist['time_splits'], plist['locality']))

        if plist['anal_for_mix']:
            forecast_dataset[:, :-1, :] = analysis_dataset[:, :-1, :]

    else:
        plist['time_splits'] = 1
        analysis_dataset = np.reshape(analysis_dataset, (analysis_dataset.shape[0]*analysis_dataset.shape[1], 1))
        accuracy_dataset = np.reshape(accuracy_dataset, (accuracy_dataset.shape[0]*accuracy_dataset.shape[1], 1))
        forecast_dataset = np.reshape(forecast_dataset, (forecast_dataset.shape[0]*forecast_dataset.shape[1], plist['locality'] * plist['degree']))

    tfdataset_analysis_train = helpfunc.create_tfdataset(analysis_dataset[:-plist['val_size']])
    tfdataset_accuracy_train = helpfunc.create_tfdataset(accuracy_dataset[:-plist['val_size']])
    tfdataset_forecast_train = helpfunc.create_tfdataset(forecast_dataset[:-plist['val_size']])

    tfdataset_analysis_val = helpfunc.create_tfdataset(analysis_dataset[-plist['val_size']:])
    tfdataset_accuracy_val = helpfunc.create_tfdataset(accuracy_dataset[-plist['val_size']:])
    tfdataset_forecast_val = helpfunc.create_tfdataset(forecast_dataset[-plist['val_size']:])
    
    #Zipping the files
    dataset_train = tf.data.Dataset.zip((tfdataset_forecast_train, tfdataset_analysis_train, tfdataset_accuracy_train))
    dataset_val = tf.data.Dataset.zip((tfdataset_forecast_val, tfdataset_analysis_val, tfdataset_accuracy_val))

    #Shuffling the dataset
    tf.random.set_seed(5)
    dataset_train = dataset_train.shuffle(1000000, seed = 1)
    dataset_train = dataset_train.batch(batch_size=plist['global_batch_size'], drop_remainder=True)

    dataset_val = dataset_val.shuffle(1000000, seed = 1)
    dataset_val = dataset_val.batch(batch_size=plist['global_batch_size_v'], drop_remainder=True)

    #Distributing the dataset
    val_dataset_dist = mirrored_strategy.experimental_distribute_dataset(dataset_val)
    train_dataset_dist = mirrored_strategy.experimental_distribute_dataset(dataset_train)

    #Get the Model
    with mirrored_strategy.scope():
        model = net.rnn_model(plist)

        #Defining Model compiling parameters
        if plist['lr_decay_rate']:
            learningrate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(plist['learning_rate'],
                                                                          decay_steps = plist['lr_decay_steps'],
                                                                          decay_rate = plist['lr_decay_rate'],
                                                                          staircase = False)
            learning_rate = learningrate_schedule 
        else:
            learning_rate = plist['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #Defining the checkpoint instance
        a_f = tf.Variable(a_f, dtype = tf.float32)
        s_f = tf.Variable(s_f, dtype = tf.float32)
        time_splits = tf.Variable(plist['time_splits'])
        checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0), model = model, a_f = a_f, s_f = s_f, time_splits = time_splits)

    #Creating summary writer
    summary_writer = tf.summary.create_file_writer(logdir= plist['log_dir'])

    #Creating checkpoint instance
    save_directory = plist['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory, 
                                        max_to_keep= plist['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()


    model2 = net.rnn_model(plist)
    checkpoint2 = tf.train.Checkpoint(epoch = tf.Variable(0), model = model2, a_f = a_f, s_f = s_f, time_splits = time_splits)
    checkpoint2.restore(manager.latest_checkpoint).expect_partial()
 
    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
            
        print('Starting training from a restored point... \n')
        return train(plist, model, checkpoint, manager, summary_writer, optimizer, train_dataset_dist, val_dataset_dist)
        
    else:
        print("No checkpoint exists.")
        
        print('Initializing from scratch... \n')
        return train(plist, model, checkpoint, manager, summary_writer, optimizer, train_dataset_dist, val_dataset_dist)

    print(learning_rate)



def val(plist, model, summary_writer, optimizer, val_dataset_dist):
   
    ename = re.search('DATA/(.+?)/', plist['netCDf_loc'])
    
    rname = plist['experiment_name'] 

    with mirrored_strategy.scope():
            
            loss_func = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM, name='LossMSE')
            #loss_func = tf.keras.losses.MeanAbsoluteError(reduction = tf.keras.losses.Reduction.SUM, name='LossMSE')

            def compute_loss(labels, predictions, accuracy):
                per_example_loss = loss_func(labels, predictions,sample_weight=accuracy)
                return per_example_loss 
          
            def compute_metric(labels, predictions, accuracy):
                per_example_metric = tf.square(tf.subtract(labels*accuracy, predictions*accuracy))
                per_example_metric = tf.reduce_sum(per_example_metric)
                return per_example_metric 

            def val_step(inputs):
                local_forecast, analysis, accuracy = inputs
                pred_analysis, _ = model(local_forecast, stat = [])
 
                try:
                    loss = compute_loss(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :]) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size_v']))
                except:
                    loss = compute_loss(analysis, pred_analysis, accuracy) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size_v']))

                try:
                    metric = (compute_metric(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :])) * (1.0 / (plist['global_batch_size_v']))
                except:
                    metric = (compute_metric(analysis, pred_analysis, accuracy)) * (1.0 / (plist['global_batch_size_v']))

                return loss, metric 

            @tf.function
            def distributed_val_step(inputs):
                per_replica_losses, per_replica_metric = mirrored_strategy.run(val_step, args=(inputs,))
                loss =  mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                met =  mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_metric, axis=None)
                return loss, tf.sqrt(met)
            
            #Starting training
            with summary_writer.as_default():

                    #Code for validation at the end of each epoch
                    for step_val, val_inputs in enumerate(val_dataset_dist):

                        val_loss, v_metric = distributed_val_step(val_inputs)

                        #if (step_val % plist['log_freq']) == 0:
                        #    print('Validation loss (for one batch) at step {}: {}'.format(step_val+1, val_loss))
                            
                        print('Validation loss acc %s %s' % (format(val_loss), float(v_metric)))

#    model.summary()
                        
    return val_loss
 

def test(plist):

    print('\nGPU Available: {}\n'.format(tf.config.list_physical_devices('GPU')))

    #Get dataset
    print("\nProcessing Dataset\n")

    forecast_dataset, analysis_dataset, accuracy_dataset, a_f, s_f = helpfunc.createdataset(plist)
    if plist['make_recurrent']:
        analysis_split = helpfunc.split_sequences(analysis_dataset[:,:,:], plist['time_splits'])
        analysis_split = np.transpose(analysis_split, (1,0,2,3))
        accuracy_split = helpfunc.split_sequences(accuracy_dataset[:,:,:], plist['time_splits'])
        accuracy_split = np.transpose(accuracy_split, (1,0,2,3))
        forecast_split = helpfunc.split_sequences(forecast_dataset[:,:,:], plist['time_splits'])
        forecast_split = np.transpose(forecast_split, (1,0,2,3))

        analysis_dataset = np.reshape(analysis_split, (analysis_split.shape[0]*analysis_split.shape[1], plist['time_splits'], 1))
        accuracy_dataset = np.reshape(accuracy_split, (accuracy_split.shape[0]*accuracy_split.shape[1], plist['time_splits'], 1))
        forecast_dataset = np.reshape(forecast_split, (forecast_split.shape[0]*forecast_split.shape[1], plist['time_splits'], plist['locality']))

        if plist['anal_for_mix']:
            forecast_dataset[:, :-1, :] = analysis_dataset[:, :-1, :]

    else:
        plist['time_splits'] = 1
        analysis_dataset = np.reshape(analysis_dataset, (analysis_dataset.shape[0]*analysis_dataset.shape[1], 1))
        accuracy_dataset = np.reshape(accuracy_dataset, (accuracy_dataset.shape[0]*accuracy_dataset.shape[1], 1))
        forecast_dataset = np.reshape(forecast_dataset, (forecast_dataset.shape[0]*forecast_dataset.shape[1], plist['locality'] * plist['degree']))

    tfdataset_analysis_val = helpfunc.create_tfdataset(analysis_dataset[:])
    tfdataset_accuracy_val = helpfunc.create_tfdataset(accuracy_dataset[:])
    tfdataset_forecast_val = helpfunc.create_tfdataset(forecast_dataset[:])
    
    #Zipping the files
    dataset_val = tf.data.Dataset.zip((tfdataset_forecast_val, tfdataset_analysis_val, tfdataset_accuracy_val))

    #Shuffling the dataset
###    tf.random.set_seed(5)
###    dataset_val = dataset_val.shuffle(1000000, seed = 1)
    plist['global_batch_size_v'] = len(list(dataset_val))
    dataset_val = dataset_val.batch(batch_size=plist['global_batch_size_v'], drop_remainder=False)

#    list(dataset_val.as_numpy_iterator())
#    quit()

    #Distributing the dataset
    val_dataset_dist = mirrored_strategy.experimental_distribute_dataset(dataset_val)

    #Get the Model
    with mirrored_strategy.scope():
        model = net.rnn_model(plist)

        #Defining Model compiling parameters
        if plist['lr_decay_rate']:
            learningrate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(plist['learning_rate'],
                                                                          decay_steps = plist['lr_decay_steps'],
                                                                          decay_rate = plist['lr_decay_rate'],
                                                                          staircase = False)
            learning_rate = learningrate_schedule 
        else:
            learning_rate = plist['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        #Defining the checkpoint instance
        a_f = tf.Variable(a_f, dtype = tf.float32)
        s_f = tf.Variable(s_f, dtype = tf.float32)
        time_splits = tf.Variable(plist['time_splits'])
        checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0), model = model, a_f = a_f, s_f = s_f, time_splits = time_splits)

    #Creating summary writer
    summary_writer = tf.summary.create_file_writer(logdir= plist['log_dir'])

#    #Creating checkpoint instance
    save_directory = plist['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory, 
                                        max_to_keep= plist['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()


    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
            
        print('Starting validation using a restored point... \n')
        return val(plist, model, summary_writer, optimizer, val_dataset_dist)
#    return val(plist, model, summary_writer, optimizer, val_dataset_dist)
        
    else:
        print("No checkpoint exists. Stop.")

