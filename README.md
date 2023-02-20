# MLBiasCorrection

This repository contains the codes for a paper by [Amemiya, Shlok, and Miyoshi](https://doi.org/10.1029/2022MS003164) (2023), *Journal of Advances in Modeling Earth Systems*.

## Acknowledgement
The main part of the Lorenz96 LETKF code is originally developed by Shigenori Otsuka ([https://github.com/otsuka-shigenori/da_demo](https://github.com/otsuka-shigenori/da_demo))  
The codes usnig Tensorflow and Optuna is developed by Shlok Mohta. 

## Run the experiment 

### Parameter settings 

Set parameters in `param.py`
```
param_model = {} 
param_model['dimension'] = 8                ### Number of grid points
param_model['dimension_coupled'] = 256      ### Number of grid points (fast variable)
param_model['forcing'] = 20                 ### Value of F 
param_model['dt'] = 0.005                   ### dt
param_model['dt_coupled'] = 0.001           ### dt/c
param_model['h'] = 1                        ### Coupling parameters
param_model['b'] = 10                       ###
param_model['c'] = 4                        ###

param_letkf = {}
param_letkf['obs_error'] = 0.10               ### Observation error standard deviation
param_letkf['obs_number'] =8                  ### Number of grids to observe
param_letkf['localization_length_scale'] = 1  ### Width scale for localization
param_letkf['localization_length_cutoff'] = 1 ### Cutoff scale for localization  
param_letkf['inflation_factor'] = 1.0         ### Multiplicative inflation factor
param_letkf['additive_inflation_factor']=0.25 ### Additive inflation factor
param_letkf['missing_value'] = -9.99e8

param_exp = {}
param_exp['exp_length'] = 30000               ### Number of DA steps
param_exp['ensembles'] = 10                   ### Ensemble size
param_exp['expdir'] = './DATA/coupled_A13'    ### Output directory
param_exp['obs_type'] = 'obs_010'             ### Output subdirectory
param_exp['da_type'] = 'test'                 ### Output subdirectory
param_exp['dt_nature'] = 0.05                 ### dt for nature run output
param_exp['dt_obs'] = 0.05                    ### dt for obs output
param_exp['dt_assim'] = 0.05                  ### dt for DA output
param_exp['spinup_length'] = 2000             ### Steps to spinup

```

### Data format and structure

Data files in a netcdf format are stored under the directory specified by param_exp['expdir']. 
`nature.nc` contains the time series of variable X (with dimension `param_model['dimension']`), `obs.nc` contains observed X (possibly with missing grid points), and `assim.nc` has the time series of forecast (first guess) and analysis value of X for all ensemble members, and thier ensemble mean values.  

```
DATA/                    ### top directory 
├── exp1/                    ### experiment 1 with specific choices of f, h, b, and c (param_exp['expdir'])
│   ├── nature.nc                ### nature run                           
│   ├── spinup/                  ### initial condition
│   ├── obs_p8_010/              ### observation settings 1 (param_exp['obs_type'])
│   |   ├── obs.nc                   ### observation data                           
│   │   ├── assim_nocorr/assim.nc    ### letkf settings 1   (param_exp['da_type'])
│   │   ├── assim_linear/assim.nc    ### letkf settings 2 
|   │   ├── assim_Dense/assim.nc     ### letkf settings 3
│   │   ...
│   ├── obs_p6_010/              ### observation settings 2
│   |   ├── obs.nc                   ### observation data
│   │   ├── assim_nocorr/assim.nc    ### letkf settings 1
│   │   ├── assim_linear/assim.nc    ### letkf settings 2
│   │   ├── assim_Dense/assim.nc     ### letkf settings 3
│   │   ...
│  ...
├── exp2/                          ### experiment 2
...
```

### Data preparation

Prepare nature run and synthetic observation.  

```
python spinup_nature.py  # make initial condition 
python nature.py         # run the model
python obsmake.py        # create synthetic observation from the nature run
```

### Data assimilation cycle without bias correction

First, set `param_bc['bc_type'] = None` in param.py. 
Run a data assimilation cycle without bias correction to create time series of analysis and forecast for training.  

```
python spinup_model.py  # make initial ensemble for LETKF
python exp.py           # run data assimilation cycle with LETKF
```

### Bias correction training 

#### linear regression 

Edit and run `reg/calc_reg_loc.py` to calculate linear regression coefficients. 
```
cd reg
python ./calc_reg_loc.py
```
The resultant 'RMSE fix' is equivalent to the training loss. The coefficients are stored in a file like `coeffw_1.nc` in the same directory as `assim.nc`.  

#### neural networks 

The scripts for bias correction using neural networks are in the directory `tf`.
It utilizes [*Tensorflow*](https://www.tensorflow.org/) and the hyperparameter optimization tool [*optuna*](https://optuna.readthedocs.io/en/stable/index.html). You can choose to use optuna with the option `--use_optuna=1`.
When it is not used, training is performed only once with a fixed set of hyperparameters specified manually in `Exp.py`. 

The network architecture (number of recurrent and non-recurrent layers, and number of nodes in each layer) is specified in the following. 
 ```
### Exp.py
      if args.recurrent:
        plist['time_splits'] = args.time_splits
        print('\nNetwork is recurrent\n')
        plist['num_rnn_layers'] = 3
        plist['RNN_output'] = []
        plist['RNN_output'].append(10)
        plist['RNN_output'].append(8)
        plist['RNN_output'].append(4)
        plist['num_dense_layers'] = 1
        plist['dense_output'] = []
        plist['dense_output'].append(9)
        for i in range(plist['num_dense_layers'] - 1):
            plist['dense_output'].append(4)
      else:
        plist['time_splits'] = 1 
        print('\nNetwork is only dense\n')
        plist['num_rnn_layers'] = 0
        plist['RNN_output'] = []
        plist['num_dense_layers'] = 3
        plist['dense_output'] = []
        plist['dense_output'].append(21)
        plist['dense_output'].append(15)
        plist['dense_output'].append(12)

      plist['dense_output'].append(1)
```

Some of the parameters and settings are also specified explicitly in `Exp.py`.  
```
    plist['activation'] = 'relu'
    plist['d_activation'] = 'relu'
    plist['rec_activation'] = 'sigmoid'
    plist['l2_regu'] = 1e-5
    plist['l1_regu'] = 0.0
    plist['rnn_dropout'] = 0.0
    plist['rec_rnn_dropout'] = 0.0
```

Use `run.sh` to launch the training with `Exp.py`. Some important parameters are supposed to be set in `run.sh` as follows.  
```
loc="../DATA/exp1/obs_p8_010/assim_nocorr/assim.nc"    ### path to the netcdf file used for training data
id_exp="test"                                          ### experiment label
recurrent=0                                            ### recurrent or not
nntype="Dense"                                         ### neural network type: ["Dense", "Dense_single", "SimpleRNN", "GRU", "LSTM"]  
```

If `Exp.py` runs successfully, it will return training and validation loss values to the standard output.
The training result will be stored in the form of pickle and checkpoint files in a subdirectory specified by `id_exp` under the base directory `tf/n_experiment`.

### Data assimilation cycle with bias correction

Set parameters for bias correction in `param.py` accordingly and specify a different directory to store analysis data. 
```
param_exp['da_type'] = 'assim_linear'                 ### Output subdirectory

param_bc = {}
param_bc['bc_type'] = 'linear'     ### Bias correction type (None/'linear'/'linear_custom'/'tf')
param_bc['offline'] = 'true'      ### Offline bias correction ?
param_bc['alpha'] = 0.01          ### For online bias correction : not used 
param_bc['gamma'] = 0.0002        ### For online bias correction : not used
param_bc['path'] = param_exp['expdir'] + '/' + param_exp['obs_type'] + '/assim_nocorr/coeffw_1.nc' ### Linear regression coefficients data
param_bc['correct_step'] = None   ### Apply bias correction every dt instead of dt_assim ? 
param_bc['tf_expname'] = 'test'  ### Neural network coefficients data
```

When `param_bc['bc_type'] = 'linear'`, the first-order linear regression coefficients in `coeffw_1.nc` are supposed to be used. When it is `'linear_custom'`, the fourth-order coefficients `'coeffw_4.nc'` are used. `param_bc['bc_type'] = 'tf'` correcponds to bias correction using trained neural networks in `tf/n_experiment/param_bc['tf_expname']`. 

Run `exp.py` and create `assim.nc` in the separated directory `param_exp['da_type']`. 

### Extended ensemble forecasts

Run `forecast.py` to perform an extended forecast experiment from analyses data.

```
python forecast.py
```

By default, this launches ensemble forecasts initialized at 100 different initial time using the analysis ensemble in `assim.nc`. Each forecast time series is not output, but the average RMSEs of ensemble forecast compared to the nature run for each forecast is calculated and stored in `stats.nc` in the same directory with `assim.nc`.   


When nature run is to be used to initialize forecasts instead of analyses, create nature time series with fast variable output.

```
python spinup_model_full.py
python nature_full.py
```

Then run the scripts.

```
python forecast_from_nature.py
python forecast_from_nature_full.py
```
