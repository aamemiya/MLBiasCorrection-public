# MLBiasCorrection

This repository contains the codes for a paper by [Amemiya, Shlok, and Miyoshi](https://doi.org/10.1029/2022MS003164) (2023), *Journal of Advances in Modeling Earth Systems*.

## Acknowledgement
The main part of the Lorenz96 LETKF code is originally developed by Shigenori Otsuka ([https://github.com/otsuka-shigenori/da_demo](https://github.com/otsuka-shigenori/da_demo))  
The codes with Keras Tensorflow is developed by Shlok Mohta. 

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
param_letkf['localization_length_scale'] = 3  ### Width scale for localization
param_letkf['localization_length_cutoff'] = 8 ### Cutoff scale for localization  
param_letkf['inflation_factor'] = 2.1         ### Multiplicative inflation factor
param_letkf['missing_value'] = -9.99e8

param_exp = {}
param_exp['exp_length'] = 30000               ### Number of DA steps
param_exp['ensembles'] = 10                   ### Ensemble size
param_exp['expdir'] = './DATA/coupled_A13'    ### Output directory
param_exp['obs_type'] = 'obs_010'             ### Output subdirectory
param_exp['da_type'] = 'test'                 ### Output subdirectory
param_exp['dt_nature'] = 0.05                 ### dt for nature run output
param_exp['dt_obs'] = 0.05                    ### dt for obs output
param_exp['dt_assim'] = 0.05                  ### dt for DA  output
param_exp['spinup_length'] = 2000             ### Steps to spinup

```

### Data preparation

```
python spinup_nature.py 
python spinup_model.py 
python nature.py 
python obsmake.py 
```

### Data assimilation cycle without bias correction

Set `param_bc['bc_type'] = None` in param.py and run `exp.py`. 
```
python exp.py
```

### Bias correction calculation

Use scripts in `tf` directory. 
 ```
cd tf
./run.sh

```

### Data assimilation cycle with bias correction

Set parameters for bias correction in param.py. 
```
param_bc = {}
param_bc['bc_type'] = 'tf'        ### Bias correction type (None/'linear'/'linear_custom'/'tf')
param_bc['offline'] = 'true'      ### Offline bias correction ?
param_bc['alpha'] = 0.01          ### For online bias correction : not used 
param_bc['gamma'] = 0.0002        ### For online bias correction : not used
param_bc['path'] = param_exp['expdir'] + '/' + param_exp['obs_type'] + '/nocorr/coeffw_4.nc' ### Linear regression coefficients data
param_bc['correct_step'] = None   ### Apply bias correction every dt instead of dt_assim ? 
param_bc['tf_expname'] = param_exp["obs_type"]+'/Dense_single'  ### Neural network coefficients data

```
Then run `exp.py`.
```
python exp.py
```

### Forecast experiment

Run `forecast.py` to perform an extended forecast experiment from analyses data.

```
python forecast.py
```

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
