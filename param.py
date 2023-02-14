###
### Common experiment parameter
###

param_model = {}
param_model['dimension'] = 8
param_model['dimension_coupled'] = 256
param_model['forcing'] = 20
param_model['dt'] = 0.005
param_model['dt_coupled'] = 0.001  # dt/c
param_model['h'] = 1
param_model['b'] = 10
param_model['c'] = 4

param_letkf = {}
param_letkf['obs_error'] = 0.10
param_letkf['obs_number'] =8
param_letkf['localization_length_scale'] = 1
param_letkf['localization_length_cutoff'] = 1
param_letkf['inflation_factor'] = 1.0
param_letkf['additive_inflation_factor'] = 0.25
param_letkf['missing_value'] = -9.99e8

param_exp = {}
param_exp['exp_length'] = 30000
param_exp['ensembles'] = 10
param_exp['expdir'] = './DATA/coupled_A13'
param_exp['obs_type'] = 'obs_p8_010'
param_exp['da_type'] = 'nocorr'
param_exp['dt_nature'] = 0.05
param_exp['dt_obs'] = 0.05
param_exp['dt_assim'] = 0.05
param_exp['spinup_length'] = 2000

param_bc = {}
param_bc['bc_type'] = None
param_bc['alpha'] = 0.01
param_bc['gamma'] = 0.0002
param_bc['offline'] = 'true'
param_bc['path'] = param_exp['expdir'] + '/' + param_exp['obs_type'] + '/nocorr/coeffw_4.nc'
param_bc['correct_step'] = None
param_bc['tf_expname'] = param_exp["obs_type"]+'/Dense_single'

