from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import numpy as np
np.random.seed(5)
import pickle
import helperfunctions as helpfunc
import os
import shutil

import mlflow
mlflow.set_tracking_uri('file:/home/mshlok/MLBiasCorrection/Python_files/mlruns_plot')
print(mlflow.tracking.get_tracking_uri())

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 7,
        }

def cal_rmse(x, y):
    return np.sqrt(np.mean(np.power(x - y, 2)))

def scatter_plot(plot_variable, variable_num, x1_label, x2_label, y_label, directory):

    x1, x2, y = plot_variable
    fig, ax = plt.subplots()

    ax.set_title('Variable {}'.format(variable_num), fontdict = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16
        })

    ax.set_xlabel(x2_label)
    ax.plot(x2, np.zeros(len(x2)), c = 'g', linewidth = 1)
    ax.scatter(x2, x2-y, s=10, marker='o', c = 'b', label= x2_label + ' - ' + y_label)
    ax.scatter(x2, x1-y, s=8, marker='*', c = 'r', label=x1_label + ' - ' + y_label)
    ax.text(x2.min() + 0.5, max((x1-y).max(), (x2-y).max()) - 0.03, 'Corrected RMSE: ' + str(cal_rmse(x1, y))[:6] + '\n' + 'Biased RMSE: ' + str(cal_rmse(x2, y))[:6], fontdict = font, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1', alpha = 0.5))
    ax.grid()
    ax.legend(loc = 1, prop={'size': 6})

    img_name = directory + '/scatter_plot_variable_{}.png'.format(variable_num)
    print('Saving image file: {}'.format(img_name))
    fig.savefig(img_name, format= 'png', dpi = 600)

def line_plot(plot_variable, variable_num, directory):

    model_forecast, forecast,  analysis = plot_variable
    time = np.arange(len(analysis)) + 1
    
    fig, ax = plt.subplots()

    ax.set_title('Variable {}'.format(variable_num), fontdict = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16
        })

    ax.set_ylabel('X')
    ax.set_xlabel('Time')
    ax.plot(time, analysis, 'g-', linewidth = 2 , label = 'Analysis')
    ax.plot(time, forecast, 'y--',  linewidth = 1, label = 'Forecast')
    ax.plot(time, model_forecast, 'k:', linewidth = 1, label = 'Corrected forecast')

    ax.text(5, analysis.max() - 1, 'Corrected RMSE: ' + str(cal_rmse(model_forecast, analysis))[:6] + '\n' + 'Biased RMSE: ' + str(cal_rmse(forecast, analysis))[:6], fontdict = font, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1', alpha = 0.5))

    ax.legend(loc = 1, prop={'size': 6})
    img_name = directory + '/line_plot_variable_{}.png'.format(variable_num)
    print('Saving image file: {}'.format(img_name))
    fig.savefig(img_name, format= 'png', dpi = 600)

def plot_func(plist, c_forecast, analysis, forecast, c_rmse, rmse, exp_name):
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name = plist['experiment_name']):
        #Randomly select five variables for plotting
        random_variables = np.random.randint(low = 0, high=analysis.shape[1], size=5)
        image_dir = (plist['experiment_dir'] + '/images')
        if not(os.path.exists(image_dir)):
            os.mkdir(image_dir)

        for i in random_variables:
            
            scatter_plot((c_forecast[:,i], forecast[:,i], analysis[:,i]), i, 'Corrected_forecast', 'Biased_Forecast', 'Analysis', image_dir)
            
            line_plot((c_forecast[:,i], forecast[:,i], analysis[:,i]), i, image_dir)
        mlflow.log_metric('C_RMSE', c_rmse)
        mlflow.log_metric('B_RMSE', rmse)
        mlflow.log_artifacts(image_dir)
        mlflow.log_param('RNN', ', '.join(str(i) for i in plist['RNN_output']))
        mlflow.log_param('Dense', ', '.join(str(i) for i in plist['dense_output']))
        mlflow.log_param('Timesteps', str(plist['time_splits']))
        mlflow.log_param('Locality', str(plist['locality']))

    shutil.rmtree(image_dir)

def plot_func_amemiya(c_forecast, analysis, forecast, exp_name, run_name, order, locality):
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name = run_name):
        
        #Randomly select five variables for plotting
        random_variables = np.random.randint(low = 0, high=analysis.shape[1], size=5)
        image_dir = ('/images')
        if not(os.path.exists(image_dir)):
            os.mkdir(image_dir)

        for i in random_variables:
        
            scatter_plot((c_forecast[:,i], forecast[:,i], analysis[:,i]), i, 'Corrected_forecast', 'Biased_Forecast', 'Analysis', image_dir)
            
            line_plot((c_forecast[:,i], forecast[:,i], analysis[:,i]), i, image_dir)

        mlflow.log_metric('C_RMSE', cal_rmse(c_forecast, analysis))
        mlflow.log_metric('B_RMSE', cal_rmse(forecast, analysis))
        mlflow.log_artifacts(image_dir)
        mlflow.log_param('Order', order)
        mlflow.log_param('Locality', locality)

    shutil.rmtree(image_dir)

def plot_func_amemiya_ef(rmse_plot, sprd_plot, time, dadir, exp_name = 'Ext_Forecast', run_name = 'test'):
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name = run_name):
        
        image_dir = ('/amemiya/images')
        if not(os.path.exists(image_dir)):
            os.mkdir(image_dir)

        ntime=len(time)
        doubling_time=0.2*2.1 ### 2.1 day
        smp_e=1
        nplt=40
        plt.figure()
        plt.grid(True)
        plt.yscale('log')

        for i in range(len(dadir)):
            plt.plot(time[:nplt], rmse_plot[i][:nplt],label=dadir[i])

        plt.legend(loc = 1, prop={'size': 10})
        plt.xlabel('TIME')
        plt.ylabel('RMSE')
        plt.xlim()
        plt.ylim()
        plt.savefig(image_dir + '/rmse.png', format= 'png', dpi = 600)

        mlflow.log_artifacts(image_dir)

    shutil.rmtree(image_dir)


def plot_history( epoch_nums, loss, val_loss ,normalized=True):
		"""
		plot the history of the traning
		:param normalized: if True normalize both validation/traning loss to 1 fort he first eppoch.
		:return: the matplotlib figure
		"""
		fig,ax = plt.subplots()
		S1 = loss[2] if normalized else 1
		S2 = val_loss[2] if normalized else 1
		ax.semilogy(np.array(loss[2:])/S1, color='gray', linewidth=2, label='train')
		ax.semilogy(np.array(val_loss[2:])/S2, color='black', linewidth=2, label='test')
		ax.legend()
		return fig
	
