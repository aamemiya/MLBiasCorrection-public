from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import numpy as np
np.random.seed(5)
import pickle
import helperfunctions as helpfunc
import os
import shutil


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
	
