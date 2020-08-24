import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import pickle

"""
5 Graphics Builder
The script provides graphing for a comparative assessment of the accuracy of trained models
"""

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plot_it(plt, ax, fig, legend_lines, legend_captions, title, ylabel, xlabel, yStart, yStop,xStart, xStop):
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(yStart, yStop)
    ax.set_xlim(xStart, xStop)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='grey', linestyle='-')
    ax.grid(b=True, which='minor', color='silver', linestyle='-', alpha=0.5)


model_titles = [
    ['deep_fusion_4_june4', 'GeoSystemNet (EuroSAT extended)'],
    ['simple_cnn_june4', 'CNN with 2 layers (EuroSAT)'],
    ['resnet_june4', 'ResNet50 (EuroSAT)'],
]

i = 0
legend_lines = []
legend_captions = []


fig1 = plt.figure()
ax1 = plt.subplot(111)

for mdl in model_titles:
    model_experimental_data = pickle.load(open('models_trained/model_'+mdl[0]+'.data', "rb"))

    history_accuracy = []
    history_val_accuracy = []
    history_loss = []
    history_val_loss = []
    for data_instance in model_experimental_data:
        history_accuracy.append (data_instance['history_accuracy'])
        history_val_accuracy.append (data_instance['history_val_accuracy'])

        history_loss.append (data_instance['history_loss'])
        history_val_loss.append (data_instance['history_val_loss'])

    acc_mean_data = np.mean(100*np.array(history_accuracy),0)
    acc_std_data  = np.std(100*np.array(history_accuracy),0)
    acc_val_mean_data = np.mean(100*np.array(history_val_accuracy),0)
    acc_val_std_data  = np.std(100*np.array(history_val_accuracy),0)

    loss_mean_data = np.mean(100*np.array(history_loss),0)
    loss_std_data  = np.std(100*np.array(history_loss),0)
    loss_val_mean_data = np.mean(100*np.array(history_val_loss),0)
    loss_val_std_data  = np.std(100*np.array(history_val_loss),0)

    x = range(1, len(data_instance['history_accuracy'])+1)

    acc_val_mean_data = scipy.signal.savgol_filter(acc_val_mean_data, 9, 1)
    acc_val_std_data = scipy.signal.savgol_filter(acc_val_std_data, 9, 1)

    k = 1.3

    ax1.plot(x, acc_val_mean_data, color='C'+str(i), label=mdl[1], linestyle='--')

    ax1.plot(x, acc_val_mean_data-k*acc_val_std_data, color='C'+str(i), linestyle=':')
    ax1.plot(x, acc_val_mean_data+k*acc_val_std_data, color='C'+str(i), linestyle=':')

    i = i + 1


plot_it(plt, ax1, fig1, legend_lines, legend_captions, '', 'Accuracy', 'Epoch', 0, 100, 1, len(data_instance['history_accuracy']))
plt.show()
fig1.savefig('accuracy.png')
