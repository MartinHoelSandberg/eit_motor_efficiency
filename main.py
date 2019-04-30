import numpy as np
import data_storage
from os import listdir
from os.path import join
from numpy import genfromtxt
import glob
from neural_network_config import NeuralNetworkConfig

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.utils import plot_model

from sys import platform as sys_pf
  

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
else:
    import matplotlib

from matplotlib import pyplot as plt

# Combines positive and negative torque limit to the same channels
def torque_setpoint (data):
    for i in range(0, len(data[:,0])):
        if data[i, 1] > 0:
            data[i, 5] = data[i, 1]
        if data[i, 2] > 0:
            data[i, 6] = data[i, 2]
        if data[i, 3] > 0:
            data[i, 7] = data[i, 3]
        if data[i,4] > 0:
            data[i, 8] = data[i, 4]
    new_data = data[:,0, np.newaxis]
    new_data = np.append(new_data, data[:,5:], axis = 1)
    return new_data

# Finds the extremes of training data to use as normaliser -1 to 1
def normalizeData (inputData):
    extreme = np.zeros((2, len(inputData[0])))
    for i in range(0, len(inputData[0])):
        extreme[0, i] = min(inputData[:,i]) * 1.2
        extreme[1, i] = max(inputData[:,i]) * 1.2
        #mini = min(inputData[:,i])
        #maxi = max(inputData[:,i])
        #inputData[:,i]=(inputData[:,i]-mini-(maxi-mini)/2)/(maxi-mini)*2
    return extreme

# Import training and test data
input_indices = range(1, 154)

data = data_storage.load_data_set("training")
data = torque_setpoint(data)
extreme = normalizeData(data)
for i in range(1, len(data[0])):
    # Normalises training data
    data[:,i] = (data[:,i] - extreme[0,i] - (extreme[1, i] - extreme[0, i]) / 2) / (extreme[1, i] - extreme[0, i]) * 2
X_train = data[:,input_indices]
Y_train = data[:,-1]

data = data_storage.load_data_set("test")
data = torque_setpoint(data)
for i in range(1, len(data[0])):
    # Normalises test data with extremes from training data
    data[:,i]=(data[:,i] - extreme[0, i] - (extreme[1, i] - extreme[0, i]) / 2) / (extreme[1, i] - extreme[0, i]) * 2
X_test = data[:,input_indices]
Y_test = data[:,-1]

# KERAS stuff
nnModel = NeuralNetworkConfig(input_indices)

data_gen_train = TimeseriesGenerator(X_train, Y_train,
                                        length = nnModel.recursive_depth,
                                        batch_size = nnModel.batch_size)

data_gen_test = TimeseriesGenerator(X_test, Y_test,
                                        length = nnModel.recursive_depth,
                                        batch_size = nnModel.batch_size)                            

nnModel.architecture.fit_generator(data_gen_train, epochs = nnModel.epochs)

y_train = nnModel.architecture.predict_generator(data_gen_train)
y_test = nnModel.architecture.predict_generator(data_gen_test)

np.savetxt("predikert.csv", y_test, delimiter=",")
np.savetxt("faktisk.csv", y_test, delimiter=",")

plt.figure(1)
plt.plot(Y_train)
plt.plot(np.append(np.zeros(nnModel.recursive_depth), y_train))

plt.figure(2)
plt.plot(Y_test)
plt.plot(np.append(np.zeros(nnModel.recursive_depth), y_test))

plt.show()