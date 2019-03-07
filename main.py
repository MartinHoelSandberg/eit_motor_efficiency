import numpy as np
import data_storage
from os import listdir
from os.path import join
from numpy import genfromtxt
import glob

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.utils import plot_model

from sys import platform as sys_pf

from scipy.optimize import curve_fit
  

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
else:
    import matplotlib

from matplotlib import pyplot as plt


def normalizeData (inputData):
    for i in range(0,len(inputData[0])):
        mini = np.min(inputData[:,i])
        maxi = np.max(inputData[:,i])
        inputData[:,i]=(inputData[:,i]-mini-(maxi-mini)/2)/(maxi-mini)*2
    return inputData

data = data_storage.load_data_set("training")
input_indices = range(1,156)
data=normalizeData(data)
X_train = data[:,input_indices]
Y_train = data[:,-1]

data = data_storage.load_data_set("test")
input_indices = range(1,156)
data=normalizeData(data)
X_test = data[:,input_indices]
Y_test = data[:,-1]

# Import endurance FSG
"""folder = "./data/endurance fsg"
data = import_log(folder)
data = normalizeData(data)
input_indices = range(1,156)

X = data[:,input_indices]
Y = data[:,-1]

X_train = X[4000:54000]
Y_train = Y[4000:54000]

X_test = X[67000:110000]
Y_test = Y[67000:110000]


# Import endurance Fss
folder = "./data/FSS_endurance"
data = import_log(folder)
data=normalizeData(data)
input_indices = range(1,156)

X = data[:,input_indices]
Y = data[:,-1]

X_train = np.append(X_train, X[69000:107000], axis=0)
Y_train = np.append(Y_train, Y[69000:107000])

X_test = np.append(X_test, X[132000:180000], axis=0)
Y_test = np.append(Y_test, Y[132000:180000], axis=0)
"""

def curve(x, a, b):
    return np.sum(x[:,1:5] * a * x[:,13:17], axis=1) + b

plt.figure(1)
plt.plot(X_train[:,0:4])
plt.show()

plt.figure(1)
plt.plot(Y_train)
plt.plot(curve(X_train, 0.5, 0))
plt.show()


# KERAS stuff
recursive_depth = 3
model = Sequential([
    Dense(10, input_shape=(recursive_depth,len(input_indices),)),
    # Activation('relu'),
    LSTM(4),
    # Activation('relu'),
    Dense(4),
    # Activation('relu'),
    Dense(1),
    Activation('tanh')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])


batch_size = 256
data_gen_train = TimeseriesGenerator(X_train, Y_train,
                               length=recursive_depth,
                               batch_size=batch_size)

data_gen_test = TimeseriesGenerator(X_test, Y_test,
                               length=recursive_depth,
                               batch_size=batch_size)                            


model.fit_generator(data_gen_train, epochs=2)


y_train = model.predict_generator(data_gen_train)
y_test = model.predict_generator(data_gen_test)

plt.figure(1)
plt.plot(Y_train)
plt.plot(np.append(np.zeros(recursive_depth), y_train))
plt.plot(curve(X_train, 0.5, 0))

plt.figure(2)
plt.plot(Y_test)
plt.plot(np.append(np.zeros(recursive_depth), y_test))


plt.show()