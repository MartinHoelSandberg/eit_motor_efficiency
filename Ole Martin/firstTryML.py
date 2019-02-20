import numpy as np
"""
from os import listdir
from os.path import join
from numpy import genfromtxt
import glob

import analyze_csv as csv_import

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
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
"""




# normalisere dataen til -1, 1.
def normalizeData (inputData):
    for i in range(0,len(inputData[0])):
        mini = min(inputData[:,i])
        maxi = max(inputData[:,i])
        inputData[:,i]=(inputData[:,i]-mini-(maxi-mini)/2)/(maxi-mini)*2
    return inputData



# hente test og treningsdata fra Marius sin logg
import data_storage #Velger Marius sin kode 
data = data_storage.load_data_set("train") #henter dataen inn i data


# Utfør normalisering
X_train = normalizeData(X_train)
Y_train = normalizeData(Y_train)
X_test = normalizeData(X_test)
Y_test = normalizeData(Y_test)

"""
recursive_depth = 3
model = Sequential([
    Dense(10, input_shape=(recursive_depth,len(input_indices),)),
    # Activation('relu'),
    LSTM(3),
    # Activation('relu'),
    Dense(5),
    # Activation('relu'),
    Dense(1),
    Activation('relu')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])


batch_size = 200
data_gen_train = TimeseriesGenerator(X_train, Y_train,
                               length=recursive_depth,
                               batch_size=batch_size)

data_gen_test = TimeseriesGenerator(X_test, Y_test,
                               length=recursive_depth,
                               batch_size=batch_size)                            


model.fit_generator(data_gen_train, epochs=7)


y_train = model.predict_generator(data_gen_train)
y_test = model.predict_generator(data_gen_test)

# dispe resultat
"""

def plot(Y_train,powerPredTrain,Y_test,PowerPredTest,recursive_depth):
    plt.figure(1)
    plt.plot(Y_train)
    plt.plot(np.append(np.zeros(recursive_depth), powerPredTrain))
    plt.title(Power Prediction based on Training Data)
    plt.xlabel(Time)
    plt.ylabel(Power)

    plt.figure(2)
    plt.plot(Y_test)
    plt.plot(np.append(np.zeros(recursive_depth), PowerPredTest))
    plt.title(Power Prediction run on Test Data)
    plt.xlabel(Time)
    plt.ylabel(Power)

    plt.show()
    return 0
