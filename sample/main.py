import numpy as np
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


def average_model(motor_rpm):
    gear_ratio = 15.58
    wheel_radius = 0.228
    return np.mean(motor_rpm, axis=1) / gear_ratio / 60 * 2 * np.pi * wheel_radius

def normalizeData (inputData):
    for i in range(0,len(inputData[0])):
        mini = min(inputData[:,i])
        maxi = max(inputData[:,i])
        inputData[:,i]=(inputData[:,i]-mini-(maxi-mini)/2)/(maxi-mini)*2
    return inputData


def import_log(folder):
    channels = [
        "AMK_FL_Actual_velocity",
        "AMK_FR_Actual_velocity",
        "AMK_RL_Actual_velocity",
        "AMK_RR_Actual_velocity",
        "AMK_FL_Temp_Motor",
        "AMK_FR_Temp_Motor",
        "AMK_RL_Temp_Motor",
        "AMK_RR_Temp_Motor",
        "AMK_FL_Temp_Inverter",
        "AMK_FR_Temp_Inverter",
        "AMK_RL_Temp_Inverter",
        "AMK_RR_Temp_Inverter",
        "AMK_FL_Setpoint_positive_torque_limit",
        "AMK_FR_Setpoint_positive_torque_limit",
        "AMK_RL_Setpoint_positive_torque_limit",
        "AMK_RR_Setpoint_positive_torque_limit"
          ]
    cell_number = 140
    for i in range(0,cell_number):
        channels.append("BMS_Cell_Temperature_" + str(i))
    
    channels.append("BMS_Tractive_System_Power")
    print(len(channels))
    filenames = [join(folder, channel) + ".csv" for channel in channels]

    raw_data = csv_import.read_csv_files(filenames)
    data = csv_import.create_single_table(raw_data)

    return data


# Import endurance FSG
folder = "./data/endurance fsg"
data = import_log(folder)

print(data[:,0])
data = normalizeData(data)
input_indices = range(1,156)
print(input_indices)

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







# plt.figure()
# plt.plot(X_train[:,8])
# plt.title("blæ")
# plt.show()


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


batch_size = 256
data_gen_train = TimeseriesGenerator(X_train, Y_train,
                               length=recursive_depth,
                               batch_size=batch_size)

data_gen_test = TimeseriesGenerator(X_test, Y_test,
                               length=recursive_depth,
                               batch_size=batch_size)                            


model.fit_generator(data_gen_train, epochs=20)


y_train = model.predict_generator(data_gen_train)
y_test = model.predict_generator(data_gen_test)

plt.figure(1)
plt.plot(Y_train)
plt.plot(np.append(np.zeros(recursive_depth), y_train))

plt.figure(2)
plt.plot(Y_test)
plt.plot(np.append(np.zeros(recursive_depth), y_test))
plt.plot(average_model(X_test[1:5]))

plt.show()