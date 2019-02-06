import numpy as np
from os import listdir
from os.path import join
from numpy import genfromtxt
import glob
import matplotlib.pyplot as plt

import analyze_csv as csv_import

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.utils import plot_model






def average_model(motor_rpm):
    gear_ratio = 15.58
    wheel_radius = 0.228
    return np.mean(motor_rpm, axis=1) / gear_ratio / 60 * 2 * np.pi * wheel_radius


def import_log(folder):
    channels = [
        "SBS_F1_Steering_Angle",
        "AMK_FL_Actual_velocity",
        "AMK_FR_Actual_velocity",
        "AMK_RL_Actual_velocity",
        "AMK_RR_Actual_velocity",
        "AMK_FL_Torque_current",
        "AMK_FR_Torque_current",
        "AMK_RL_Torque_current",
        "AMK_RR_Torque_current",
        "INS_Vx"
                ]
    filenames = [join(folder, channel) + ".csv" for channel in channels]

    raw_data = csv_import.read_csv_files(filenames)
    data = csv_import.create_single_table(raw_data)

    return data



input_indices = range(1,10)


# Import endurance FSG
folder = "./data/endurance fsg"
data = import_log(folder)

X = data[:,input_indices]
Y = data[:,-1]

X_train = X[4000:54000]
Y_train = Y[4000:54000]

X_test = X[67000:110000]
Y_test = Y[67000:110000]


# Import endurance FSS
folder = "./data/FSS_endurance"
data = import_log(folder)

X = data[:,input_indices]
Y = data[:,-1]

X_train = np.append(X_train, X[69000:107000], axis=0)
Y_train = np.append(Y_train, Y[69000:107000])

X_test = np.append(X_test, X[132000:180000], axis=0)
Y_test = np.append(Y_test, Y[132000:180000], axis=0)






normalizer = np.array([ 230.0, 23000, 23000, 23000, 23000, 100000, 100000, 100000, 100000])
X_train = X_train  / normalizer
X_test = X_test / normalizer

Y_train = Y_train / 30
Y_test = Y_test / 30

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




batch_size = 200
data_gen_train = TimeseriesGenerator(X_train, Y_train,
                               length=recursive_depth,
                               batch_size=batch_size)

data_gen_test = TimeseriesGenerator(X_test, Y_test,
                               length=recursive_depth,
                               batch_size=batch_size)                            


model.fit_generator(data_gen_train, epochs=7)



plot_model(model, to_file='model.png')


y_train = model.predict_generator(data_gen_train)
y_test = model.predict_generator(data_gen_test)

plt.figure()
plt.plot(Y_train)
plt.plot(np.append(np.zeros(recursive_depth), y_train))

plt.figure()
plt.plot(Y_test)
plt.plot(np.append(np.zeros(recursive_depth), y_test))
plt.plot(average_model(X_test[1:5]))

plt.show()