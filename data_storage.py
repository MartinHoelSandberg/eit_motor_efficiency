import numpy as np
from scipy.interpolate import interp1d
from os import path
from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
else:
    import matplotlib

from matplotlib import pyplot as plt

data_dir = "./data/"

def read_csv_files(filenames):
    data = [None for i in range(len(filenames))]
    for i in range(len(filenames)):
        filename = filenames[i]
        t = np.genfromtxt(filename, delimiter=',')
        t = t[~np.isnan(t).any(axis=1)]
        data[i] = t
    
    return data

def get_common_time(data):
    time = np.copy(data[0][:,0])

    for channel in data:
        # Bounding common time by time of each channel
        time = time[ time > channel[0,0]  ]
        time = time[ time < channel[-1,0] ]

    return time

def create_single_table(data):
    # First channel is used for interpolating all other channels

    time = get_common_time(data)
    n_samples = time.shape[0]
    n_channels = len(data)
    result = np.zeros(( n_samples, n_channels + 1))

    result[:,0] = time

    for i in range(len(data)):
        time_data = data[i][:,0]
        value_data = data[i][:,1]
        f = interp1d(time_data, value_data, kind="zero")
        result[:,i + 1] = f(time)

    return result

def add_to_dataset():
    
    log_name = input("enter log name: ")
    folder = path.join(data_dir, log_name)

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

    filenames = [path.join(folder, channel) + ".csv" for channel in channels]

    print("Reading log ...")
    data = read_csv_files(filenames)
    print("Log read!")
    print("Creating single table and interpolating")
    data = create_single_table(data)
    print("Finished interpolating, table created")

    while True:
        channel_index = int(input("Select channel index to plot (0 for exit): "))
        if channel_index < 1:
            break
        
        plt.figure()
        plt.plot(data[:,0], data[:,channel_index])
        user_coord = plt.ginput(2) # Shows plot and blocks

        start_x = user_coord[0][0]
        end_x = user_coord[1][0]

        start_i = np.argmin(abs(data[:,0] - start_x))
        end_i = np.argmin(abs(data[:,0] - end_x))

        trimmed_data = data[start_i:end_i,:]

        plt.figure()
        plt.plot(trimmed_data[:,0], trimmed_data[:,channel_index])
        plt.show()

        looks_good = input("Does this look good? y, else drop: ")

        if looks_good != "y":
            continue
        
        data_set_name = input("Name of data set: ")
        data_set_path = path.join(data_dir, data_set_name + ".npy")

        
        if path.exists(data_set_path):
            data_set = np.load(data_set_path)
            data_set = np.append(data_set, trimmed_data, axis=0)
        else:
            data_set = trimmed_data
        
        np.save(data_set_path, data_set)
    
def load_data_set(name):
    data_set_path = path.join(data_dir, name + ".npy")
    data_set = np.load(data_set_path)
    return data_set

if __name__ == '__main__':
    add_to_dataset()