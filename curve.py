import numpy as np

import data_storage

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
else:
    import matplotlib

from matplotlib import pyplot as plt

from scipy.optimize import curve_fit

data = data_storage.load_data_set("training")
input_indices = range(1,156)


def tensor_summary(t):
    print("Shape {}, min: {}, max {}".format(t.shape, np.min(t), np.max(t)))


t_shift = 0
t = data[:,-1]
#t = np.append(np.zeros((t_shift,)), t[:-t_shift])
pos_power = t > 40
neg_power = t < 0
t = np.reshape(t, (t.shape[0], 1))



# pos_power
torque = data[:,13:17] / 20
training_data = pos_power

# neg_power
# torque = data[:,17:21]
# training_data = neg_power

rpm = data[:,1:5] / 20000

motor_power =  rpm * torque

rpm = rpm[training_data,:]
motor_power = motor_power[training_data,:]
torque = torque[training_data,:]
t = t[training_data,:]


def lin_curve(x, w, b):
    return x @ w.T + b

def d_lin_curve_d_w(x):
    return x

def d_lin_curve_d_b(x):
    return np.ones((x.shape[0],1))

def loss(t,y):
    return np.sum(np.power((t - y), 2)) / t.shape[0]

def d_loss_d_y(t, y):
    return -2 * (t - y) / t.shape[0]

epochs = 1000
batch_size = 10

b = 0

print("motor_power.shape=", motor_power.shape)
print("rpm.shape=", rpm.shape)
print("torque.shape=", torque.shape)

#x = np.hstack((motor_power, rpm, torque))
x = motor_power
w = np.zeros((1, x.shape[1]))
#w = np.array([[ 53.07791556, 48.61226578, 57.80308425, 59.14017956]])
lr = 1e-5

tensor_summary(x)

for epoch in range(epochs):
    print("epoch", epoch)
    y = lin_curve(x,w,b)
    print("loss", loss(t,y))
    xt = np.hstack((x,t))
    np.random.shuffle(xt)
    x_train = xt[:,:-1]
    t_train = xt[:,-1, np.newaxis]
    for batch in range(x.shape[0] // batch_size):
        # print("Epoch ", epoch, " Batch ", batch)
        # print(w)
        batch_begin = batch_size * batch
        batch_end = batch_size * (batch + 1)

        x_batch = x_train[batch_begin:batch_end,:]
        # print("x_batch.shape=", x_batch.shape)

        t_batch = t_train[batch_begin:batch_end,:]
        # print("t_batch.shape=", t_batch.shape)

        y_batch = lin_curve(x_batch, w, b)
        # print("y_batch.shape=", y_batch.shape)

        d_l_d_y = d_loss_d_y(t_batch, y_batch)
        # print("d_l_d_y.shape=", d_l_d_y.shape)

        d_l_d_w = d_lin_curve_d_w(x_batch)
        # print("d_l_d_w.shape=", d_l_d_w.shape)

        w -= lr * (d_l_d_w.T @ d_l_d_y).T
        b -= lr * d_l_d_y.T @  d_lin_curve_d_b(x_batch)

print("w", w)
print("b", b)



plt.figure()
plt.plot(rpm)
plt.show()

plt.figure()
plt.plot(motor_power)
plt.show()

plt.figure()
plt.plot(t)
plt.plot(lin_curve(x,w,b))

plt.show()