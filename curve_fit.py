import numpy as np
import data_storage
import matplotlib.pyplot as plt

PI = 3.1415926
nWheels = 4
wheelOffset = 4

channels = [
    "AMK_FL_Actual_velocity",
    "AMK_FR_Actual_velocity",
    "AMK_RL_Actual_velocity",
    "AMK_RR_Actual_velocity",
    "AMK_FL_Setpoint_positive_torque_limit",
    "AMK_FR_Setpoint_positive_torque_limit",
    "AMK_RL_Setpoint_positive_torque_limit",
    "AMK_RR_Setpoint_positive_torque_limit",
    "AMK_FL_Setpoint_negative_torque_limit",
    "AMK_FR_Setpoint_negative_torque_limit",
    "AMK_RL_Setpoint_negative_torque_limit",
    "AMK_RR_Setpoint_negative_torque_limit",
    "BMS_Tractive_System_Power"
]

for i in range(len(channels)):
    channels[i] = "./data/FSS_endurance/" + channels[i] + ".csv"

print("Loading channels from csv...")
data = data_storage.read_csv_files(channels)
table = data_storage.create_single_table(data)
print("Successfully Loaded the Dataset")

print("Calculating power estimates...")
powerLossTemp = []
bmsList = []
for i in range(table.shape[0]):
    
    bms = table[i][-1] * 1000 # Converting from kWh -> W
    
    if bms >= 0:
        # Estimates power for each wheel for a given moment. 
        estimatedPower = 0
        for j in range(nWheels):
            # Estimated Power = Angular Velocity * Torque // In an ideal world
            angularVelocity = table[i][j + 1] * (2 * PI / 60)
            positiveTorque = table[i][j + wheelOffset + 1]
            negativeTorque = table[i][j + (wheelOffset * 2) + 1]
            
            # Should not estimate power if KERS is in use (negative torque)
            #if negativeTorque == 0:
            estimatedPower += angularVelocity * (positiveTorque)# + negativeTorque)

        powerLossTemp.append(bms - estimatedPower)
    else:
        powerLossTemp.append(0)

    bmsList.append(bms)

powerLoss = []
clusterSize = 500
tmpSum = 0
for i in range(len(powerLossTemp)):
    tmpSum += powerLossTemp[i]
    if i % clusterSize == 0 and i != 0:
        powerLoss.append(tmpSum / clusterSize)
        tmpSum = 0

maxi = max(powerLoss)
mini = min(powerLoss)

print(powerLoss)

print(maxi)
print(mini)
print(sum(powerLoss) / len(powerLoss))

plt.plot(powerLoss)
#plt.plot(bmsList)
plt.ylabel('some numbers')
plt.show()