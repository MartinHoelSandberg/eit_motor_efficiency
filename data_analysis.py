import numpy as np
import data_storage
import matplotlib.pyplot as plt
from collections import OrderedDict

###########################################
#               Parameters                #
###########################################

PI = 3.1415926
nWheels = 4
maxDeviationPercentage = 100
minDeviationPercentage = -100

clusterSize = 50

channels = [
    "AMK_FL_Actual_velocity",
    "AMK_FR_Actual_velocity",
    "AMK_RL_Actual_velocity",
    "AMK_RR_Actual_velocity",
    "AMK_FL_Torque_current",
    "AMK_FR_Torque_current",
    "AMK_RL_Torque_current",
    "AMK_RR_Torque_current",
    "AMK_FL_Temp_Motor",
    "AMK_FR_Temp_Motor",
    "AMK_RL_Temp_Motor",
    "AMK_RR_Temp_Motor",
    "BMS_Tractive_System_Power"
]

datasets = {
    "fss_endurance" : "FSS_endurance",
    "fsg_endurance" : "endurance fsg"
}

dataset = datasets["fsg_endurance"]

###########################################
#           Utility Functions             #
###########################################

def mean(someList):
    total = 0
    for a in someList:
        total += float(a)
    mean = total/len(someList)
    return mean

def standDev(someList):
    listMean = mean(someList)
    dev = 0.0
    for i in range(len(someList)):
        dev += (someList[i]-listMean)**2
    dev = dev**(1/2.0)
    return dev

def correlCo(someList1, someList2):
    # First establish the means and standard deviations for both lists.
    xMean = mean(someList1)
    yMean = mean(someList2)
    xStandDev = standDev(someList1)
    yStandDev = standDev(someList2)
    # r numerator
    rNum = 0.0
    for i in range(len(someList1)):
        rNum += (someList1[i]-xMean)*(someList2[i]-yMean)

    # r denominator
    rDen = xStandDev * yStandDev

    r =  rNum/rDen
    return r

def estimatePowerLoss(table, bms):
    # Estimates power for each wheel for a given moment. 
    estimatedPower = 0
    for j in range(nWheels):
        # Estimated Power = Angular Velocity * Torque // In an ideal world
        angularVelocity = table[i][j + 1] * (2 * PI / 60)
        positiveTorque = table[i][j + nWheels + 1] * 0.26 / 1000

        estimatedPower += angularVelocity * positiveTorque

    return bms - estimatedPower


def summateInverterTemp(table):
    estimatedTemp = 0
    for j in range(nWheels):
        estimatedTemp = table[i][j + (nWheels * 2) + 1]
    return estimatedTemp

def calculateDeviationPercentage(powerloss, bms, prePowerLoss):
    
    deviationPercentage = powerloss / bms
    if deviationPercentage > maxDeviationPercentage or deviationPercentage < minDeviationPercentage:
        deviationPercentage = prePowerLoss
    
    return deviationPercentage

def clusterArrays(arraysToCluster, rows):
    clusteredArrays = []
    for value in arraysToCluster.values():
        clusteredArray = []
        clusterSum = 0
        for i in range(rows):
            clusterSum += value[i]
            
            if i % clusterSize == 0 and i != 0:
                clusterAvg = clusterSum / clusterSize
                clusteredArray.append(clusterAvg)
                clusterSum = 0
        clusteredArrays.append(clusteredArray)
    
    return clusteredArrays

###########################################
#                   CODE                  #
###########################################

for i in range(len(channels)):
    channels[i] = "./data/" + dataset + "/" + channels[i] + ".csv"

print("Loading channels from csv...")
data = data_storage.read_csv_files(channels)
table = data_storage.create_single_table(data)
nRows = table.shape[0]
print("Successfully Loaded the " + dataset + " dataset")

print("Calculating power estimates...")

lists = OrderedDict([
    ("bms", []),
    ("inverterTemps", []),
    ("powerLoss", []),
    ("powerLossDeviationPercentage", [])
])

for i in range(nRows):
    
    bms = table[i][-1] * 1000 # Converting from kWh -> W

    if bms > 0:
        currentPowerLoss = estimatePowerLoss(table, bms)
        lists["powerLoss"].append(currentPowerLoss)

        deviationPercentage = calculateDeviationPercentage(currentPowerLoss, bms, lists["powerLossDeviationPercentage"][-1])
        lists["powerLossDeviationPercentage"].append(deviationPercentage)
    else:
        lists["powerLoss"].append(0)
        lists["powerLossDeviationPercentage"].append(0)
    
    currentInverterTemp = summateInverterTemp(table)
    lists["inverterTemps"].append(currentInverterTemp)

    lists["bms"].append(bms)

# This array contains the same IDs as the lists
clusteredArrays = clusterArrays(lists, nRows)

correlation = np.corrcoef(np.array(lists["inverterTemps"]), np.array(lists["powerLossDeviationPercentage"]))

print(correlCo(lists["inverterTemps"], lists["powerLossDeviationPercentage"]))

plt.plot(clusteredArrays[list(lists).index("bms")])
plt.show()

plt.plot(clusteredArrays[list(lists).index("powerLossDeviationPercentage")])
plt.show()

plt.plot(clusteredArrays[list(lists).index("powerLoss")])
plt.plot(clusteredArrays[list(lists).index("bms")])
plt.ylabel('some numbers')
plt.show()