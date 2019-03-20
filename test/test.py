import numpy as np
a = [-1,-5,1,5,10.0]
b = [-11,-6,2,6,11]
e = [-12,-7,3,7,12]
f = [-13,-8,4,8,13]
g = [-14,-9,5,9,14]
c = np.array([a,b,e,f,g])
d = c*0
if a[0] < b[0]:
    a = b
else:
    b = a
c = np.array([a,b,e,f,g])
print(c)


for i in range(0,len(c[0])):
    cmin = min(c[:,i])
    cmax = max(c[:,i])
    d[:,i] = (c[:,i]-cmin-(cmax-cmin)/2)/(cmax-cmin)*2
x = c[:,0:2]
print(x)
