import os
import csv
import h5py
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

matplotlib.rcParams.update({'font.size': 12})

# do we need to save the images again?
grid = True

# load data under analysis
data = np.genfromtxt("../data/logs.csv", delimiter=",")

fig, ax = plt.subplots()
plt.plot(data[:,0], data[:,1], label = 'Training loss')
plt.plot(data[:,0], data[:,2], label = 'Validation loss')
plt.xlabel('epochs [-]')
plt.ylabel(r'MS($\hat{\omega} - \omega$) [$1/s^2$]')
plt.grid(grid)

gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

plt.legend()

plt.tight_layout()
#plt.savefig('/home/fedepare/Preliminary Thesis Report/Include/images/mse.eps', bbox_inches='tight')

plt.show()