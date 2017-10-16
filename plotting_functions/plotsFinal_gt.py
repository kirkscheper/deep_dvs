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

# load ground truth data
venFlow050 = np.loadtxt('../results_final/ground_truth/circle_050.txt')
venFlow075 = np.loadtxt('../results_final/ground_truth/circle_075.txt')
venFlow100 = np.loadtxt('../results_final/ground_truth/circle_100.txt')
venFlow125 = np.loadtxt('../results_final/ground_truth/circle_125.txt')
venFlow150 = np.loadtxt('../results_final/ground_truth/circle_150.txt')

# ventral flow vx
f, axarr = plt.subplots(1, 2, figsize=(12, 3))
l1, = axarr[0].plot(venFlow050[:-1,0], venFlow050[:-1,1])
axarr[1].plot(venFlow050[:-1,0], venFlow050[:-1,2])

l2, = axarr[0].plot(venFlow075[:-1,0], venFlow075[:-1,1])
axarr[1].plot(venFlow075[:-1,0], venFlow075[:-1,2])

l3, = axarr[0].plot(venFlow100[:-1,0], venFlow100[:-1,1])
axarr[1].plot(venFlow100[:-1,0], venFlow100[:-1,2])

l4, = axarr[0].plot(venFlow125[:-1,0], venFlow125[:-1,1])
axarr[1].plot(venFlow125[:-1,0], venFlow125[:-1,2])

l5, = axarr[0].plot(venFlow150[:-1,0], venFlow150[:-1,1])
axarr[1].plot(venFlow150[:-1,0], venFlow150[:-1,2])

axarr[0].set_ylabel(r'$\omega_x$ [1/s]')
axarr[1].set_ylabel(r'$\omega_y$ [1/s]')
axarr[0].set_xlabel(r'$t$ [ms]')
axarr[1].set_xlabel(r'$t$ [ms]')
axarr[0].set_ylim([-3.5, 3.5])
axarr[1].set_ylim([-3.5, 3.5])
axarr[0].grid(grid)
axarr[1].grid(grid)

lgd = f.legend((l1, l2, l3, l4, l5), ('r = 0.50 m', 'r = 0.75 m', 'r = 1.00 m', 'r = 1.25 m', 'r = 1.50 m'), loc="upper left", bbox_to_anchor=[0.165, 1.065], ncol=5, frameon=False)

plt.tight_layout()

gridlines = axarr[0].get_xgridlines() + axarr[0].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')
gridlines = axarr[1].get_xgridlines() + axarr[1].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

#f.savefig('/home/fedepare/Preliminary Thesis Report/Include/images/test_dataset.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')
#plt.show()