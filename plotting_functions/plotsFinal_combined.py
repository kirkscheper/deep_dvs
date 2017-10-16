import os
import csv
import h5py
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

matplotlib.rcParams.update({'font.size': 14})

# do we need to save the images again?
grid = True

# load ground truth data
venFlow050 = np.loadtxt('../results_final/ground_truth/circle_050.txt')
venFlow100 = np.loadtxt('../results_final/ground_truth/circle_100.txt')
venFlow150 = np.loadtxt('../results_final/ground_truth/circle_150.txt')
venFlow050 = venFlow050[:-1,:]
venFlow100 = venFlow100[:-1,:]
venFlow150 = venFlow150[:-1,:]

# load data under analysis
circle_050 = np.genfromtxt("../results_final/experiment_1/sim10_050_road.csv", delimiter=",")
circle_100 = np.genfromtxt("../results_final/experiment_1/sim10_100_road.csv", delimiter=",")
circle_150 = np.genfromtxt("../results_final/experiment_1/sim10_150_road.csv", delimiter=",")

# ventral flow vx
f, axarr = plt.subplots(2, 1, figsize=(8, 5))
l1, = axarr[0].plot(circle_150[:,0], circle_150[:,1], color='#ff7f0e')
axarr[1].plot(circle_150[:,0], circle_150[:,2], color='#ff7f0e')

l2, = axarr[0].plot(venFlow150[:,0], venFlow150[:,1], color='#1f77b4')
axarr[1].plot(venFlow150[:,0], venFlow150[:,2], color='#1f77b4')

axarr[0].set_ylim([-4, 4])
axarr[1].set_ylim([-4, 4])

axarr[0].set_ylabel(r'$\omega_x$ [1/s]')
axarr[1].set_ylabel(r'$\omega_y$ [1/s]')
axarr[1].set_xlabel(r'$t$ [ms]')
axarr[0].set_title(r'Set 3: $r=1.50$ m', fontsize=16)
axarr[0].grid(grid)
axarr[1].grid(grid)

lgd = f.legend((l2, l1), ('Ground truth', 'Estimated'), loc="upper left", bbox_to_anchor=[0.23, 1.065], ncol=2, frameon=False, fontsize=16)

for line in lgd.get_lines():
    line.set_linewidth(2)

gridlines = axarr[0].get_xgridlines() + axarr[0].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')
gridlines = axarr[1].get_xgridlines() + axarr[1].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

plt.tight_layout()
plt.savefig('/home/fedepare/Preliminary Thesis Report/Include/images/results/combo_1.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()