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
venFlow100 = np.loadtxt('../results_final/ground_truth/circle_100.txt')
venFlow150 = np.loadtxt('../results_final/ground_truth/circle_150.txt')
venFlow050 = venFlow050[:-1,:]
venFlow100 = venFlow100[:-1,:]
venFlow150 = venFlow150[:-1,:]

# load data under analysis
circle_050 = np.genfromtxt("../results_final/experiment_1/sim10_050_checker.csv", delimiter=",")
circle_100 = np.genfromtxt("../results_final/experiment_1/sim10_100_checker.csv", delimiter=",")
circle_150 = np.genfromtxt("../results_final/experiment_1/sim10_150_checker.csv", delimiter=",")

# ventral flow vx
f, axarr = plt.subplots(2, 3, figsize=(12, 5))
l1, = axarr[0,0].plot(circle_050[:,0], circle_050[:,1], color='#ff7f0e')
axarr[0,1].plot(circle_100[:,0], circle_100[:,1], color='#ff7f0e')
axarr[0,2].plot(circle_150[:,0], circle_150[:,1], color='#ff7f0e')
axarr[1,0].plot(circle_050[:,0], circle_050[:,2], color='#ff7f0e')
axarr[1,1].plot(circle_100[:,0], circle_100[:,2], color='#ff7f0e')
axarr[1,2].plot(circle_150[:,0], circle_150[:,2], color='#ff7f0e')

l2, = axarr[0,0].plot(venFlow050[:,0], venFlow050[:,1], color='#1f77b4')
axarr[0,1].plot(venFlow100[:,0], venFlow100[:,1], color='#1f77b4')
axarr[0,2].plot(venFlow150[:,0], venFlow150[:,1], color='#1f77b4')
axarr[1,0].plot(venFlow050[:,0], venFlow050[:,2], color='#1f77b4')
axarr[1,1].plot(venFlow100[:,0], venFlow100[:,2], color='#1f77b4')
axarr[1,2].plot(venFlow150[:,0], venFlow150[:,2], color='#1f77b4')

axarr[0,0].set_ylim([-4.75, 4.75])
axarr[0,1].set_ylim([-4.75, 4.75])
axarr[0,2].set_ylim([-4.75, 4.75])
axarr[1,0].set_ylim([-4.75, 4.75])
axarr[1,1].set_ylim([-4.75, 4.75])
axarr[1,2].set_ylim([-4.75, 4.75])

axarr[0,0].set_ylabel(r'$\omega_x$ [1/s]')
axarr[1,0].set_ylabel(r'$\omega_y$ [1/s]')
axarr[1,0].set_xlabel(r'$t$ [ms]')
axarr[1,1].set_xlabel(r'$t$ [ms]')
axarr[1,2].set_xlabel(r'$t$ [ms]')
axarr[0,0].set_title(r'$r=0.5$ m')
axarr[0,1].set_title(r'$r=1.0$ m')
axarr[0,2].set_title(r'$r=1.5$ m')
axarr[0,0].grid(grid)
axarr[0,1].grid(grid)
axarr[1,0].grid(grid)
axarr[1,1].grid(grid)
axarr[0,2].grid(grid)
axarr[1,2].grid(grid)

lgd = f.legend((l2, l1), ('Ground truth', 'Estimated'), loc="upper left", bbox_to_anchor=[0.34, 1.065], ncol=2, frameon=False, fontsize=14)

for line in lgd.get_lines():
    line.set_linewidth(2)

gridlines = axarr[0,0].get_xgridlines() + axarr[0,0].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')
gridlines = axarr[0,1].get_xgridlines() + axarr[0,1].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')
gridlines = axarr[1,0].get_xgridlines() + axarr[1,0].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')
gridlines = axarr[1,1].get_xgridlines() + axarr[1,1].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')
gridlines = axarr[0,2].get_xgridlines() + axarr[0,2].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')
gridlines = axarr[1,2].get_xgridlines() + axarr[1,2].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

plt.tight_layout()
#plt.savefig('/home/fedepare/Preliminary Thesis Report/Include/images/results/exp1_2.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()