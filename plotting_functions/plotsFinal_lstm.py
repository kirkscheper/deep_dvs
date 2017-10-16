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
venFlow100 = np.loadtxt('../results_final/ground_truth/circle_100.txt')
venFlow100 = venFlow100[:-1,:]

# load data under analysis
circle_050 = np.genfromtxt("../results_final/experiment_5/ref.csv", delimiter=",")
circle_100 = np.genfromtxt("../results_final/experiment_5/sim07_100_road.csv", delimiter=",")

# ventral flow vx
f, axarr = plt.subplots(2, 2, figsize=(12, 5))
l1, = axarr[0,0].plot(circle_050[:,0], circle_050[:,1], color='#ff7f0e')
axarr[0,1].plot(circle_100[:,0], circle_100[:,1], color='#ff7f0e')
axarr[1,0].plot(circle_050[:,0], circle_050[:,2], color='#ff7f0e')
axarr[1,1].plot(circle_100[:,0], circle_100[:,2], color='#ff7f0e')

l2, = axarr[0,0].plot(venFlow100[:,0], venFlow100[:,1], color='#1f77b4')
axarr[0,1].plot(venFlow100[:,0], venFlow100[:,1], color='#1f77b4')
axarr[1,0].plot(venFlow100[:,0], venFlow100[:,2], color='#1f77b4')
axarr[1,1].plot(venFlow100[:,0], venFlow100[:,2], color='#1f77b4')

axarr[0,0].set_ylim([-2.75, 2.75])
axarr[0,1].set_ylim([-2.75, 2.75])
axarr[1,0].set_ylim([-2.75, 2.75])
axarr[1,1].set_ylim([-2.75, 2.75])

axarr[0,0].set_ylabel(r'$\omega_x$ [1/s]')
axarr[1,0].set_ylabel(r'$\omega_y$ [1/s]')
axarr[1,0].set_xlabel(r'$t$ [ms]')
axarr[1,1].set_xlabel(r'$t$ [ms]')
axarr[0,0].set_title(r'CNN')
axarr[0,1].set_title(r'RCNN: $N_{b}=64$, $N_{cells}=128$')
axarr[0,0].grid(grid)
axarr[0,1].grid(grid)
axarr[1,0].grid(grid)
axarr[1,1].grid(grid)

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

plt.tight_layout()
#plt.savefig('/home/fedepare/Preliminary Thesis Report/Include/images/results/exp5_1.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()