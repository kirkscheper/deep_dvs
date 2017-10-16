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
checker_100_05 = np.genfromtxt("../results_final/experiment_lstm2/1000_100_road.csv", delimiter=",")
checker_100_10 = np.genfromtxt("../results_final/experiment_lstm2/5000_100_grass.csv", delimiter=",")
checker_100_20 = np.genfromtxt("../results_final/experiment_lstm2/1000_100_checker.csv", delimiter=",")

# ventral flow vx
f, axarr = plt.subplots(2, 3, figsize=(12, 5))
l1, = axarr[0,0].plot(checker_100_05[:-1,0], checker_100_05[:-1,1], color='#ff7f0e')
axarr[1,0].plot(checker_100_05[:-1,0], checker_100_05[:-1,2], color='#ff7f0e')

axarr[0,1].plot(checker_100_10[:-1,0], checker_100_10[:-1,1], color='#ff7f0e')
axarr[1,1].plot(checker_100_10[:-1,0], checker_100_10[:-1,2], color='#ff7f0e')

axarr[0,2].plot(checker_100_20[:-1,0], checker_100_20[:-1,1], color='#ff7f0e')
axarr[1,2].plot(checker_100_20[:-1,0], checker_100_20[:-1,2], color='#ff7f0e')

l2, = axarr[0,1].plot(venFlow100[:,0], venFlow100[:,1], color='#1f77b4')
axarr[1,1].plot(venFlow100[:,0], venFlow100[:,2], color='#1f77b4')
axarr[0,0].plot(venFlow100[:,0], venFlow100[:,1], color='#1f77b4')
axarr[1,0].plot(venFlow100[:,0], venFlow100[:,2], color='#1f77b4')
axarr[0,2].plot(venFlow100[:,0], venFlow100[:,1], color='#1f77b4')
axarr[1,2].plot(venFlow100[:,0], venFlow100[:,2], color='#1f77b4')

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
axarr[0,0].set_title('Set 3')
axarr[0,1].set_title('Set 4')
axarr[0,2].set_title('Set 5')
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
#plt.savefig('/home/fedepare/Preliminary Thesis Report/Include/images/lstm02.eps', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()