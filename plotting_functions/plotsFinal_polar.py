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
circle_050 = np.genfromtxt("../results_final/experiment_1/sim10_050_road.csv", delimiter=",")
circle_100 = np.genfromtxt("../results_final/experiment_1/sim10_100_road.csv", delimiter=",")
circle_150 = np.genfromtxt("../results_final/experiment_1/sim10_150_road.csv", delimiter=",")

# stack two additional columns for ventral flow errors
circle_050 = np.hstack((circle_050, np.zeros((circle_050.shape[0], 2))))
circle_100 = np.hstack((circle_100, np.zeros((circle_100.shape[0], 2))))
circle_150 = np.hstack((circle_150, np.zeros((circle_150.shape[0], 2))))

# linear interpolation for the computation of the error
circle_050[:,3] = np.interp(circle_050[:,0], venFlow050[:,0], venFlow050[:,1]) - circle_050[:,1]
circle_050[:,4] = np.interp(circle_050[:,0], venFlow050[:,0], venFlow050[:,2]) - circle_050[:,2]
circle_100[:,3] = np.interp(circle_100[:,0], venFlow100[:,0], venFlow100[:,1]) - circle_100[:,1]
circle_100[:,4] = np.interp(circle_100[:,0], venFlow100[:,0], venFlow100[:,2]) - circle_100[:,2]
circle_150[:,3] = np.interp(circle_150[:,0], venFlow150[:,0], venFlow150[:,1]) - circle_150[:,1]
circle_150[:,4] = np.interp(circle_150[:,0], venFlow150[:,0], venFlow150[:,2]) - circle_150[:,2]

f = plt.figure(1, figsize=(12, 5))
ax = plt.subplot(111, projection='polar')

th = []
for i in xrange(33,5985):
	angle = np.arctan(venFlow150[i,2]/venFlow150[i,1])
	if venFlow150[i,1] < 0: angle += np.pi
	th.append(angle)

ax.plot(th, circle_150[:,3])
ax.plot(th, circle_150[:,4])
plt.show()