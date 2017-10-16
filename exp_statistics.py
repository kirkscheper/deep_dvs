import os
import csv
import h5py
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# do we need to save the images again?
save = False

# load ground truth data
venFlow050 = np.loadtxt('results_final/ground_truth/circle_050.txt')
venFlow075 = np.loadtxt('results_final/ground_truth/circle_075.txt')
venFlow100 = np.loadtxt('results_final/ground_truth/circle_100.txt')
venFlow125 = np.loadtxt('results_final/ground_truth/circle_125.txt')
venFlow150 = np.loadtxt('results_final/ground_truth/circle_150.txt')

# folder
path = 'results_final/experiment_variance/'

# collect and process the data 
data = []
entries = glob.glob1(path,"*.csv")
for entry in entries:

	# entry name
	entrySplit = entry.split('_')

	# load numpy array
	aux = np.genfromtxt(path + entry, delimiter=',')

	if entrySplit[-2] == '050':
		venFlow = venFlow050[:-1,:]
	elif entrySplit[-2] == '075':
		venFlow = venFlow075[:-1,:]
	elif entrySplit[-2] == '100':
		venFlow = venFlow100[:-1,:]
	elif entrySplit[-2] == '125':
		venFlow = venFlow125[:-1,:]
	elif entrySplit[-2] == '150':
		venFlow = venFlow150[:-1,:]

	# error computation
	aux = np.hstack((aux, np.zeros((aux.shape[0], 2))))
	aux[:,3] = np.interp(aux[:,0], venFlow[:,0], venFlow[:,1]) - aux[:,1]
	aux[:,4] = np.interp(aux[:,0], venFlow[:,0], venFlow[:,2]) - aux[:,2]

	# append the matrix to data
	data.append(aux)

# create dictionary
dic = dict(zip(entries, data))

# mean absolute error and variance
print('Mean absolute error and variance')
for key in dic:

	# entry name
	keySplit = key.split('_')

	if keySplit[-2] == '050':
		venFlow = venFlow050[:-1,:]
	elif keySplit[-2] == '075':
		venFlow = venFlow075[:-1,:]
	elif keySplit[-2] == '100':
		venFlow = venFlow100[:-1,:]
	elif keySplit[-2] == '125':
		venFlow = venFlow125[:-1,:]
	elif keySplit[-2] == '150':
		venFlow = venFlow150[:-1,:]

	print '(%s)\t avg_vx: %.4f, var_vx: %.4f \n\t avg_vy: %.4f, var_vy: %.4f' % \
	(key, mean_absolute_error(np.interp(dic[key][:,0], venFlow[:,0], venFlow[:,1]), dic[key][:,1]), np.var(dic[key][:,3]), \
	mean_absolute_error(np.interp(dic[key][:,0], venFlow[:,0], venFlow[:,2]), dic[key][:,2]), np.var(dic[key][:,4]))