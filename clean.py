#!/usr/bin/env python
import os
import shutil

while True:
	entry = raw_input("Are you sure? [y/n] ")
	if entry == 'y' or entry == 'Y':

		# clean the data directory
		path = 'data'
		if os.path.exists(path):
			shutil.rmtree(path)
			
		break
	elif entry == 'n' or entry == 'N':
		break
	else:
		print "Try again."
