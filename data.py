import sys
import csv
import glob
import random
import os.path
import operator
import numpy as np
import pandas as pd
from keras.utils import np_utils
from processor import load_image

from PIL import Image
from keras.preprocessing.image import img_to_array, load_img

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),'dvs_simulator'))
from dataset_bagfiles import *

class SimSetting():
    def __init__ (self, accumTime=1000, imType='temporal', expScale=0.00025, imageResMs=5):
        self.accumTime = accumTime
        self.imType = imType
        self.expScale = expScale
        self.imageResMs = imageResMs
    
class DataSet():

    # constructor
    def __init__(self, seq_length = 150, image_shape = (128, 128, 1), dataFolder = 'dataset', imType = None,
                  separation = 1, pxShift = False, sequence_batch = False, online_sim = False, 
                  sim_settings = SimSetting()):

        # get the name of the dataset folder
        self.folder = dataFolder

        # legth of the sequence
        self.seq_length = seq_length

        # size of the image (channels as third dim.)
        self.image_shape = image_shape

        # image type
        self.imType = imType

        # get the data
        self.data = self.get_data()
        
        # get number of .png files
        self.usable_data()
        
        # assign trainging and validation data
        self.split_train_val()

        # starting image
        self.starting = 11

        # batch is a sequence?
        self.sequence_batch = sequence_batch

        # image separation
        self.separation = separation
        if self.separation <= 0:
            self.separation = 1

        # pixel shift as ground truth
        self.pxShift = pxShift
        
        self.online_sim = online_sim
        self.sim_settings = sim_settings

        # ASSUMPTION: CONSTANT ALTITUDE
        FoV  = 70. # deg
        alt  = 0.5 # m
        self.m2px = (self.image_shape[0] / 2.) / ((alt*np.sin((FoV/2.)*np.pi/180.))/(np.sin((90-(FoV/2.))*np.pi/180.)))

        
    # load the data from .csv file
    def get_data(self):
        with open('./' + self.folder + '/data_file.csv', 'r') as fin:
            reader = csv.reader(fin)
            data   = list(reader)

        return data


    # number of .png files
    def usable_data(self):
        
        for item in self.data:
            if self.imType == 'OFF' or self.imType == 'ON':
                path = self.folder + '/' + item[1] + '/' + self.imType + '/'
            elif self.imType == 'BOTH':
                path = self.folder + '/' + item[1] + '/' + 'ON' + '/' # just for counting frames
            else:
                path = self.folder + '/' + item[1] + '/'
            pngs = len(glob.glob1(path,"*.png"))

            # number of images
            item.append(pngs)

        del path, pngs


    # split the incoming data in train and val groups
    def split_train_val(self):

        self.train = []
        self.val  = []
        for item in self.data:
            if item[0] == 'train':
                self.train.append(item[1:])
            elif item[0] == 'val':
                self.val.append(item[1:])

    # get an image sequence of length self.seq_length starting at sample, separated by self.separation image frames
    def get_image_sequence(self, folder, sample_num, batch_size=1):

        # number of frames used
        self.frame_num = []
        
        # 1D array with the frames in this sample
        path = self.folder + '/' + folder
        
        data = dataset('')
        if self.online_sim:
                
            sequence = data.generate_images( 
                pathFrom = self.folder,
                bagFile = folder,
                accumTime = self.sim_settings.accumTime,
                imtype = self.sim_settings.imType,
                expScale = self.sim_settings.expScale,
                imageResMs = self.sim_settings.imageResMs,
                imgCntStart = sample_num,
                store_imgs = False,
                num_imgs = batch_size)
            
            self.frame_num = range(sample_num, sample_num + batch_size*self.sim_settings.imageResMs, self.sim_settings.imageResMs) #self.sim_settings.imgSkip
            
        else:
            for i in range(sample_num, sample_num + self.seq_length*self.separation, self.separation):

                # update the frame index vector
                self.frame_num.append(i)
            
                # include the number of the desired image
                if self.imType == 'OFF' or self.imType == 'ON':
                    frame = load_image(path + '/' + self.imType + '/' + str(i) + '.png', self.image_shape)
                elif self.imType == 'BOTH':
                    frame = load_image(path + '/ON' + '/' + str(i) + '.png', self.image_shape)
                    np.dstack((frame, load_image(path + '/OFF' + '/' + str(i) + '.png', self.image_shape)))
                else:
                    frame = load_image(path + '/' + str(i) + '.png', self.image_shape)
                
            if not 'sequence' in vars():
                sequence = frame
            else:
                np.dstack((sequence, frame))
                    
        return sequence


    def get_grndTruth(self, folder, sample_nums):
        path = self.folder + '/' + folder
            
        if self.seq_length == 2 and self.pxShift == True: # pixel shift
            filename = 'trajectory.txt'
            gt = np.loadtxt(path + '/' + filename)

            # load trajectory data
            x, y = [], []
            for i in range(0, len(self.frame_num)):
                x.append(gt[sample_nums[i], 1])
                y.append(gt[sample_nums[i], 2])

            # compute pixel shift
            pxShift = [-(x[0] - x[1])*self.m2px, -(y[0] - y[1])*self.m2px]
            return np.array(pxShift).astype(np.float32)

        else: # ventral flow computed from blender

            filename = 'ventral_flow.txt'
            gt = np.loadtxt(path + '/' + filename)
            return gt[sample_nums, 1:3].astype(np.float32)


    def train_generator(self, batch_size):
        
        while 1:

            # get the folder and sequence number
            if self.sequence_batch == True:
                folder = random.choice(self.train)
                if self.online_sim:
                    max_image_number = 980 # data.get_duration_of_dataset_ms(self.folder + '/' + str(self.train[0][0]) + '/' + str(self.train[0][0]) + '.bag')
                    max_image_number = folder[1]
                seqNum = random.randint(self.starting, max_image_number - (self.seq_length*self.separation) - batch_size)
            
            X, y = [], []
            
            # Generate batch_size samples.
            for _ in range(batch_size):
                
                # get the folder and sequence number
                if self.sequence_batch == False:
                    folder = random.choice(self.train)
                    if self.online_sim:
                        max_image_number = 980 #data.get_duration_of_dataset_ms(self.folder + '/' + str(self.train[0][0]) + '/' + str(self.train[0][0]) + '.bag')
                    else:
                        max_image_number = folder[1]
                    seqNum = random.randint(self.starting, max_image_number - (self.seq_length*self.separation))
                
                # reset
                sequence = None
                    
                # read image sequence from premade files
                sequence = self.get_image_sequence(folder[0], seqNum)
                    
                if sequence is None:
                    print("\nCan't find sequence. Did you generate them?")
                    sys.exit()
                
                # get the ground-truth data 
                gt = self.get_grndTruth(folder[0], seqNum)
                
                # next step in the sequence
                seqNum += 1

                X.append(sequence)
                y.append(gt)

            yield np.array(X), np.array(y)


    def validate_generator(self, batch_size):

        while 1:

            # get the folder and sequence number
            if self.sequence_batch == True:
                folder = random.choice(self.train)
                if self.online_sim:
                    max_image_number = 980 # data.get_duration_of_dataset_ms(self.folder + '/' + str(self.train[0][0]) + '/' + str(self.train[0][0]) + '.bag')
                    max_image_number = folder[1]
                seqNum = random.randint(self.starting, max_image_number - (self.seq_length*self.separation) - batch_size)
            
            X, y = [], []
            
            # Generate batch_size samples.
            for _ in range(batch_size):
                
                # get the folder and sequence number
                if self.sequence_batch == False:
                    folder = random.choice(self.val)
                    if self.online_sim:
                        max_image_number = 980 #data.get_duration_of_dataset_ms(self.folder + '/' + str(self.train[0][0]) + '/' + str(self.train[0][0]) + '.bag')
                    else:
                        max_image_number = folder[1]
                    seqNum = random.randint(self.starting, max_image_number - (self.seq_length*self.separation))
                
                # reset
                sequence = None
                    
                # read image sequence from premade files
                sequence = self.get_image_sequence(folder[0], seqNum)
                    
                if sequence is None:
                    print("\nCan't find sequence. Did you generate them?")
                    sys.exit()
                
                # get the ground-truth data 
                gt = self.get_grndTruth(folder[0], seqNum)
                
                # next step in the sequence
                seqNum += 1

                X.append(sequence)
                y.append(gt)

            yield np.array(X), np.array(y)


    def val(self, folder, seqNum):

        X, y = [], []
            
        # build the image sequence
        sequence = self.get_image_sequence(folder, seqNum)

        # get the ground-truth data 
        gt = self.get_grndTruth(folder, self.frame_num)

        # next step in the sequence
        seqNum += 1

        if sequence is None:
            print("\nCan't find sequence. Did you generate them?")
            sys.exit()

        X.append(sequence)
        y.append(gt)

        # get image properties
        Nef = 0
        var_img = 0
        for i in range(0, sequence.shape[-1]):
            I = np.asarray(sequence[:,:,i], dtype=np.int32)
            Nef += (I > 0).sum()
            var_img += np.var(I)   

        Nef     = float(Nef) / float(sequence.shape[-1])
        var_img = float(var_img) / float(sequence.shape[-1])

        return np.array(X), np.array(y), Nef, var_img


    def val_batch(self, folder, seqNum, batch_size, pngs):

        X, y = [], []

        # Generate batch_size samples.
        for _ in range(batch_size):

            if seqNum < pngs:
                
                # build the image sequence
                sequence = self.get_image_sequence(folder, seqNum)

                # get the ground-truth data 
                gt = self.get_grndTruth(folder, self.frame_num)

                # next step in the sequence
                seqNum += 1

                if sequence is None:
                    print("\nCan't find sequence. Did you generate them?")
                    sys.exit()

                X.append(sequence)
                y.append(gt)

            else: break

        return np.array(X), np.array(y), seqNum



    def data_function(self, i, batch_size, lamb, data_type):

        if i == 0:
            # get the folder and sequence number
            if data_type == 'train':
                self.sample_folder = random.choice(self.train)
            else:
                self.sample_folder = random.choice(self.val)
            
            self.sample_seqNum = random.randint(self.starting / self.sim_settings.imageResMs, 980 / self.sim_settings.imageResMs - (self.seq_length*self.separation) - batch_size)    # somehow there are no events after 986ms
            
        if lamb < 0.001:
            lamb = 0.001
        elif lamb > 1:
            lamb =  1
        #print 'lamb ' + str(lamb) + ' 1/ms, mean lifetime ' + str(0.001/lamb) + ' s'

        self.sim_settings.expScale = lamb / 1000
        
        # build the image sequence
        sequence = self.get_image_sequence(self.sample_folder[0], self.sample_seqNum, batch_size)
        
        # get the ground-truth data 
        gt = self.get_grndTruth(self.sample_folder[0], self.frame_num)

        # next step in the sequence
        self.sample_seqNum += self.separation / self.sim_settings.imageResMs

        if sequence is None:
            print("\nCan't find sequence. Did you generate them?")
            sys.exit()

        X = sequence
        y = gt
        lamb = np.full((batch_size, 1), lamb)
        
        #return [np.array(X), np.array(lamb)], np.array(y)
        return np.array(X), np.array(y)
    