import sys
import csv
import glob
import random
import os.path
import operator
import numpy as np
import pandas as pd
from keras.utils import np_utils
from processor import stack_image, temporal_image


class DataSet():

    # constructor
    def __init__(self, seq_length = 150, image_shape = (128, 128, 1), dataFolder = 'dataset', stack = True, imType = None, separation = 0, pxShift = False, sequence = False):

        # get the name of the dataset folder
        self.folder = dataFolder

        # legth of the sequence
        self.seq_length = seq_length

        # size of the image (channels as third dim.)
        self.image_shape = image_shape

        # get the data
        self.data = self.get_data()

        # image type
        self.imType = imType

        # get number of .png files
        self.usable_data()

        # starting image
        self.starting = 11

        # stack images
        self.stack = stack

        # batch is a sequence?
        self.sequence = sequence

        # image separation
        self.separation = separation - 1

        # pixel shift as ground truth
        self.pxShift = pxShift

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
            elif self.imType == 'both':
                path = self.folder + '/' + item[1] + '/' + 'ON' + '/' # just for counting frames
            else:
                path = self.folder + '/' + item[1] + '/'
            pngs = len(glob.glob1(path,"*.png"))

            # number of images
            item.append(pngs)

        del path, pngs


    # split the incoming data in train and test groups
    def split_train_test(self):

        train = []
        test  = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item[1:])
            else:
                test.append(item[1:])

        return train, test


    # get the corresponding frames from a sample
    def get_frames_for_sample(self, sample):

        # number of frames used
        self.numFrames = []
        
        # 1D array with the frames in this sample
        images = []
        path = self.folder + '/' + sample[0] + '/'

        # if temporally separated images are needed
        if self.separation != 0:
            for i in range(0, self.seq_length):

                # select image
                if i == 0:
                    imgIdx = sample[1]
                else:
                    imgIdx = sample[1] - self.separation * i

                # update vector
                self.numFrames.append(imgIdx)
            
                # include the number of the desired image
                if self.imType == 'OFF' or self.imType == 'ON':
                    images.append(path + self.imType + '/' + str(imgIdx) + '.png')
                elif self.imType == 'both':
                    images.append(path + 'ON' + '/' + str(imgIdx) + '.png')
                    images.append(path + 'OFF' + '/' + str(imgIdx) + '.png')
                else:
                    images.append(path + str(imgIdx) + '.png')

        # successive images
        else:
            for i in range(sample[1], sample[1] - self.seq_length, -1):

                # update the frame index vector
                self.numFrames.append(i)

                # include the number of the desired image
                if self.imType == 'OFF' or self.imType == 'ON':
                    images.append(path + self.imType + '/' + str(i) + '.png')
                elif self.imType == 'both':
                    images.append(path + 'ON' + '/' + str(i) + '.png')
                    images.append(path + 'OFF' + '/' + str(i) + '.png')
                else:
                    images.append(path + str(i) + '.png')

        return images


    def build_image_sequence(self, frames):

        if self.stack == True: return stack_image(frames, self.image_shape)
        else:
            sequence = []
            if self.imType == 'OFF' or self.imType == 'ON' or self.imType == None:
                for n in range(len(frames),0,-1):
                    sequence.append(temporal_image(frames[n-1], self.image_shape))
            elif self.imType == 'both':
                sequence.append(stack_image(frames, self.image_shape)) 

            return sequence


    def get_grndTruth(self, sample):

        path = self.folder + '/' + sample[0] + '/'

        if self.seq_length == 2 and self.pxShift == True: # pixel shift
            filename = 'trajectory.txt'
            gt = np.loadtxt(path + filename)

            # load trajectory data
            x, y = [], []
            for i in range(0, len(self.numFrames)):
                x.append(gt[self.numFrames[i] - 1, 1])
                y.append(gt[self.numFrames[i] - 1, 2])

            # compute pixel shift
            pxShift = [-(x[0] - x[1])*self.m2px, -(y[0] - y[1])*self.m2px]
            return np.array(pxShift).astype(np.float32)

        else: # ventral flow computed from blender

            filename = 'ventral_flow.txt'
            gt = np.loadtxt(path + filename)
            return gt[sample[1] - 1, 1:3].astype(np.float32)


    def train_generator(self, batch_size):

        # get the right dataset for the generator
        train, test = self.split_train_test()
        self.train = train

        while 1:

            # get the folder and sequence number
            if self.sequence == True:
                folder = random.choice(self.train)
                seqNum = random.randint(self.starting + self.seq_length - 1 + self.separation*self.seq_length, folder[1] - batch_size - 1)

            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):

                # get the folder and sequence number
                if self.sequence == False:
                    folder = random.choice(self.train)
                    seqNum = random.randint(self.starting + self.seq_length - 1 + self.separation*self.seq_length, folder[1] - batch_size - 1)

                # reset
                sequence = None
                    
                # get the frames in this sample
                sample = [folder[0], seqNum]
                frames = self.get_frames_for_sample(sample)

                # build the image sequence
                sequence = self.build_image_sequence(frames)

                # get the ground-truth data 
                gt = self.get_grndTruth(sample)

                # next step in the sequence
                seqNum += 1

                if sequence is None:
                    print("\nCan't find sequence. Did you generate them?")
                    sys.exit()

                X.append(sequence)
                y.append(gt)

            yield np.array(X), np.array(y)


    def validate_generator(self, batch_size):

        # Get the right dataset for the generator
        train, test = self.split_train_test()
        self.test = test

        while 1:

            if self.sequence == True:
                folder = random.choice(self.test)
                seqNum = random.randint(self.starting + self.seq_length - 1 + self.separation*self.seq_length, folder[1] - batch_size - 1)

            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):

                if self.sequence == False:
                    folder = random.choice(self.test)
                    seqNum = random.randint(self.starting + self.seq_length - 1 + self.separation*self.seq_length, folder[1] - batch_size - 1)

                # reset
                sequence = None
                    
                # get the frames in this sample
                sample = [folder[0], seqNum]
                frames = self.get_frames_for_sample(sample)

                # build the image sequence
                sequence = self.build_image_sequence(frames)

                # get the ground-truth data 
                gt = self.get_grndTruth(sample)

                # next step in the sequence
                seqNum += 1

                if sequence is None:
                    print("\nCan't find sequence. Did you generate them?")
                    sys.exit()

                X.append(sequence)
                y.append(gt)

            yield np.array(X), np.array(y)


    def test(self, folder, seqNum):

        X, y = [], []
            
        # get the frames in this sample
        sample = [folder, seqNum]
        frames = self.get_frames_for_sample(sample)

        # build the image sequence
        sequence = self.build_image_sequence(frames)

        # get the ground-truth data 
        gt = self.get_grndTruth(sample)

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


    def test_batch(self, folder, seqNum, batch_size, pngs):

        X, y = [], []

        # Generate batch_size samples.
        for _ in range(batch_size):

            if seqNum < pngs:
                
                # get the frames in this sample
                sample = [folder, seqNum]
                frames = self.get_frames_for_sample(sample)

                # build the image sequence
                sequence = self.build_image_sequence(frames)

                # get the ground-truth data 
                gt = self.get_grndTruth(sample)

                # next step in the sequence
                seqNum += 1

                if sequence is None:
                    print("\nCan't find sequence. Did you generate them?")
                    sys.exit()

                X.append(sequence)
                y.append(gt)

            else: break

        return np.array(X), np.array(y), seqNum