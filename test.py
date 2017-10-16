import os
import sys
import h5py
import time
import glob
import csv
import time
import os.path
from data import DataSet
from models import ResearchModels
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger


def check(model, saved_model = None, seq_length = 150, dataFolder = 'dataset', stack = True, imType = None, separation = 0, pxShift = False, sequence = False):

    maneuverList = ['circle_natural_050', 'circle_natural_075', 'circle_natural_100', 'circle_natural_125', 'circle_natural_150']
    fileList     = ['050_grass.csv', '075_grass.csv', '100_grass.csv', '125_grass.csv', '150_grass.csv']

    for i in range(0, len(maneuverList)):

        maneuver = maneuverList[i]
        file = 'results_final/experiment_variance/' + fileList[i]
        print(maneuver)

        #########################

        write = True
        if os.path.isfile(file):
            while True:
                entry = input("Do you want to overwrite the existing file? [y/n] ")
                if entry == 'y' or entry == 'Y':
                    write = True
                    break
                elif entry == 'n' or entry == 'N':
                    write = False
                    break
                else:
                    print("Try again.")

        if write == True:
            ofile  = open(file, "w")
            writer = csv.writer(ofile, delimiter=',')

        # initialize the dataset
        data = DataSet(seq_length = seq_length, dataFolder = dataFolder, stack = stack, imType = imType, separation = separation, pxShift = pxShift, sequence = sequence)

        # initialize the model
        rm = ResearchModels(model, seq_length, saved_model, imType)

        # count the number of images in this folder
        if imType == 'OFF' or imType == 'ON': path = dataFolder + '/' + maneuver + '/' + imType + '/'
        elif imType == 'both': path = dataFolder + '/' + maneuver + '/' + 'ON' + '/' # just for counting frames
        else: path = dataFolder + '/' + maneuver + '/'
        pngs = len(glob.glob1(path,"*.png"))

        # starting image
        seqNum  = 12
        seqNum += seq_length - 1 + separation*seq_length

        while seqNum <= pngs:

            # get data, predict, and store
            X, y, Nef, var_img = data.test(maneuver, seqNum)
            output = rm.model.predict(X)

            # write output
            if write == True: writer.writerow([seqNum, float(output[0][0]), float(output[0][1]), Nef, var_img])

            # progress
            sys.stdout.write('\r' + '{0}\r'.format(int((seqNum/pngs)*100)),)
            sys.stdout.flush()

            # update sequence number
            seqNum += 1


if __name__ == '__main__':

    # input number frames
    seq_length = 2

    # name of the model
    model = 'FlowNetSimple'

    # path to dataset
    dataFolder = 'images_split_variance'

    # load a checkpoint
    saved_model = 'results_final/experiment_variance/ref.hdf5'

    # is the batch a sequence?
    if model == 'convLSTM':
        sequence = True
    else:
        sequence = False

    # image separation (if seq_length > 1)
    separation = 15
    
    # image type
    imType = 'both'

    # stack images
    stack = True

    # train the model
    check(model, saved_model = saved_model, seq_length = seq_length, dataFolder = dataFolder, stack = stack, imType = imType, separation = separation, sequence = sequence)