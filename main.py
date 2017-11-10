import os
import sys
import h5py
import time
from data import DataSet
from models import ResearchModels
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger


def train(model, saved_model = None, transfer_learning = None, seq_length = 150, dataFolder = 'dataset', stack = True, imType = None, separation = 0, pxShift = False, sequence = False):

    # generate directories
    pathCP   = './data/checkpoints/'
    pathLogs = './data/logs'
    if not os.path.exists(pathCP):
        os.makedirs(pathCP)
    if not os.path.exists(pathLogs):
        os.makedirs(pathLogs)

    # save the model 
    checkpointer = ModelCheckpoint(
        filepath = pathCP + model + '-' + '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose = 1,
        save_best_only = True,
        save_weights_only = True)

    # tensorboard
    tb = TensorBoard(log_dir = pathLogs)

    # stop when we stop learning
    early_stopper = EarlyStopping(patience = 10)

    # save results.
    timestamp = time.time()
    csv_logger = CSVLogger(pathLogs + model + '-' + 'training-' + str(timestamp) + '.log')

    # initialize the dataset
    data = DataSet(seq_length = seq_length, dataFolder = dataFolder, stack = stack, imType = imType, separation = separation, pxShift = pxShift, sequence = sequence)

    # set training variables
    nb_epoch = 1000
    batch_size = 8

    # set generators
    generator     = data.train_generator(batch_size)
    val_generator = data.validate_generator(batch_size)

    # initialize the model
    rm = ResearchModels(model, seq_length, saved_model, imType, transfer_learning = transfer_learning)

    # use fit generator for training
    rm.model.fit_generator(
        generator = generator,
        steps_per_epoch = 2500,
        epochs  = nb_epoch,
        verbose = 1,
        callbacks = [checkpointer, tb, csv_logger],
        validation_data  = val_generator,
        validation_steps = 1000)
    

if __name__ == '__main__':

    # input number frames
    seq_length = 1

    # name of the model (FlowNetSimple, convLSTM, convLSTM_dt, VGG_16)
    model = 'FlowNetSimple'

    # path to dataset
    dataFolder = '../dvs_simulator/generated_datasets/images/temporal_250'

    # load a checkpoint
    saved_model = None
    #saved_model = 'data/checkpoints/convLSTM-.008-0.607.hdf5'
    transfer_learning = None
    if saved_model is not None:
        transfer_learning = None

    # is the batch a sequence?
    if model == 'convLSTM' or model == 'VGG_16':
        sequence = True
    else:
        sequence = False

    # image separation (if seq_length > 1)
    separation = 1
    
    # image type (ON, OFF, BOTH, None)
    imType = None

    # stack images
    stack = False

    # train the model
    train(model, saved_model = saved_model, transfer_learning = transfer_learning, seq_length = seq_length, dataFolder = dataFolder, stack = stack, imType = imType, separation = separation, sequence = sequence)
