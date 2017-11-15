import os
import sys
import h5py
import time
from data import DataSet
from models import ResearchModels
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import numpy as np 

def train(model, saved_model = None, transfer_learning = None, seq_length = 150, dataFolder = 'dataset', imType = None, separation = 0, pxShift = False, sequence_batch = False):

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
    data = DataSet(seq_length = seq_length, dataFolder = dataFolder, imType = imType, separation = separation, pxShift = pxShift, sequence_batch = sequence_batch)

    # set training variables
    nb_epoch = 100
    nb_samples = 50
    nb_validation = 20
    
    batch_size = 5

    # initialize the model
    rm = ResearchModels(model, 1, saved_model, imType, transfer_learning = transfer_learning)

    # set generators
    generator     = data.train_generator(1)
    val_generator = data.validate_generator(1)

    # train the model
    for e in range(0,nb_epoch):
        for s in range(0,nb_samples):

            lamb = 1.0
            sum_loss = 0.
            for b in range(0,seq_length):

                X, y = data.data_function(b, batch_size, lamb, 'train')
                output = rm.model.predict(X, batch_size=batch_size, verbose=0)
                loss = rm.model.fit(X, y, batch_size=batch_size, epochs = 1, verbose = 0)
                lamb = output[0][-1]
                sum_loss += loss.history['loss'][0]
        train_loss = sum_loss/seq_length/nb_samples
        
        for v in range(0,nb_validation):
            
            lamb = 1.0
            sum_loss = 0.
            for b in range(0,seq_length):
                X, y = data.data_function(b, batch_size, lamb, 'train')
                output = rm.model.predict(X, batch_size=batch_size, verbose=0)
                loss = rm.model.fit(X, y, batch_size=batch_size, epochs = 1, verbose = 0)
                lamb = output[0][-1]
                sum_loss += loss.history['loss'][0]
                val_loss = sum_loss/seq_length/nb_validation
        print 'Epoch ' + str(e) + ', Training Loss: ' + str(train_loss) + ', Validation Loss: ' + str(val_loss)

    # use fit generator for training
    # rm.model.fit_generator(
    #     generator = generator,
    #     steps_per_epoch = 2500,
    #     epochs  = nb_epoch,
    #     verbose = 1,
    #     callbacks = [checkpointer, tb, csv_logger],
    #     validation_data  = val_generator,
    #     validation_steps = 1000)
    

if __name__ == '__main__':

    # input number frames
    seq_length = 2

    # name of the model (FlowNetSimple, convLSTM, convLSTM_dt, VGG_16)
    model = 'FlowNetSimpleRNN'

    # path to dataset
    dataFolder = '../dvs_simulator/generated_datasets/fede'#../dvs_simulator/generated_datasets/images/temporal_250'

    # load a checkpoint
    saved_model = None
    #saved_model = 'data/checkpoints/convLSTM-.008-0.607.hdf5'
    transfer_learning = None
    if saved_model is not None:
        transfer_learning = None

    # is the batch a sequence?
    if model == 'convLSTM' or model == 'VGG_16':
        sequence_batch = True
    else:
        sequence_batch = False

    # image separation (if seq_length > 1)
    separation = 1
    
    # image type (ON, OFF, BOTH, None)
    imType = None

    # train the model
    train(model, saved_model = saved_model, transfer_learning = transfer_learning, seq_length = seq_length, dataFolder = dataFolder, imType = imType, separation = separation, sequence_batch = sequence_batch)
