import sys, os
import numpy as np
import scipy.io as sio

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')

from sklearn.model_selection import train_test_split

def load_samples_label(dataPath):
    X_samples = []
    Y_label = []
    for sampleFile in os.listdir(dataPath):
        sampleFileAndPath = dataPath + "/" + sampleFile

        if ".mat" in sampleFileAndPath:
            data_dic = sio.loadmat(sampleFileAndPath)
            samples = np.array(data_dic['samples'])
            labels = np.array(data_dic['label'])
            X_samples = np.append(X_samples, samples[:,:])
            Y_label = np.append(Y_label, labels[:,:])
    return X_samples.reshape((-1, 36)), Y_label

def select_positive_negative_samples_randmly(samples, labels):
    dataset = np.hstack((samples, np.expand_dims(labels, axis=1)))
    dataset_0 = dataset[np.where(labels==0)]
    dataset_1 = dataset[np.where(labels==1)]
    select_0 = np.random.choice(dataset_0.shape[0],dataset_1.shape[0], replace=False)
    dataset_0_selected = dataset_0[select_0, :]
    dataset_test_train = np.vstack((dataset_0_selected, dataset_1))
    np.random.shuffle(dataset_test_train)
    np.random.shuffle(dataset_test_train)
    return dataset_test_train

def data():
    X_norm_sample, Y_norm_label = load_samples_label('/media/hpc/codes/GitLab/sigpy_github/data/train_test_dataset')
    norm_samples_label = select_positive_negative_samples_randmly(X_norm_sample, Y_norm_label)
    X_train, X_test, y_train, y_test = train_test_split(norm_samples_label[:, :-1],
                                            norm_samples_label[:, -1], test_size=0.3, random_state=42)

    X_pacing_sample, Y_pacing_sample = load_samples_label('/media/hpc/codes/GitLab/sigpy_github/data/train_test_dataset/test')
    pacing_samples_label = select_positive_negative_samples_randmly(X_pacing_sample, Y_pacing_sample)
    #
    X_test = np.vstack((X_test, pacing_samples_label[:,:-1]))
    y_test = np.hstack((y_test, pacing_samples_label[:, -1]))
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)
    return X_train, y_train, X_test, y_test

def data_with_pacing_training():
    X_norm_sample, Y_norm_label = load_samples_label('/media/hpc/codes/GitLab/sigpy_github/data/train_test_dataset')
    norm_samples_label = select_positive_negative_samples_randmly(X_norm_sample, Y_norm_label)

    X_pacing_sample, Y_pacing_sample = load_samples_label('/media/hpc/codes/GitLab/sigpy_github/data/train_test_dataset/test')
    pacing_samples_label = select_positive_negative_samples_randmly(X_pacing_sample, Y_pacing_sample)
    dataset = np.vstack((norm_samples_label, pacing_samples_label))
    np.random.shuffle(dataset)
    np.random.shuffle(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1],
                                        dataset[:, -1], test_size=0.3, random_state=42)
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)
    return X_train, y_train, X_test, y_test

def d2model(X_train, y_train, X_test, y_test):
    model = Sequential()
    #1st Convolution Layer
    model.add(Convolution2D(32, 3, 3, activation='relu',
                            kernel_initializer='glorot_uniform',
                            input_shape=(1, 6, 6)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #2nd Convolution Layer
    model.add(Convolution2D(32, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Dropout(0.5))

    #Fully connected layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # Output
    model.add(Dense(2, activation='softmax'))

    # Optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #Compile the model
    model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

    # Train the network
    model.fit(X_train, y_train, nb_epoch=15, verbose=1)
    scores, acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', acc)


def d1model(X_train, y_train, X_test, y_test):
    model = Sequential()
    #1st Convolution Layer
    model.add(Convolution1D(64,9, activation='relu',
                            kernel_initializer='glorot_uniform',
                            input_shape=(36, 1)))
    model.add(MaxPooling1D(pool_size=2))

    #2nd Convolution Layer
    model.add(Convolution1D(32, 3, activation='relu'))
    model.add(Dropout(0.5))

    #Fully connected layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output
    model.add(Dense(2, activation='softmax'))

    # Optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    #Compile the model
    model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

    # Train the network
    model.fit(np.expand_dims(X_train, axis=2), y_train, nb_epoch=15)
    scores, acc = model.evaluate(np.expand_dims(X_test, axis=2), y_test)
    print('Test accuracy:', acc)

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__' :
    X_train, y_train, X_test, y_test = data_with_pacing_training()
    d1model(X_train, y_train, X_test, y_test)

    X_train = X_train.reshape((-1, 1, 6, 6))
    X_test = X_test.reshape((-1, 1, 6, 6))
    d2model(X_train, y_train, X_test, y_test)
