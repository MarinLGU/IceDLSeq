import os
import h5py

import numpy as np

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

xrange = range

FLAGS = tf.app.flags.FLAGS


def read_data(path, augmentation=False):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        X = np.array(hf.get('X'))
        Ylabel = np.array(hf.get('Ylabel')).astype('float32')
        X0 = np.copy(X)
        Ylabel0 = np.copy(Ylabel)
        if augmentation:
            for k in range(1, 4):
                rotX = np.rot90(X, k, (2, 3))
                rotYlabel = np.rot90(Ylabel, k, (1, 2))
                X0 = np.append(X0, rotX, axis=0)
                Ylabel0 = np.append(Ylabel0, rotYlabel, axis=0)

    return X0, Ylabel0


def scaler(X_train0, X_test0, Ylabel_train0, Ylabel_test0):
    X_train = np.copy(X_train0)
    X_test = np.copy(X_test0)
    Ylabel_train = np.copy(Ylabel_train0)
    Ylabel_test = np.copy(Ylabel_test0)
    concat0 = np.concatenate((X_train, X_test))
    airtempscaler = MinMaxScaler((0, 1)).fit(concat0[:, :, :, :, 0].max(axis=(0, 1)))
    prcptnscaler = MinMaxScaler((0, 1)).fit(concat0[:, :, :, :, 1].max(axis=(0, 1)))
    ocnscaler = MinMaxScaler((0, 1)).fit(concat0[:, :, :, :, 2].max(axis=(0, 1)))
    topgscaler = MinMaxScaler((-1, 1)).fit(concat0[:, :, :, :, 3].max(axis=(0, 1)))
    for b in range(X_test.shape[0]):
        for t in range(X_test.shape[1]):
            for c in range(X_test.shape[-1]):
                if c == 0:
                    X_test[b, t, :, :, 0] = airtempscaler.transform(X_test[b, t, :, :, 0])
                if c == 1:
                    X_test[b, t, :, :, 1] = prcptnscaler.transform(X_test[b, t, :, :, 1])
                if c == 2:
                    X_test[b, t, :, :, 2] = ocnscaler.transform(X_test[b, t, :, :, 2])
                if c == 3:
                    X_test[b, t, :, :, 3] = topgscaler.transform(X_test[b, t, :, :, 3])
    for b in range(X_train.shape[0]):
        for t in range(X_train.shape[1]):
            for c in range(X_train.shape[-1]):
                if c == 0:
                    X_train[b, t, :, :, 0] = airtempscaler.transform(X_train[b, t, :, :, 0])
                if c == 1:
                    X_train[b, t, :, :, 1] = prcptnscaler.transform(X_train[b, t, :, :, 1])
                if c == 2:
                    X_train[b, t, :, :, 2] = ocnscaler.transform(X_train[b, t, :, :, 2])
                if c == 3:
                    X_train[b, t, :, :, 3] = topgscaler.transform(X_train[b, t, :, :, 3])

    concat0 = np.concatenate((Ylabel_train0, Ylabel_test0))
    icescaler = MinMaxScaler((0, 1)).fit(concat0[:, :, :, :, 0].max(axis=(0, 1)))
    for b in range(Ylabel_train0.shape[0]):
        for t in range(Ylabel_train0.shape[1]):
            Ylabel_train[b, t, :, :, 0] = icescaler.transform((Ylabel_train0[b, t, :, :, 0]))
    for b in range(Ylabel_test0.shape[0]):
        for t in range(Ylabel_test0.shape[1]):
            Ylabel_test[b, t, :, :, 0] = icescaler.transform((Ylabel_test0[b, t, :, :, 0]))

    return X_train, X_test, Ylabel_train, Ylabel_test


def rescaler(Ylabel_train0, Ylabel_test0, pred):
    concat = np.concatenate((Ylabel_train0, Ylabel_test0))
    icescaler = MinMaxScaler((0, 1)).fit(concat[:, :, :, :, 0].max(axis=(0, 1)))
    for b in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            pred[b, t, :, :] = icescaler.inverse_transform((pred[b, t, :, :]))
    return pred


def make_data(sess, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)
