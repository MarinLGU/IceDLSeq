import tensorflow as tf
from tensorflow.python.client import device_lib
#from tensorflow.contrib.layers import xavier_initializer as xav
from tensorflow.python.ops.init_ops import random_normal_initializer
xav=random_normal_initializer(0,0.1)

print(device_lib.list_local_devices())

activation=tf.nn.relu

def convnet3D(X):
    # input shape=(20, 50, 50, 3)
    conv1 = tf.layers.conv3d(X, 8, [5, 1, 1], activation=activation, name="conv3D_1", kernel_initializer=xav)
    conv2 = tf.layers.conv3d(conv1, 16, [5, 1, 1], activation=activation,
                             name='conv3D_2')
    conv3 = tf.layers.conv3d(conv2, 32, [5, 1, 1], activation=activation,
                             name='conv3D_3', kernel_initializer=xav)
    conv4 = tf.layers.conv3d(conv3, 64, [5, 1, 1], activation=activation, name='conv3D_4', kernel_initializer=xav)
    conv5 = tf.layers.conv3d(conv4, 128, [4, 1, 1], activation=activation, name='conv3D_5', kernel_initializer=xav)
    conv6 = tf.layers.conv3d(conv5, 256, [1, 3, 3], activation=activation, padding='SAME', kernel_initializer=xav)
    conv7 = tf.layers.conv3d(conv6, 512, [1, 3, 3], activation=activation, padding='SAME', kernel_initializer=xav)
    return conv7


def deconvnet2D(encoder):
    deconv1 = tf.layers.conv3d_transpose(encoder, 256, [1, 3, 3], activation=activation, strides=1, name='deconv2D_1',
                                         padding='SAME', kernel_initializer=xav)
    deconv2 = tf.layers.conv3d_transpose(deconv1, 128, [1, 3, 3], activation=activation, strides=1, name='deconv2D_2',
                                         padding='SAME', kernel_initializer=xav)
    deconv3 = tf.layers.conv3d_transpose(deconv2, 64, [4, 1, 1], activation=activation, strides=1, name='deconv2D_3', kernel_initializer=xav)
    deconv4 = tf.layers.conv3d_transpose(deconv3, 32, [5, 1, 1], activation=activation, strides=1, name='deconv2D_4', kernel_initializer=xav)
    deconv5 = tf.layers.conv3d_transpose(deconv4, 16, [5, 1, 1], activation=activation, strides=1, name='deconv2D_5', kernel_initializer=xav)
    deconv6 = tf.layers.conv3d_transpose(deconv5, 8, [5, 1, 1], activation=activation, strides=1, name='deconv2D_6', kernel_initializer=xav)
    deconv7 = tf.layers.conv3d_transpose(deconv6, 1, [5, 1, 1], activation=activation, strides=1, name='deconv2D_7', kernel_initializer=xav)
    return deconv7


def inference(x):
    dense_scenario = convnet3D(x)
    out = deconvnet2D(dense_scenario)
    return out, 'model0'
