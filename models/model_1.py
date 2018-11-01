import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.ops.init_ops import random_normal_initializer

init=None#random_normal_initializer(0,0.1)
print(device_lib.list_local_devices())

activation=tf.nn.relu

def convnet3D(X):
    # input shape=(20, 50, 50, 3)
    conv1 = tf.layers.conv3d(X, 8,[3, 3, 3], activation=activation, kernel_initializer=init, name="conv3D_1")
    conv2 = tf.layers.conv3d(conv1, 16, [3, 3, 3], activation=activation,
                             name='conv3D_2')
    conv3 = tf.layers.conv3d(conv2, 32, [3, 3, 3], activation=activation,
                             name='conv3D_3')
    conv4 = tf.layers.conv3d(conv3, 64, [3, 3, 3], activation=activation, kernel_initializer=init,name='conv3D_4')
    conv5 = tf.layers.conv3d(conv4, 128, [3, 3, 3], activation=activation, kernel_initializer=init,name='conv3D_5')
    conv6 = tf.layers.conv3d(conv5, 256, [3, 3, 3], activation=activation,kernel_initializer=init,name='conv3D_6')
    conv7 = tf.layers.conv3d(conv6, 512, [3, 3, 3], activation=activation, kernel_initializer=init,name='conv3D_7')
    conv8 = tf.layers.conv3d(conv7, 512, [3, 3, 3], activation=activation,kernel_initializer=init, name='conv3D_8')
    conv9 = tf.layers.conv3d(conv8, 512, [3, 3, 3], activation=activation, kernel_initializer=init,name='conv3D_9')
    conv10 = tf.layers.conv3d(conv9, 512, [2, 3, 3], activation=activation,kernel_initializer=init, name='conv3D_10')
    return conv10


def deconvnet2D(encoder):
    deconv1 = tf.layers.conv3d_transpose(encoder, 256, [2, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_1')
    deconv2 = tf.layers.conv3d_transpose(deconv1, 128, [3, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_2')
    deconv3 = tf.layers.conv3d_transpose(deconv2, 64, [3, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_3')
    deconv4 = tf.layers.conv3d_transpose(deconv3, 32, [3, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_4')
    deconv5 = tf.layers.conv3d_transpose(deconv4, 16,  [3, 3, 3], activation=activation, strides=1,kernel_initializer=init, name='deconv2D_5')
    deconv6 = tf.layers.conv3d_transpose(deconv5, 8,  [3, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_6')
    deconv7 = tf.layers.conv3d_transpose(deconv6, 1, [3, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_7')
    deconv8 = tf.layers.conv3d_transpose(deconv7, 1, [3, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_8')
    deconv9 = tf.layers.conv3d_transpose(deconv8, 1, [3, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_9')
    deconv10 = tf.layers.conv3d_transpose(deconv9, 1,  [3, 3, 3], activation=activation, strides=1, kernel_initializer=init,name='deconv2D_10')
    return deconv10


def inference(x):
    dense_scenario = convnet3D(x)
    out = deconvnet2D(dense_scenario)
    return out, 'model1'