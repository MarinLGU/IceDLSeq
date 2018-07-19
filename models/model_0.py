import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

activation=tf.nn.relu

def convnet3D(X):
    # input shape=(20, 50, 50, 3)
    conv1 = tf.layers.conv3d(X, 8, [5, 1, 1], activation=activation, name="conv3D_1")
    conv2 = tf.layers.conv3d(conv1, 16, [5, 1, 1], activation=activation,
                             name='conv3D_2')
    conv3 = tf.layers.conv3d(conv2, 32, [5, 1, 1], activation=activation,
                             name='conv3D_3')
    conv4 = tf.layers.conv3d(conv3, 64, [5, 1, 1], activation=activation, name='conv3D_4')
    conv5 = tf.layers.conv3d(conv4, 128, [4, 1, 1], activation=activation, name='conv3D_5')
    conv6 = tf.layers.conv3d(conv5, 256, [1, 3, 3], activation=activation, padding='SAME')
    conv7 = tf.layers.conv3d(conv6, 512, [1, 3, 3], activation=activation, padding='SAME')
    return conv7


def deconvnet2D(encoder):
    deconv1 = tf.layers.conv3d_transpose(encoder, 256, [1, 3, 3], activation=activation, strides=1, name='deconv2D_1',
                                         padding='SAME')
    deconv2 = tf.layers.conv3d_transpose(deconv1, 128, [1, 3, 3], activation=activation, strides=1, name='deconv2D_2',
                                         padding='SAME')
    deconv3 = tf.layers.conv3d_transpose(deconv2, 64, [4, 1, 1], activation=activation, strides=1, name='deconv2D_3')
    deconv4 = tf.layers.conv3d_transpose(deconv3, 32, [5, 1, 1], activation=activation, strides=1, name='deconv2D_4')
    deconv5 = tf.layers.conv3d_transpose(deconv4, 16, [5, 1, 1], activation=activation, strides=1, name='deconv2D_5')
    deconv6 = tf.layers.conv3d_transpose(deconv5, 8, [5, 1, 1], activation=activation, strides=1, name='deconv2D_6')
    deconv7 = tf.layers.conv3d_transpose(deconv6, 1, [5, 1, 1], activation=activation, strides=1, name='deconv2D_7')
    return deconv7


def inference(x):
    dense_scenario = convnet3D(x)
    out = deconvnet2D(dense_scenario)
    return out, 'model0'
