import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.ops.init_ops import random_normal_initializer
print(device_lib.list_local_devices())


# def activation(tensor):
#     return (tf.tanh(tf.scalar_mul(2, tensor) - tf.ones_like(tensor)) +tf.ones_like(tensor))/tf.scalar_mul(2,tf.ones_like(tensor))

init=None #random_normal_initializer(0,0.05)
activation=tf.nn.relu

def convnet3D(X):
    # input shape=(20, 50, 50, 3)
    conv0=tf.layers.conv3d(X, 16, [3, 3, 3], strides=(1,1,1), kernel_initializer=init, activation=activation, padding='SAME', name="conv3D_0")
    conv1 = tf.layers.conv3d(conv0, 32, [3, 3, 3], strides=(1,1,1), kernel_initializer=init, activation=activation, name="conv3D_1") #18,48,48
    conv2 = tf.layers.conv3d(conv1, 64, [3, 3, 3], strides=(1,1,1),activation=activation, #16, 46,46
                             name='conv3D_2')
    conv30=tf.layers.conv3d(conv2, 128, [3, 3, 3], strides =(2,2,2), kernel_initializer=init,activation=activation,
                             name='conv3D_30')
    conv3 = tf.layers.conv3d(conv30, 128, [3, 3, 3], kernel_initializer=init,activation=activation, padding='SAME',
                             name='conv3D_3') #7, 22, 22
    conv40=tf.layers.conv3d(conv3, 256, [3, 3, 3], kernel_initializer=init,activation=activation, padding='SAME',
                             name='conv3D_40')
    conv4 = tf.layers.conv3d(conv40, 256, [3, 3, 3], strides=(2,2,2),kernel_initializer=init, activation=activation, name='conv3D_4') #3, 10, 10
    conv5 = tf.layers.conv3d(conv4, 512, [3, 3, 3], kernel_initializer=init,activation=activation, name='conv3D_5') #1, 8, 8
    fc1=tf.layers.dense(conv5, 512, activation=activation)
    fc2=tf.layers.dense(fc1, 512, activation = activation)
    return fc2


def deconvnet2D(encoder):
    deconv0 =  tf.layers.conv3d_transpose(encoder, 256, 3,kernel_initializer=init, activation=activation, strides=1, name='deconv2D_0', padding='SAME')
    deconv1 = tf.layers.conv3d_transpose(deconv0, 128, 3,kernel_initializer=init, activation=activation, strides=1, name='deconv2D_1')
    deconv2 = tf.layers.conv3d_transpose(deconv1, 64, 3, kernel_initializer=init,activation=activation, strides=2, name='deconv2D_2')
    deconv3 = tf.layers.conv3d_transpose(deconv2, 32, [3,4,4],kernel_initializer=init, activation=activation, strides=2, name='deconv2D_3')
    deconv4 = tf.layers.conv3d_transpose(deconv3, 16, [4,4,4], kernel_initializer=init, activation=activation, strides=1, name='deconv2D_4')
    deconv5 = tf.layers.conv3d_transpose(deconv4, 1, [3,4,4], kernel_initializer=init, activation=activation, strides=1, name='deconv2D_5')
    # deconv6 = tf.layers.conv3d_transpose(deconv5, 8, [5, 1, 1], activation=activation, strides=1, name='deconv2D_6')
    # deconv7 = tf.layers.conv3d_transpose(deconv6, 1, [5, 1, 1], activation=activation, strides=1, name='deconv2D_7')
    return deconv5


def inference(x):
    dense_scenario = convnet3D(x)
    out = deconvnet2D(dense_scenario)
    return out, 'model3'