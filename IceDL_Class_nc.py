import os
import time
import numpy as np
import tensorflow as tf
from utils import read_data, scaler, rescaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import h5py
import netCDF4 as nc


class IceDL(object):
    def __init__(self,
                 sess,
                 model_function,
                 image_width=50,
                 image_height=50,
                 temporal_depth=20,
                 X_channels=4,
                 checkpoint_dir=None,
                 batch_size=10,
                 learning_rate=1E-4,
                 summaries_dir='',
                 specifications=''
                 ):
        self.sess = sess
        self.model_function = model_function
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.temporal_depth = temporal_depth
        self.summaries_dir = summaries_dir
        self.specifications = specifications
        self.X_channels = X_channels
        self.checkpoint_dir = checkpoint_dir
        self.X = tf.placeholder(tf.float32,
                                [None, self.temporal_depth, self.image_width,
                                 self.image_height, self.X_channels],
                                name='X')

        self.Ylabel = tf.placeholder(tf.float32,
                                     [None, self.temporal_depth, self.image_width, self.image_height, 1], name='Ylabel')
        self.pred, self.name = self.model_function(self.X)

        # self.loss = tf.losses.absolute_difference(self.Ylabel[:,:,:,:,0], self.pred[:,:,:,:,0])
        # self.loss = tf.losses.mean_squared_error(self.Ylabel[:,:,:,:,0], self.pred[:,:,:,:,0])
        #self.loss = tf.losses.mean_squared_error(tf.log1p(self.Ylabel), tf.log1pself.pred)
        self.loss = tf.losses.absolute_difference(tf.log1p(10*self.Ylabel), tf.log1p(10*self.pred))

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                            global_step=self.global_step)

        tf.add_to_collection('global_step', self.global_step)
        self.saver = tf.train.Saver()

        self.path = "/%s_%s_%s_%s" % (
            self.name, str(self.learning_rate), self.specifications, str(self.batch_size))

    def feed_dict(self, train, config, idx, X_train, Ylabel_train, X_test, Ylabel_test):
        if train:
            batch_X = X_train[idx * config.batch_size: (idx + 1) * config.batch_size]
            batch_Ylabel = Ylabel_train[idx * config.batch_size: (idx + 1) * config.batch_size]
        else:
            batch_X = X_test  # [idx * config.testbatch_size: (idx + 1) * config.testbatch_size]
            batch_Ylabel = Ylabel_test  # [idx * config.testbatch_size: (idx + 1) * config.testbatch_size]
        return {self.X: batch_X, self.Ylabel: batch_Ylabel}

    def train(self, config):

        train_dir = config.train_dir
        test_dir = config.test_dir

        # test_dir = config.test_dir

        X_train0, Ylabel_train0 = read_data(train_dir, augmentation=False)
        X_test0, Ylabel_test0 = read_data(test_dir, augmentation=False)
        X_train0, Ylabel_train0 = shuffle(X_train0, Ylabel_train0, random_state=0)
        # Ylabel_train0 = np.stack([Ylabel_train0], axis=-1)
        # Ylabel_test0 = np.stack([Ylabel_test0], axis=-1)
        X_train, X_test, Ylabel_train, Ylabel_test = scaler(X_train0, X_test0, Ylabel_train0, Ylabel_test0)
        print("maxs =", np.max(X_train[:, :, :, :, 0]), np.max(Ylabel_train))
        print(X_train.shape, X_test.shape, Ylabel_train.shape, Ylabel_test.shape)

        if self.load(self.checkpoint_dir, config):
            print(" [*] Load SUCCESS")
            print(self.global_step.eval())
            train_writer = tf.summary.FileWriter(
                self.summaries_dir + self.path + '/train')
            test_writer = tf.summary.FileWriter(
                self.summaries_dir + self.path + '/test')
        else:
            train_writer = tf.summary.FileWriter(
                self.summaries_dir + self.path + '/train',
                self.sess.graph)
            test_writer = tf.summary.FileWriter(
                self.summaries_dir + self.path + '/test')
            self.sess.run(tf.global_variables_initializer())
            print(" [!] Load failed...")

        with tf.name_scope('metrics'):
            with tf.name_scope('MSE'):
                MSE = tf.losses.mean_squared_error(self.Ylabel[:, :, :, :, 0], self.pred[:, :, :, :, 0])
                tf.summary.scalar('MSE', MSE)
            with tf.name_scope('MSE'):
                fMSE = tf.losses.mean_squared_error(self.Ylabel[:, -1, :, :, 0], self.pred[:, -1, :, :, 0])
                tf.summary.scalar('fMSE', fMSE)
            with tf.name_scope('MAE'):
                MAE = tf.losses.absolute_difference(self.Ylabel[:, :, :, :, 0], self.pred[:, :, :, :, 0])
                tf.summary.scalar('MAE', MAE)
            with tf.name_scope('fMAE'):
                fMAE = tf.losses.absolute_difference(self.Ylabel[:, -1, :, :, 0], self.pred[:, -1, :, :, 0])
                tf.summary.scalar('fMAE', fMAE)
            self.sess.run(tf.local_variables_initializer())

        counter = 0
        start_time = time.time()
        merged = tf.summary.merge_all()

        if config.is_train:
            print("Training...")

            for ep in range(config.epochs):

                batch_idxs = len(X_train) // config.batch_size
                for idx in range(batch_idxs):
                    counter += 1
                    summary, _ = self.sess.run([merged, self.train_op],
                                               feed_dict=self.feed_dict(True, config,
                                                                        idx, X_train,
                                                                        Ylabel_train, X_test,
                                                                        Ylabel_test))
                train_writer.add_summary(summary, self.global_step.eval() / batch_idxs)

                if ep % 10 == 0:
                    summary, loss = self.sess.run([merged, self.loss],
                                                  feed_dict=self.feed_dict(False, config,
                                                                           idx, X_train,
                                                                           Ylabel_train, X_test,
                                                                           Ylabel_test))
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                          % ((ep + 1), self.global_step.eval(), time.time() - start_time, loss))
                    test_writer.add_summary(summary, self.global_step.eval() / batch_idxs)

                if ep % 40 == 0:
                    self.save(config.checkpoint_dir, self.global_step, config)

                # tf.get_default_graph().finalize()
            self.save(config.checkpoint_dir, self.global_step, config)
            train_writer.close()
            test_writer.close()
        else:
            print("Testing.......")
            pred = self.sess.run(self.pred, feed_dict=self.feed_dict(False, config,
                                                                     0, X_train,
                                                                     Ylabel_train, X_test,
                                                                     Ylabel_test))

            predarray = np.array(pred)[:, :, :, :, 0]
            predarray = rescaler(Ylabel_train0, Ylabel_test0, predarray)
            trutharray = Ylabel_test0[:, :, :, :, 0]
            predtensor = tf.convert_to_tensor(predarray[:, -1, :, :], dtype='float32')
            truthtensor = tf.convert_to_tensor(trutharray[:, -1, :, :], dtype='float32')
            if config.save_h5py:
                h5f = h5py.File("TestResultsmodel3")
                h5f.create_dataset("predarray", data=predarray)
                h5f.create_dataset("trhutharray", data=trutharray)
                h5f.close()
            fMSE, upMSE = tf.metrics.mean_squared_error(truthtensor, predtensor)
            fMRE, upMRE = tf.metrics.mean_relative_error(truthtensor, predtensor, truthtensor)
            fMAE, upMAE = tf.metrics.mean_absolute_error(truthtensor, predtensor)
            self.sess.run(tf.local_variables_initializer())
            self.sess.run([upMAE, upMRE, upMSE])
            print('fMRE =', self.sess.run(fMRE))
            print('fMSE = ', self.sess.run(fMSE))
            print('fMAE=', self.sess.run(fMAE) * 100)
            if config.save_netcdf:
                # for k in range(predarray.shape[0]):
                path = "%s_%s" % (
                    self.name, self.specifications)

                file = nc.Dataset(config.result_fold + path + '.nc', mode='w')
                file.createDimension("x", size=50)
                file.createDimension("y", size=50)
                file.createDimension("t", size=20)
                for k in range(predarray.shape[0]):
                    label = file.createVariable("/labels" "/label_thk" + str(k), Ylabel_test0.dtype, ('t', 'x', 'y'))
                    label[:] = Ylabel_test0[k, :, :, :, 0]
                    prednc = file.createVariable("/preds/" + "pred_thk" + str(k), predarray.dtype, ('t', 'x', 'y'))
                    prednc[:] = predarray[k, :, :, :]
                    difference=file.createVariable("/difference/" + "diff-thk" + str(k), predarray.dtype, ('t', 'x', 'y'))
                    difference[:]=predarray[k,:,:,:]-Ylabel_test0[k, :, :, :, 0]
                file.close()

    def save(self, checkpoint_dir, step, config):
        model_name = "IceDL.model"
        model_dir = self.path
        checkpoint_dir = checkpoint_dir + self.path

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, config):
        print(" [*] Reading checkpoints...")
        model_dir = checkpoint_dir
        checkpoint_dir = model_dir + self.path

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # saver = tf.train.import_meta_graph('%s.meta'%os.path.join(checkpoint_dir, ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
