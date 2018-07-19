import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from IceDL_Class_nc import IceDL
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
from models.model_0 import inference


flags = tf.app.flags

flags.DEFINE_float("learning_rate", 1E-3, "leaning rate")
flags.DEFINE_integer("epochs",450, "number of epochs")
flags.DEFINE_integer("inference_start", 40, "epoch when inference train op starts instead of train mode")
flags.DEFINE_integer("batch_size", 23, "batch size")
flags.DEFINE_integer("testbatch_size", 69, "batch size")
flags.DEFINE_integer("image_width", 50, "Image Width in pixel")
flags.DEFINE_integer("image_height", 50, "Image height in pixel")
flags.DEFINE_integer("temporal_depth", 20, "Image temporal depth")
flags.DEFINE_integer("X_channels", 4, "Number of climate scenario")


flags.DEFINE_string("specifications", "relulogMAEstop450", "specification for run identification")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of chepoints directory")
flags.DEFINE_string("train_dir", "run2train", "h5 train dataset file directory")
flags.DEFINE_string("test_dir", "run2test", "h5 test dataset file directory" )
flags.DEFINE_string("result_fold", "./DSManagement/run2/", "fold where save png outputs")
flags.DEFINE_string("summaries_dir", "summaries", "Tensorboard Sumaries Directory" )

flags.DEFINE_boolean("is_train", False, "True if training, False if testing")
flags.DEFINE_boolean("save_netcdf", True, "True if want to save netcdf output while testing")
flags.DEFINE_boolean("save_h5py", False, "True if want to save h5py output while testing")

FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
        iceDL = IceDL(sess,
                      inference,
                      image_width=FLAGS.image_width,
                      image_height=FLAGS.image_height,
                      temporal_depth=FLAGS.temporal_depth,
                      X_channels=FLAGS.X_channels,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      batch_size=FLAGS.batch_size,
                      learning_rate=FLAGS.learning_rate,
                      specifications=FLAGS.specifications,
                      summaries_dir=FLAGS.summaries_dir,
                      )
        iceDL.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
