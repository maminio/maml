import csv  # <---- 
import numpy as np # <---- 
import pickle
import random
import tensorflow as tf
from tensorflow import flags

from model.data_generator import DataGenerator
from model.maml import MAML
from model.model import ModelBuilder
from actors.test import Tester
from utils import exp_init

######################## FLAGS ########################
FLAGS = flags.FLAGS
training_type = '5_way_1_shot'
if(training_type == '5_way_1_shot'):
    flags.DEFINE_string('datasource', 'omniglot', 'sinusoid or omniglot or miniimagenet')
    flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot 
    flags.DEFINE_integer('meta_batch_size', 32, 'number of tasks sampled per meta-update')
    flags.DEFINE_integer('update_batch_size', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
    flags.DEFINE_float('update_lr', 0.4, 'step size alpha for inner gradient update.') # 0.1 for omniglot
    flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
    flags.DEFINE_string('logdir', 'logs/omniglot5way/', 'directory for summaries and checkpoints.')

flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')


## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', False, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

################################################################################################

test_num_updates = 600

#################################### INIT MODEL ####################################

builder = ModelBuilder()
model, data_generator = builder.construct_model()

################## TENSORFLOW SESSION INITIALIZATION ##################

saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
sess = tf.InteractiveSession()

################ TESTING ################
exp_string = exp_init()
resume_itr = 0

tf.global_variables_initializer().run()
tf.train.start_queue_runners()
tester = Tester()
tester.test(model, saver, sess, exp_string, data_generator, test_num_updates)
##########################################
