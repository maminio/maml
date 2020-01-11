import csv  # <---- 
import numpy as np # <---- 
import pickle
import random
import tensorflow as tf
from data_generator import DataGenerator
from tensorflow import flags
from maml import MAML
from train import Trainer 

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
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot




#################################### MAIN ####################################


test_num_updates = 10
#data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
dim_output = data_generator.dim_output
tf_data_load = True
num_classes = data_generator.num_classes
dim_input = data_generator.dim_input


if FLAGS.train: # only construct training model if needed
    random.seed(5)
    image_tensor, label_tensor = data_generator.make_data_tensor()
    inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
    inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
    labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
    labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
    input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

#  ----> CHECK WHY RANDOM SEED IS CHANGING
random.seed(6)
image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}



################# MODEL INITILIZATION ##################

model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
model.construct_model(input_tensors=input_tensors, prefix='metatrain_')

model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

model.summ_op = tf.summary.merge_all()
########################################################

################## TENSORFLOW SESSION INITIALIZATION ##################
saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
sess = tf.InteractiveSession()
#######################################################################


if FLAGS.train_update_batch_size == -1:
    FLAGS.train_update_batch_size = FLAGS.update_batch_size
if FLAGS.train_update_lr == -1:
    FLAGS.train_update_lr = FLAGS.update_lr

exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

if FLAGS.num_filters != 64:
    exp_string += 'hidden' + str(FLAGS.num_filters)
if FLAGS.max_pool:
    exp_string += 'maxpool'
if FLAGS.stop_grad:
    exp_string += 'stopgrad'

if FLAGS.norm == 'batch_norm':
    exp_string += 'batchnorm'
elif FLAGS.norm == 'layer_norm':
    exp_string += 'layernorm'
elif FLAGS.norm == 'None':
    exp_string += 'nonorm'
else:
    print('Norm setting not recognized.')


resume_itr = 0
################ TRAINING ################
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
trainer = Trainer()
trainer.train(model, saver, sess, exp_string, data_generator, resume_itr)
##########################################


FLAGS.pretrain_iterations
