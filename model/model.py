
from __future__ import print_function
import numpy as np
import pickle
import sys
import tensorflow as tf
import random

from tensorflow import flags

from utils import mse, xent, conv_block, normalize
from model.maml import MAML
from model.data_generator import DataGenerator

FLAGS = flags.FLAGS


class ModelBuilder:
    def __init__(self):
        self.initilized = True

    def construct_model(self):
        test_num_updates = 10
        if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
            if FLAGS.train:
                data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
        else:
            data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
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
        return model, data_generator
