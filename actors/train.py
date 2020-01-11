from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

class Trainer:
    def __init__(self):
        self.initilized = True
            
    def train(self, model, saver, sess, exp_string, data_generator, resume_itr=0):
        SUMMARY_INTERVAL = 100
        SAVE_INTERVAL = 1000
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    
        if FLAGS.log:
            train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
        print('Done initializing, starting training.')
        prelosses, postlosses = [], []
    
        num_classes = data_generator.num_classes # for classification, 1 otherwise
        multitask_weights, reg_weights = [], []
    
        for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
            feed_dict = {}
            if itr < FLAGS.pretrain_iterations:
                input_tensors = [model.pretrain_op]
            else:
                input_tensors = [model.metatrain_op]
    
            if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
                input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
                if model.classification:
                    input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
    
            result = sess.run(input_tensors, feed_dict)
    
    
            if itr % SUMMARY_INTERVAL == 0:
                prelosses.append(result[-2])
                if FLAGS.log:
                    train_writer.add_summary(result[1], itr)
                postlosses.append(result[-1])
    
            if (itr!=0) and itr % PRINT_INTERVAL == 0:
                if itr < FLAGS.pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
                print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
                print(print_str)
                prelosses, postlosses = [], []
    
            if (itr!=0) and itr % SAVE_INTERVAL == 0:
                saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
    
            if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
                feed_dict = {}
                input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                result = sess.run(input_tensors, feed_dict)
                print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))
    
        saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
