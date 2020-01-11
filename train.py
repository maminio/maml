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
        if FLAGS.datasource == 'sinusoid':
            PRINT_INTERVAL = 1000
            TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
        else:
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
            if 'generate' in dir(data_generator):
                batch_x, batch_y, amp, phase = data_generator.generate()
    
                if FLAGS.baseline == 'oracle':
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    for i in range(FLAGS.meta_batch_size):
                        batch_x[i, :, 1] = amp[i]
                        batch_x[i, :, 2] = phase[i]
    
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}
    
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
    
            # sinusoid is infinite data, so no need to test on meta-validation set.
            if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
                if 'generate' not in dir(data_generator):
                    feed_dict = {}
                    if model.classification:
                        input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                    else:
                        input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
                else:
                    batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                    inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                    inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                    labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                    labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                    feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                    if model.classification:
                        input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                    else:
                        input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]
    
                result = sess.run(input_tensors, feed_dict)
                print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))
    
        saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))
