
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

# calculated for omniglot
NUM_TEST_POINTS = 600

class Tester:
    def __init__(self):
        self.initilized = True

    def test(self,model, saver, sess, exp_string, data_generator, test_num_updates):
        num_classes = data_generator.num_classes # for classification, 1 otherwise

        np.random.seed(1)
        random.seed(1)

        metaval_accuracies = []

        for _ in range(NUM_TEST_POINTS):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
            metaval_accuracies.append(result)

        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

        print('Mean validation accuracy/loss, stddev, and confidence intervals')
        print((means, stds, ci95))

        out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
        out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
        with open(out_pkl, 'wb') as f:
            pickle.dump({'mses': metaval_accuracies}, f)
        with open(out_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['update'+str(i) for i in range(len(means))])
            writer.writerow(means)
            writer.writerow(stds)
            writer.writerow(ci95)
