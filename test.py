import os
import numpy as np
import tensorflow as tf
from datasets import cifar10 as dataset
from models.nn import ShakeNet as ConvNet
from learning.evaluators import AccuracyEvaluator as Evaluator

# Load test set(val_set)
X_train, X_val, Y_train, Y_val = dataset.read_CIFAR10_subset()
train_set = dataset.DataSet(X_train, Y_train)
test_set = dataset.DataSet(X_val, Y_val)

# Sanity check
print('Test set stats:')
print(test_set.images.shape)


""" 2. Set test hyperparameters """
hp_d = dict()

# FIXME: Test hyperparameters
hp_d['batch_size'] = 128
hp_d['augment_pred'] = False

""" 3. Build graph, load weights, initialize a session and start test """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([32, 32, 3], 10, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, './tmp/model.ckpt')    # restore learned weights
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
