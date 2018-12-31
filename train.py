import os
import numpy as np
import tensorflow as tf
from datasets import cifar10 as dataset
from models.nn import ShakeNet as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import AccuracyEvaluator as Evaluator

from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

# Load training set(train_set) and test set(val_set)
X_train, X_val, Y_train, Y_val = dataset.read_CIFAR10_subset()
train_set = dataset.DataSet(X_train, Y_train)
val_set = dataset.DataSet(X_val, Y_val)

# Sanity check
print('Training set stats:')
print(train_set.images.shape)

print('Validation set stats:')
print(val_set.images.shape)

""" 2. Set training hyperparameters """
hp_d = dict()

# FIXME: Training hyperparameters
hp_d['batch_size'] = 128
hp_d['num_epochs'] = 1800

hp_d['augment_train'] = True

hp_d['init_learning_rate'] = 0.2
hp_d['momentum'] = 0.9

# FIXME: Regularization hyperparameters
hp_d['weight_decay'] = 0.0001
hp_d['dropout_prob'] = 0.0

# FIXME: Evaluation hyperparameters
hp_d['score_threshold'] = 1e-4


""" 3. Build graph, initialize a session and start training """
# Initialize
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([32, 32, 3], 10, **hp_d)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)
