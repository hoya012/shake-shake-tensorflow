import time
from abc import abstractmethod
import tensorflow as tf
import numpy as np
import math
from models.layers import *


class ConvNet(object):
    """Base class for Convolutional Neural Networks."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Model initializer.
        :param input_shape: tuple, the shape of inputs (H, W, C), ranged [0.0, 1.0].
        :param num_classes: int, the number of classes.
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(tf.float32, [None] + [num_classes])
        self.is_train = tf.placeholder(tf.bool)

        # Build model and loss function
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        This should be implemented.
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
            - batch_size: int, batch size for each iteration.
        :return _y_pred: np.ndarray, shape: (N, num_classes).
        """
        batch_size = kwargs.pop('batch_size', 128)

        if dataset.labels is not None:
            assert len(dataset.labels.shape) > 1, 'Labels must be one-hot encoded.'
        num_classes = int(self.y.get_shape()[-1])

        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps*batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False,
                                      augment=False, is_train=False)

            # Compute predictions
            y_pred = sess.run(self.pred, feed_dict={self.X: X, self.is_train: False})    # (N, num_classes)

            _y_pred.append(y_pred)
        if verbose:
            print('Total prediction time(sec): {}'.format(time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)    # (N, num_classes)

        return _y_pred

class ShakeNet(ConvNet):
    """ShakeNet class."""

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building ShakeNet.
            - batch_size: int, the batch size.
        :return d: dict, containing outputs on each layer.
        """
        d = dict()    # Dictionary to save intermediate values returned from each layer.
        batch_size = kwargs.pop('batch_size', 128)
        num_classes = int(self.y.get_shape()[-1])

        # input
        X_input = self.X

        # first residual block's channels (26 2x32d --> 32)
        first_channel = 32 

        # the number of residual blocks (it means (depth-2)/6, i.e. 26 2x32d --> 4)
        num_blocks = 4

        # conv1 - batch_norm1
        with tf.variable_scope('conv1'):
            d['conv1'] = conv_layer_no_bias(X_input, 3, 1, 16, padding='SAME')
            print('conv1.shape', d['conv1'].get_shape().as_list())

        with tf.variable_scope('batch_norm1'):
            d['batch_norm1'] = batch_norm(d['conv1'], is_training = self.is_train)
            print('batch_norm1.shape', d['batch_norm1'].get_shape().as_list())

        # shake stage 1
        with tf.variable_scope('shake_s1'):
           d['shake_s1'] = self.shake_stage(d['batch_norm1'], first_channel, num_blocks, 1, batch_size, d)
           print('shake_s1.shape', d['shake_s1'].get_shape().as_list())

        # shake stage 2
        with tf.variable_scope('shake_s2'):
           d['shake_s2'] = self.shake_stage(d['shake_s1'], first_channel * 2, num_blocks, 2, batch_size, d)
           print('shake_s2.shape', d['shake_s2'].get_shape().as_list())

        # shake stage 3 with relu
        with tf.variable_scope('shake_s3'):
           d['shake_s3'] = tf.nn.relu(self.shake_stage(d['shake_s2'], first_channel * 4, num_blocks, 2, batch_size, d))
           print('shake_s3.shape', d['shake_s3'].get_shape().as_list())
       
        d['avg_pool_shake_s3'] = tf.reduce_mean(d['shake_s3'], reduction_indices=[1, 2])
        print('avg_pool_shake_s3.shape', d['avg_pool_shake_s3'].get_shape().as_list())

        # Flatten feature maps
        f_dim = int(np.prod(d['avg_pool_shake_s3'].get_shape()[1:]))
        f_emb = tf.reshape(d['avg_pool_shake_s3'], [-1, f_dim])
        print('f_emb.shape', f_emb.get_shape().as_list())

        with tf.variable_scope('fc1'):
           d['logits'] = fc_layer(f_emb, num_classes)
           print('logits.shape', d['logits'].get_shape().as_list())
 
        # softmax
        d['pred'] = tf.nn.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments for regularization term.
            - weight_decay: float, L2 weight decay regularization coefficient.
        :return tf.Tensor.
        """
        weight_decay = kwargs.pop('weight_decay', 0.0001)
        variables = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # Softmax cross-entropy loss function
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        softmax_loss = tf.reduce_mean(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss

    def shake_stage(self, x, output_filters, num_blocks, stride, batch_size, d):
        """
        Build sub stage with many shake blocks.
        :param x: tf.Tensor, input of shake_stage, shape: (N, H, W, C).
        :param output_filters: int, the number of output filters in shake_stage.
        :param num_blocks: int, the number of shake_blocks in one shake_stage.
        :param stride: int, the stride of the sliding window to be applied shake_block's branch. 
        :param batch_size: int, the batch size.
        :param d: dict, the dictionary for saving outputs of each layers.
        :return tf.Tensor.
        """

        shake_stage_idx = int(math.log2(output_filters // 16))  #FIXME if you change 'first_channel' parameter

        for block_idx in range(num_blocks):
           stride_block = stride if (block_idx == 0) else 1
           with tf.variable_scope('shake_s{}_b{}'.format(shake_stage_idx, block_idx)):
              x = self.shake_block(x, shake_stage_idx, block_idx, output_filters, stride_block, batch_size)
              d['shake_s{}_b{}'.format(shake_stage_idx, block_idx)] = x

        return d['shake_s{}_b{}'.format(shake_stage_idx, num_blocks-1)]

    def shake_block(self, x, shake_stage_idx, block_idx, output_filters, stride, batch_size):
        """
        Build one shake-shake blocks with branch and skip connection.
        :param x: tf.Tensor, input of shake_block, shape: (N, H, W, C).
        :param shake_layer_idx: int, the index of shake_stage.
        :param block_idx: int, the index of shake_block.
        :param output_filters: int, the number of output filters in shake_block.
        :param stride: int, the stride of the sliding window to be applied shake_block's branch. 
        :param batch_size: int, the batch size.
        :return tf.Tensor.
        """

        num_branches = 2

        # Generate random numbers for scaling the branches.
        
        rand_forward = [
          tf.random_uniform([batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32) for _ in range(num_branches)
        ]
        rand_backward = [
          tf.random_uniform([batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32) for _ in range(num_branches)
        ]

        # Normalize so that all sum to 1.
        total_forward = tf.add_n(rand_forward)
        total_backward = tf.add_n(rand_backward)
        rand_forward = [samp / total_forward for samp in rand_forward]
        rand_backward = [samp / total_backward for samp in rand_backward]
        zipped_rand = zip(rand_forward, rand_backward)

        branches = []
        for branch, (r_forward, r_backward) in enumerate(zipped_rand):
            with tf.variable_scope('shake_s{}_b{}_branch_{}'.format(shake_stage_idx, block_idx, branch)):
                b = self.shake_branch(x, output_filters, stride, r_forward, r_backward, num_branches)
                branches.append(b)
        res = self.shake_skip_connection(x, output_filters, stride)

        return res + tf.add_n(branches)


    def shake_branch(self, x, output_filters, stride, random_forward, random_backward, num_branches):
        """
        Build one shake-shake branch.
        :param x: tf.Tensor, input of shake_branch, shape: (N, H, W, C).
        :param output_filters: int, the number of output filters in shake_branch.
        :param stride: int, the stride of the sliding window to be applied shake_block's branch. 
        :param random_forward: tf.float32, random scalar weight, in paper (alpha or 1 - alpha) for forward propagation.
        :param random_backward: tf.float32, random scalar weight, in paper (alpha or 1 - alpha) for backward propagation.
        :param num_branches: int, the number of branches.
        :return tf.Tensor.
        """
        # relu1 - conv1 - batch_norm1 with stride = stride
        with tf.variable_scope('branch_conv_bn1'):
           x = tf.nn.relu(x) 
           x = conv_layer_no_bias(x, 3, stride, output_filters)
           x = batch_norm(x, is_training=self.is_train) 

        # relu2 - conv2 - batch_norm2 with stride = 1
        with tf.variable_scope('branch_conv_bn2'):
           x = tf.nn.relu(x)
           x = conv_layer_no_bias(x, 3, 1, output_filters) # stirde = 1
           x = batch_norm(x, is_training=self.is_train)

        x = tf.cond(self.is_train, lambda: x * random_backward + tf.stop_gradient(x * random_forward - x * random_backward) , lambda: x / num_branches)

        return x

    def shake_skip_connection(self, x, output_filters, stride):
        """
        Build one shake-shake skip connection.
        :param x: tf.Tensor, input of shake_branch, shape: (N, H, W, C).
        :param output_filters: int, the number of output filters in shake_branch.
        :param stride: int, the stride of the sliding window to be applied shake_block's branch. 
        :return tf.Tensor.
        """
        input_filters = int(x.get_shape()[-1])
        
        if input_filters == output_filters:
           return x

        x = tf.nn.relu(x)

        # Skip connection path 1.
        # avg_pool1 - conv1 
        with tf.variable_scope('skip1'):
           path1 = tf.nn.avg_pool(x, [1, 1, 1, 1], [1, stride, stride, 1], "VALID")
           path1 = conv_layer_no_bias(path1, 1, 1, int(output_filters / 2))

        # Skip connection path 2.
        # pixel shift2 - avg_pool2 - conv2 
        with tf.variable_scope('skip2'):
           path2 = tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]])[:, 1:, 1:, :]
           path2 = tf.nn.avg_pool(path2, [1, 1, 1, 1], [1, stride, stride, 1], "VALID")
           path2 = conv_layer_no_bias(path2, 1, 1, int(output_filters / 2))
 
        # Concatenation path 1 and path 2 and apply batch_norm
        with tf.variable_scope('concat'):
           concat_path = tf.concat(values=[path1, path2], axis= -1)
           bn_path = batch_norm(concat_path, is_training=self.is_train)
        
        return bn_path


