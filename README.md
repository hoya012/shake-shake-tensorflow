# shake-shake-tensorflow
Simple Code Implementation of ["Shake-Shake Regularization"](https://arxiv.org/pdf/1705.07485.pdf) using TensorFlow.
![](https://github.com/hoya012/shake-shake-tensorflow/blob/master/assets/shake-shake.PNG)


*Last update : 2018/12/31*

## Contributor
* hoya012

## Paper Review(Korean only)
I wrote [some posting](http://research.sualab.com/machine-learning/computer-vision/2018/06/28/shake-shake-regularization-review.html) about this paper review. 

## Requirements
Python 3.5
```
numpy >= 1.13.3
matplotlib >= 2.0.2
scikit-learn >= 0.19.1
scikit-image >= 0.13.0
tensorflow-gpu == 1.4.1
```

## Implementation
There are two implementation of codes. First, learning rate schedueling method `SGDR`. Second, `ShakeNet` architecture.

Implementation of SGDR is simple. The ShakeNet has a hierarchical struecture. So i will explain 4 part of ShakeNet.
Important part of code is `Shake Branch` and i use `stop_gradient` trick.

- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/pdf/1608.03983.pdf) method
```python
def _update_learning_rate_cosine(self, global_step, num_iterations):
        """
        update current learning rate, using Cosine function without restart(Loshchilov & Hutter, 2016).
        """
        global_step = min(global_step, num_iterations)
        decay_step = num_iterations
        alpha = 0
        cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_step))
        decayed = (1 - alpha) * cosine_decay + alpha
        new_learning_rate = self.init_learning_rate * decayed

        self.curr_learning_rate = new_learning_rate
```

- Shake Stage 
```python
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
```

- Shake Block
```python
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
```

- Shake Branch **(Important)**
```python
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
```

- Shake Skip Connection
```python
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
```
## Usage
For training, testing, i used `CIFAR-10` Dataset and you can simply run my code.

```python
python train.py
python test.py
```

If you tune hyper-parameter, just change value of `hp_d` dictionary.

## Result
This is my result of `CIFAR-10` dataset and is similar with result of original paper.

| - | Original paper | My implementation  |
| - | :-: | :-: | 
| Accuracy | 96.45% | 96.33% |  

This is plot of my learning curve. Blue line means `accuracy of training set` and red line means `accuracy of validation set`. Almost, we need 1800 or more epochs for saturation.

![](https://github.com/hoya012/shake-shake-tensorflow/blob/master/assets/plot.PNG)


## Reference
- [SUALAB classification code implementation](https://github.com/sualab/asirra-dogs-cats-classification)

