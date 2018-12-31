import os
import time
from abc import abstractmethod
import tensorflow as tf
from learning.utils import plot_learning_curve
import math

class Optimizer(object):
    """Base class for gradient-based optimization algorithms."""

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        Optimizer initializer.
        :param model: ConvNet, the model to be learned.
        :param train_set: DataSet, training set to be used.
        :param evaluator: Evaluator, for computing performance scores during training.
        :param val_set: DataSet, validation set to be used, which can be None if not used.
        :param kwargs: dict, extra arguments containing training hyperparameters.
            - batch_size: int, batch size for each iteration.
            - num_epochs: int, total number of epochs for training.
            - init_learning_rate: float, initial learning rate.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        # Training hyperparameters
        self.batch_size = kwargs.pop('batch_size', 128)
        self.num_epochs = kwargs.pop('num_epochs', 1800)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.2)

        self.learning_rate_placeholder = tf.placeholder(tf.float32)    # Placeholder for current learning rate
        self.optimize = self._optimize_op()

        self._reset()

    def _reset(self):
        """Reset some variables."""
        self.curr_epoch = 1
        self.num_bad_epochs = 0    # number of bad epochs, where the model is updated without improvement.
        self.best_score = self.evaluator.worst_score    # initialize best score with the worst one
        self.curr_learning_rate = self.init_learning_rate    # current learning rate

    @abstractmethod
    def _optimize_op(self, **kwargs):
        """
        tf.train.Optimizer.minimize Op for a gradient update.
        This should be implemented, and should not be called manually.
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate (if needed) on every epoch, by its own schedule.
        This should be implemented, and should not be called manually.
        """
        pass

    def _step(self, sess, **kwargs):
        """
        Make a single gradient update and return its results.
        This should not be called manually.
        :param sess: tf.Session.
        :param kwargs: dict, extra arguments containing training hyperparameters.
            - augment_train: bool, whether to perform augmentation for training.
        :return loss: float, loss value for the single iteration step.
                y_true: np.ndarray, true label from the training set.
                y_pred: np.ndarray, predicted label from the model.
        """
        augment_train = kwargs.pop('augment_train', True)

        # Sample a single batch
        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True,
                                              augment=augment_train, is_train=True)

        # update extra ops for batch_normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Compute the loss and make update
        with tf.control_dependencies(extra_update_ops):
           _, loss, y_pred = sess.run([self.optimize, self.model.loss, self.model.pred], feed_dict={self.model.X: X, self.model.y: y_true, self.model.is_train: True, self.learning_rate_placeholder: self.curr_learning_rate})

        return loss, y_true, y_pred

    def train(self, sess, save_dir='./tmp', details=False, verbose=True, **kwargs):
        """
        Run optimizer to train the model.
        :param sess: tf.Session.
        :param save_dir: str, the directory to save the learned weights of the model.
        :param details: bool, whether to return detailed results.
        :param verbose: bool, whether to print details during training.
        :param kwargs: dict, extra arguments containing training hyperparameters.
        :return train_results: dict, containing detailed results of training.
        """
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())    # initialize all weights

        train_results = dict()    # dictionary to contain training(, evaluation) results and details
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        # Start training loop
        for i in range(num_steps):
            # Perform a gradient update from a single minibatch
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)

            # Perform evaluation in the end of each epoch
            if (i+1) % num_steps_per_epoch == 0:
                # Evaluate model with current minibatch, from training set
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)

                # If validation set is initially given, use it for evaluation
                if self.val_set is not None:
                    # Evaluate model with the validation set
                    eval_y_pred = self.model.predict(sess, self.val_set, verbose=False, **kwargs)
                    eval_score = self.evaluator.score(self.val_set.labels, eval_y_pred)
                    eval_scores.append(eval_score)

                    if verbose:
                        # Print intermediate results
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))
                        # Plot intermediate results
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = eval_score

                # else, just use results from current minibatch for evaluation
                else:
                    if verbose:
                        # Print intermediate results
                        print('[epoch {}]\tloss: {} |Train score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                        # Plot intermediate results
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = step_score

                # Keep track of the current best model,
                # by comparing current score and the best score
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt')) # save current weights
                else:
                    self.num_bad_epochs += 1

                #self._update_learning_rate(**kwargs)
                self._update_learning_rate_cosine(i, num_steps)

                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} score: {}'.format('evaluation' if eval else 'training',
                                             self.best_score))
        print('Done.')

        if details:
            # Store training results in a dictionary
            train_results['step_losses'] = step_losses    # (num_iterations)
            train_results['step_scores'] = step_scores    # (num_epochs)
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores    # (num_epochs)

            return train_results


class MomentumOptimizer(Optimizer):
    """Gradient descent optimizer, with Momentum algorithm."""

    def _optimize_op(self, **kwargs):
        """
        tf.train.MomentumOptimizer.minimize Op for a gradient update.
        :param kwargs: dict, extra arguments for optimizer.
            - momentum: float, the momentum coefficient.
        :return tf.Operation.
        """
        momentum = kwargs.pop('momentum', 0.9)

        
        # update extra ops for batch_normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Compute the loss and make update
        with tf.control_dependencies(extra_update_ops):
           update_vars = tf.trainable_variables()
           return tf.train.MomentumOptimizer(self.learning_rate_placeholder, momentum, use_nesterov=True)\
                .minimize(self.model.loss, var_list=update_vars)

    def _update_learning_rate(self, **kwargs):
        """
        update current learning rate, when evaluation score plateaus.
        :param kwargs: dict, extra arguments for learning rate scheduling.
            - learning_rate_patience: int, number of epochs with no improvement
                                      after which learning rate will be reduced.
            - learning_rate_decay: float, factor by which the learning rate will be updated.
            - eps: float, if the difference between new and old learning rate is smaller than eps,
                   the update is ignored.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # decay learning rate only when the difference is higher than epsilon.
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_bad_epochs = 0

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

