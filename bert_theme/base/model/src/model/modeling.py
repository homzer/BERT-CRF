import os

import numpy as np
import tensorflow as tf

from src.utils.CheckpointUtil import filter_compatible_params

__all__ = ['Model']


class Model(object):

    def __init__(self, features, labels, model_graph):
        """
        Building Model
        :param features: `tf.placeholder`, e.g. tf.placeholder("int32", [None, seq_length])
        :param labels: `tf.placeholder`, e.g. self.labels = tf.placeholder("int32", [None, ])
        :param model_graph: a function defining the graph of model,
        which should take `features` and `labels` as input,
        and take `predictions`, `loss`, `variables` as return, where `variables` can be optional,
        which can be set to `None`.
         updated_var_list: list of variables being updated during backward propagation.
        Default to `None`, represents to update all trainable variables.
        """
        self.features = features
        self.labels = labels
        self.model_graph = model_graph
        self.sess = tf.Session()
        self.__build()
        self.__initialize()

    def __build(self):
        """ Building model structure """
        print("Building model graph......")
        self.predictions, self.loss, self.variables = self.model_graph(
            self.features, self.labels)
        self.global_step = tf.Variable(
            initial_value=0, trainable=False, name='global_step', dtype=tf.int32)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss, self.global_step)
        print("Building model graph complete!")

    def __initialize(self):
        """ Initialize model graph """
        self.sess.run(tf.global_variables_initializer())

    def train(self, features, labels):
        """ Training model """
        self.sess.run(
            self.optimizer,
            feed_dict={
                self.features: features,
                self.labels: labels})
        global_step = self.sess.run(
            self.global_step,
            feed_dict={
                self.features: features,
                self.labels: labels})
        return global_step

    def evaluate(self, features, labels):
        """ Evaluate model """
        loss = self.sess.run(self.loss, feed_dict={self.features: features, self.labels: labels})
        predicts = self.sess.run(self.predictions, feed_dict={self.features: features, self.labels: labels})
        labels = np.array(labels)
        predicts = np.array(predicts)
        labels = np.reshape(labels, [-1, ])
        predicts = np.reshape(predicts, [-1, ])
        accuracy = np.mean([int(label) == int(predict) for label, predict in zip(labels, predicts)])
        return loss, accuracy

    def predict(self, features, labels):
        """ Prediction """
        predicts = self.sess.run(self.predictions, feed_dict={self.features: features, self.labels: labels})
        return predicts

    def validate(self, features, labels):
        """ Return the real value of some variables. """
        variables = None if self.variables is None else self.sess.run(
            self.variables, feed_dict={self.features: features, self.labels: labels})
        return variables

    def save(self, step=0, output_dir="result"):
        """ Save model parameters """
        tvars = tf.trainable_variables()
        saver = tf.train.Saver(tvars)
        save_path = os.path.join(output_dir, "model.ckpt-" + str(step))
        print("Saving model to %s......" % save_path)
        saver.save(self.sess, save_path)
        print("Saving model complete!")

    def restore(self, checkpoint_file="config/model.ckpt"):
        print("Restore model from %s......" % checkpoint_file)
        # Filtering compatible parameters
        rvars = filter_compatible_params(checkpoint_file)
        print("\n".join([str(var.name) for var in rvars]))
        saver = tf.train.Saver(rvars)
        saver.restore(self.sess, checkpoint_file)
        print("Restoring model complete!")
