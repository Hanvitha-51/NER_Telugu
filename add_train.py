import tensorflow as tf
import numpy as np
import pandas as pd

def add_train(self, lr_method, lr, loss, clip=-1):
    _lr_m = lr_method.lower() 

    with tf.variable_scope("train_step"):
        if _lr_m == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
        elif _lr_m == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(lr)
        elif _lr_m == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif _lr_m == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(lr)

        if clip > 0:  # gradient clipping if clip is positive
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm = tf.clip_by_global_norm(grads, clip)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            self.train_op = optimizer.minimize(loss)
