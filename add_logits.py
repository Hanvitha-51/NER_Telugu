import tensorflow as tf
import numpy as np
import pandas as pd

def add_logits(self):
    with tf.variable_scope("bi-lstm"):
        cell_fw = tf.keras.layers.LSTMCell(self.config.hidden_size_lstm)
        cell_bw = tf.keras.layers.LSTMCell(self.config.hidden_size_lstm)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn( cell_fw, cell_bw, self.word_embeddings, sequence_length=self.sequence_lengths, dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.nn.dropout(output, self.dropout)

    with tf.variable_scope("proj", reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", dtype=tf.float32, shape=[2*self.config.hidden_size_lstm, self.config.ntags])

        b = tf.get_variable("b", shape=[self.config.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())

        nsteps = tf.shape(output)[1]
        output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
        pred = tf.matmul(output, W) + b
        self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])