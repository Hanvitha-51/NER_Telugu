import tensorflow as tf
import numpy as np
import pandas as pd

def predict_batch(self, words):
    # Construct feed dictionary
    fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
    """
    if self.config.use_crf:
        viterbi_sequences = []

        logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=fd)

        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit, trans_params)
            viterbi_sequences.append(viterbi_seq)
        return viterbi_sequences, sequence_lengths

    else:
    """  
        # Run prediction operation
        labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

        return labels_pred, sequence_lengths