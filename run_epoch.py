import tensorflow as tf
import numpy as np
import pandas as pd

def run_epoch(self, train, dev, epoch):
    batch_size = self.config.batch_size
    nbatches = (len(train) + batch_size - 1) // batch_size
    prog = Progbar(target=nbatches)

    # Iterate over the dataset
    for i, (words, labels) in enumerate(minibatches(train, batch_size)):
        fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

        # Update model parameters
        _, train_loss, _ = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

        # Update progress bar
        prog.update(i + 1, [("train loss", train_loss)])

        # Tensorboard logging
        #if i % 10 == 0:
        #    self.file_writer.add_summary(summary, epoch * nbatches + i)

    # Evaluate on dev set
    metrics = self.run_evaluate(dev)

    # Return F1 score
    return metrics["f1"]