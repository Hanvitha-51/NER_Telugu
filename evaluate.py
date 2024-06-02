import tensorflow as tf
import numpy as np
import pandas as pd

def evaluate(self, test):
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.

    # Prediction and Evaluation Loop
    for words, labels in minibatches(test, self.config.batch_size):
        labels_pred, sequence_lengths = self.predict_batch(words)

        for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]

            lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
            lab_pred_chunks = set(get_chunks(lab_pred, self.config.vocab_tags))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)

    # Accuracy Calculation
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)

    # Return Statement
    return {"acc": 100 * acc, "f1": 100 * f1}