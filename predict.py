import tensorflow as tf
import numpy as np
import pandas as pd

def predict(self, words_raw):
    # Process words
    words = [self.config.processing_word(w) for w in words_raw]

    # Check if words are represented as tuples
    if isinstance(words[0], tuple):
        words = zip(*words)

    # Obtain predictions using predict_batch method
    pred_ids, _ = self.predict_batch([words])

    # Convert prediction indices to tags using idx_to_tag mapping
    preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

    return preds