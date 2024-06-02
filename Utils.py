import numpy as np

UNKNOWN_TOKEN = "$UNK$"
NUMERIC = "MISC"
DEFAULT_TAG = "O"

class Dataset:
    def __init__(self, filepath, process_word=None, process_tag=None, max_iterations=None):
        self.filepath = filepath
        self.process_word = process_word
        self.process_tag = process_tag
        self.max_iterations = max_iterations

    def __iter__(self):
        iteration_count = 0
        with open(self.filepath) as file:
            words, tags = [], []
            for line in file:
                line = line.strip()
                if not line or line.startswith("-DOCSTART-"):
                    if words:
                        iteration_count += 1
                        if self.max_iterations and iteration_count > self.max_iterations:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    word, tag = line.split(' ')[0], line.split(' ')[-1]
                    if self.process_word:
                        word = self.process_word(word)
                    if self.process_tag:
                        tag = self.process_tag(tag)
                    words.append(word)
                    tags.append(tag)

    def __len__(self):
        return sum(1 for _ in self)

def load_vocab(filepath):
    with open(filepath) as file:
        return {word.strip(): idx for idx, word in enumerate(file)}

def save_trimmed_glove_vectors(vocab, glove_filepath, trimmed_filepath, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filepath) as file:
        for line in file:
            word, *embedding = line.strip().split(' ')
            if word in vocab:
                embeddings[vocab[word]] = np.asarray(embedding, dtype=np.float32)
    np.savez_compressed(trimmed_filepath, embeddings=embeddings)

def load_trimmed_glove_vectors(filepath):
    with np.load(filepath) as data:
        return data["embeddings"]

def process_word_function(vocab_words=None, vocab_chars=None, lowercase=False, chars=False, allow_unknown=True):
    def process_word(word):
        char_ids = [vocab_chars.get(char, vocab_chars[UNKNOWN_TOKEN]) for char in word] if vocab_chars and chars else None
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUMERIC
        word = vocab_words.get(word, vocab_words[UNKNOWN_TOKEN] if allow_unknown else word)
        return (char_ids, word) if char_ids else word
    return process_word

def pad_sequences(sequences, pad_token, levels=1):
    if levels == 1:
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]
        sequence_lengths = [len(seq) for seq in sequences]
    else:
        max_word_len = max(max(len(word) for word in seq) for seq in sequences)
        padded_sequences, sequence_lengths = zip(*(pad_sequences(seq, pad_token, 1) for seq in sequences))
        max_seq_len = max(len(seq) for seq in sequences)
        padded_sequences, _ = pad_sequences(padded_sequences, [pad_token] * max_word_len, 1)
        sequence_lengths, _ = pad_sequences(sequence_lengths, 0, 1)
    return padded_sequences, sequence_lengths

def generate_minibatches(data, batch_size):
    x_batch, y_batch = [], []
    for x, y in data:
        if len(x_batch) == batch_size:
            yield x_batch, y_batch
            x_batch, y_batch = []
        x_batch.append(list(zip(*x)) if isinstance(x[0], tuple) else x)
        y_batch.append(y)
    if x_batch:
        yield x_batch, y_batch

def extract_chunks(sequence, tag_map):
    default_tag = tag_map[DEFAULT_TAG]
    idx_to_tag = {idx: tag for tag, idx in tag_map.items()}
    chunks, chunk_type, chunk_start = [], None, None
    for i, token in enumerate(sequence):
        if token == default_tag and chunk_type is not None:
            chunks.append((chunk_type, chunk_start, i))
            chunk_type, chunk_start = None, None
        elif token != default_tag:
            tok_chunk_class, tok_chunk_type = idx_to_tag[token].split('-')
            if chunk_type is None or tok_chunk_type != chunk_type or tok_chunk_class == "B":
                if chunk_type is not None:
                    chunks.append((chunk_type, chunk_start, i))
                chunk_type, chunk_start = tok_chunk_type, i
    if chunk_type is not None:
        chunks.append((chunk_type, chunk_start, len(sequence)))
    return chunks
