from .data_utils import get_trimmed_glove_vectors, load_vocab, get_processing_word

class MyConfig:
    def __init__(self):
        self.dir_vocab = "/data/LSTM-CRF/vocab/no-dev/"
        self.filename_words = self.dir_vocab + "words.txt"
        self.filename_tags = self.dir_vocab + "tags.txt"
        self.filename_chars = self.dir_vocab + "chars.txt"

        self.filename_glove = "/data/vectors/wiki.te.vec"
        self.filename_trimmed = "/data/vectors/wiki.te.vec.trimmed.npz"
        self.use_pretrained = True

        self.filename_dev = "/data/train.txt"
        self.filename_test = "/data/test.txt"
        self.filename_train = "/data/train.txt"
        self.filename_predictions = "/data/LSTM-CRF/predictions_no-dev.txt"

        self.dim_word = 300
        self.dim_char = 100

        self.train_embeddings = False
        self.nepochs = 25
        self.dropout = 0.4
        self.batch_size = 20
        self.lr_method = "adam"
        self.lr = 0.001
        self.lr_decay = 0.9
        self.clip = -1
        self.nepoch_no_imprv = 7

        self.hidden_size_char = 100
        self.hidden_size_lstm = 300
        self.use_crf = True
        self.use_chars = True

        self.load()

    def load(self):
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        self.processing_word = get_processing_word(
            self.vocab_words, self.vocab_chars, lowercase=False, chars=self.use_chars)
        self.processing_tag = get_processing_word(
            self.vocab_tags, lowercase=False, allow_unk=False)

        self.embeddings = get_trimmed_glove_vectors(
            self.filename_trimmed) if self.use_pretrained else None
