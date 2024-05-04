def add_word_embeddings(self):
    with tf.variable_scope("words", reuse=tf.AUTO_REUSE):
        _word_embeddings = tf.get_variable(name="_word_embeddings", dtype=tf.float32, shape=[self.config.nwords, self.config.dim_word])

        word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

    with tf.variable_scope("chars", reuse=tf.AUTO_REUSE):
        # Get char embeddings matrix
        _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32, shape=[self.config.nchars, self.config.dim_char])
        char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")

        # Put the time dimension on axis=1
        s = tf.shape(char_embeddings)
        char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], self.config.dim_char])
        word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

        # Bidirectional LSTM on chars
        lstm_fw = tf.keras.layers.LSTM(self.config.hidden_size_char, return_sequences=True)
        lstm_bw = tf.keras.layers.LSTM(self.config.hidden_size_char, return_sequences=True, go_backwards=True)

        # Wrap LSTM layers with Bidirectional
        bidirectional_fw = tf.keras.layers.Bidirectional(lstm_fw)
        bidirectional_bw = tf.keras.layers.Bidirectional(lstm_bw)

        # Run bidirectional LSTM
        output_fw = bidirectional_fw(char_embeddings)
        output_bw = bidirectional_bw(char_embeddings)

        # Concatenate outputs
        _output = tf.keras.layers.Concatenate()([output_fw, output_bw])

        # Reshape and concatenate with word_embeddings
        output = tf.reshape(_output, shape=[s[0], s[1], 2 * self.config.hidden_size_char])
        word_embeddings = tf.concat([word_embeddings, output], axis=-1)

    self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)