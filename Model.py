import tensorflow as tf

class Model:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
        x = tf.keras.layers.Embedding(input_dim=10000, output_dim=128)(inputs)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def reinitialize_weights(self, scope_name):
        for layer in self.model.layers:
            if layer.name.startswith(scope_name):
                layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
                if layer.bias is not None:
                    layer.bias.assign(layer.bias_initializer(layer.bias.shape))

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        lr_method = lr_method.lower()
        optimizer_dict = {
            'adam': tf.keras.optimizers.Adam,
            'adagrad': tf.keras.optimizers.Adagrad,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop
        }

        optimizer = optimizer_dict[lr_method](learning_rate=lr)

        if clip > 0:
            gradients = optimizer.compute_gradients(loss)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip)
            self.train_op = optimizer.apply_gradients(clipped_gradients)
        else:
            self.train_op = optimizer.minimize(loss)

    def train(self, train_data, dev_data):
        best_score = 0
        epochs_no_improvement = 0

        for epoch in range(self.config.nepochs):
            print(f"Epoch {epoch + 1} out of {self.config.nepochs}")
            score = self.run_epoch(train_data, dev_data, epoch)
            self.config.lr *= self.config.lr_decay

            if score >= best_score:
                epochs_no_improvement = 0
                best_score = score
                print("- New best score!")
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= self.config.nepoch_no_imprv:
                    print(f"- Early stopping: {epochs_no_improvement} epochs without improvement")
                    break

    def evaluate(self, test_data):
        print("Testing model over test set")
        metrics = self.run_evaluate(test_data)
        msg = " - ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        print(msg)

    def run_epoch(self, train_data, dev_data, epoch):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits = self.model(x_batch_train, training=True)
                loss_value = self.model.compiled_loss(y_batch_train, logits)
                total_loss += loss_value
                total_accuracy += self.model.compiled_metrics[0].result()

            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            num_batches += 1

            if step % 100 == 0:
                print(f"Step {step}: loss = {loss_value.numpy()}, accuracy = {self.model.compiled_metrics[0].result().numpy()}")

        average_loss = total_loss / num_batches
        average_accuracy = total_accuracy / num_batches
        print(f"Epoch {epoch + 1} finished with loss = {average_loss.numpy()}, accuracy = {average_accuracy.numpy()}")

        val_score = self.run_evaluate(dev_data)
        return val_score['accuracy']

    def run_evaluate(self, test_data):
        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for x_batch_test, y_batch_test in test_data:
            logits = self.model(x_batch_test, training=False)
            loss_value = self.model.compiled_loss(y_batch_test, logits)
            total_loss += loss_value
            total_accuracy += self.model.compiled_metrics[0].result()
            num_batches += 1

        average_loss = total_loss / num_batches
        average_accuracy = total_accuracy / num_batches

        metrics = {
            'loss': average_loss.numpy(),
            'accuracy': average_accuracy.numpy()
        }

        return metrics
