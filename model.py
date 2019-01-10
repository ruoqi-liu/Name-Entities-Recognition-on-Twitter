import tensorflow as tf
import numpy as np

class BiLSTMModel:
    def __declare_placeholders(self):
        self.input_batch = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input_batch')
        self.ground_truth_tags = tf.placeholder(shape=[None, None], dtype=tf.int32, name='ground_truths')

        self.lengths = tf.placeholder(shape=[None], dtype=tf.int32, name='lengths')

        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
        self.learning_rate_ph = tf.placeholder(shape=[], dtype=tf.float32)


    def __build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
        initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
        embedding_matrix_varible = tf.Variable(initial_value=initial_embedding_matrix, dtype=tf.float32)

        forward_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_rnn),
            input_keep_prob=self.dropout_ph,
            output_keep_prob=self.dropout_ph,
            state_keep_prob=self.dropout_ph
        )
        backward_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_rnn),
            input_keep_prob=self.dropout_ph,
            output_keep_prob=self.dropout_ph,
            state_keep_prob=self.dropout_ph
        )

        embeddings = tf.nn.embedding_lookup(params=embedding_matrix_varible, ids=self.input_batch)

        (rnn_output_fw, rnn_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=forward_cell, cell_bw=backward_cell, inputs=embeddings,
            sequence_length=self.lengths, dtype=tf.float32)

        rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

        self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)


    def __compute_predictions(self):
        softmax_output = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(softmax_output, axis=-1)


    def __compute_loss(self, n_tags, PAD_index):
        ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=ground_truth_tags_one_hot, logits=self.logits)
        mask = tf.cast(tf.not_equal(self.input_batch, PAD_index), tf.float32)
        self.loss = tf.reduce_mean(mask*loss_tensor)


    def __perform_optimization(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clip_norm = tf.cast(1.0, tf.float32)
        self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for (grad, var) in self.grads_and_vars]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)


    def __init__(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
        self.__declare_placeholders()
        self.__build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
        self.__compute_predictions()
        self.__compute_loss(n_tags, PAD_index)
        self.__perform_optimization()



    def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_prob):
        feed_dict={
            self.input_batch: x_batch,
            self.ground_truth_tags: y_batch,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_prob,
            self.lengths: lengths
        }
        session.run(self.train_op, feed_dict=feed_dict)


    def predict_for_batch(self, session, x_batch, lengths):
        feed_dict = {
            self.input_batch: x_batch,
            self.lengths: lengths
        }
        predictions = session.run(self.predictions, feed_dict=feed_dict)
        return predictions


















