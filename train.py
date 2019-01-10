from model import *
from evaluation import *


tf.reset_default_graph()

model = BiLSTMModel(vocabulary_size=len(idx2token), n_tags=len(idx2tag),
                    embedding_dim=200, n_hidden_rnn=200, PAD_index=token2idx['<PAD>'])
batch_size = 32
n_epochs = 5
learning_rate = 0.005
learning_rate_decay = np.sqrt(2.0)
dropout_keep_probability = 0.9

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training...\n')
for epoch in range(n_epochs):
    print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
    print('Train data evaluation:')
    eval_conll(model, sess, train_tokens, train_tags, short_report=True)
    print('Validation data evaluation:')
    eval_conll(model, sess, validation_tokens, validation_tags, short_report=True)

    for x_batch, y_batch, lengths in batches_generator(batch_size, train_tokens, train_tags):
        model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)

    learning_rate = learning_rate / learning_rate_decay

print('...training finished.')