import numpy as np
import os

RESOURCEPATH = {
    'DATA_FOLDER': 'data',
    'TRAIN': 'train.txt',
    'VALIDATION': 'validation.txt',
    'TEST': 'test.txt'
}



def read_data(file_path):
    tokens, tags = [], []
    tweet_tokens, tweet_tags = [], []

    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens, tweet_tags = [], []
        else:
            token, tag = line.split()
            if token.startswith('http://') or token.startswith('https://'):
                token = '<URL>'
            if token.startswith('@'):
                token = '<USR>'
            tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags


def build_dict(tokens_or_tags, special_tokens):

    unique_set = set()

    for tweet in tokens_or_tags:
        for token in tweet:
            if token not in special_tokens:
                unique_set.add(token)

    idx2tok = [token for token in special_tokens] + list(unique_set)
    tok2idx = {token: idx for idx, token in enumerate(idx2tok)}

    return tok2idx, idx2tok


special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']




data_folder = RESOURCEPATH['DATA_FOLDER']
train_data_path = os.path.join(data_folder, RESOURCEPATH['TRAIN'])
validation_data_path = os.path.join(data_folder, RESOURCEPATH['VALIDATION'])
test_data_path = os.path.join(data_folder, RESOURCEPATH['TEST'])


train_tokens, train_tags = read_data(train_data_path)
validation_tokens, validation_tags = read_data(validation_data_path)
test_tokens, test_tags = read_data(test_data_path)


token2idx, idx2token = build_dict(train_tokens+validation_tokens, special_tokens)
tag2idx, idx2tag = build_dict(train_tags+validation_tags, special_tags)



def words2idxs(tokens_list):
    return [token2idx[word] for word in tokens_list]

def tags2idxs(tags_list):
    return [tag2idx[tag] for tag in tags_list]

def idxs2words(idxs):
    return [idx2token[idx] for idx in idxs]

def idxs2tags(idxs):
    return [idx2tag[idx] for idx in idxs]


def batches_generator(batch_size, tokens, tags, shuffle=True, allow_smaller_last_batch=True):
    n_samples = len(tokens)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.array(n_samples)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k+1)*batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list, y_list = [], []
        max_len_token = 0
        for idx in order[batch_start:batch_end]:
            x_list.append(words2idxs(tokens[idx]))
            y_list.append(tags2idxs(tags[idx]))
            max_len_token = max(max_len_token, len(tags[idx]))


        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * token2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tag2idx['O']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            lengths[n] = utt_len
            y[n, :utt_len] = y_list[n]
        yield x, y, lengths


















