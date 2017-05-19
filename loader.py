import codecs
import itertools
import os
import re
import numpy as np
import pickle
import io


def initializeEmbedding(*shape):
    drange = np.sqrt(6. / (np.sum(shape)))
    value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return np.array(value, dtype=np.float64)


def prepare_data(bio_file):
    # read bio file
    data = []
    for line in codecs.open(bio_file, 'r', 'utf-8').read().split('\n\n'):
        line = line.strip()
        if not line:
            continue
        sent = dict()
        sent['word'] = []
        sent['label'] = []
        tokens = line.split('\n')
        for t in tokens:
            t_word = t.split(' ')[0]
            t_label = t.split(' ')[-1]

            sent['word'].append(t_word)
            sent['label'].append(t_label)

        data.append(sent)
    return data


def generate_mapping(data, test_data, pre_emb):
    if pre_emb:
        pretrained = load_pretrained(pre_emb)
    else:
        pretrained = {}
    pretrained = set(pretrained.keys())
    emb2augment = []
    for sent in test_data:
        for w in sent['word']:
            if any(x in pretrained for x in [
                w,
                w.lower(),
                re.sub('\d', '0', w.lower())
            ]):
                emb2augment.append(w)

    # generate mapping table
    words = list(itertools.chain(*[sent['word'] for sent in data])) + emb2augment
    words = ['<UNK>'] + list(set(words))
    chars = list(set(''.join(words)))
    chars = ['<UNK>'] + chars
    labels = set(itertools.chain(*[sent['label'] for sent in data]))

    id2word = {i: w for i, w in enumerate(words)}
    word2id = {w: i for i, w in id2word.items()}
    id2char = {i: c for i, c in enumerate(chars)}
    char2id = {c: i for i, c in id2char.items()}
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in id2label.items()}

    return id2word, word2id, id2char, char2id, id2label, label2id


def load_pretrained(pre_emb):
    # if os.path.basename(pre_emb).endswith('.xz'):
    #     f = lzma.open(pre_emb)
    # else:
    #     f = codecs.open(pre_emb, 'r', 'utf-8')
    f = codecs.open(pre_emb, 'r', 'utf-8')
    pretrained = {}
    for i, line in enumerate(f):
        if type(line) == bytes:
            try:
                line = str(line, 'utf-8')
            except UnicodeDecodeError:
                continue
        line = line.rstrip().split()
        pretrained[line[0]] = np.array(
            [float(x) for x in line[1:]]
        ).astype(np.float32)

    return pretrained


def prepare_input(data, word2id, char2id, label2id):
    # prepare training data
    for sent in data:
        sent['word_id'] = []
        for w in sent['word']:
            if w not in word2id:
                w = '<UNK>'
            sent['word_id'].append(word2id[w])

        sent['char_id'] = []
        for w in sent['word']:
            char_id = []
            for char in w:
                if char not in char2id:
                    char = '<UNK>'
                char_id .append(char2id[char])
            sent['char_id'].append(char_id)

        sent['label_id'] = []
        for l in sent['label']:
            if l not in label2id:
                raise ValueError('Unknown label found in dev or test set: %s' % l)
            sent['label_id'].append(label2id[l])

    return data


word_dim = 50
char_dim = 25
pre_emb = '/Users/zhangb8/Documents/fb_intern/data/emb/eng_senna.emb'
pre_emb = ""
model_dir = 'model'

# POS tagging data
train_file = "/Users/zhangb8/Documents/cDNN/data/updated_UD_English/en-ud-train.bio"
test_file = "/Users/zhangb8/Documents/cDNN/data/updated_UD_English/en-ud-dev.bio"
train_file = test_file

train_data = prepare_data(train_file)
test_data = prepare_data(test_file)

id2word, word2id, id2char, char2id, id2label, label2id = generate_mapping(train_data, test_data, pre_emb)

print('%d unique words.' % len(id2word))
print('%d labels.' % len(id2label))

train_input = prepare_input(train_data, word2id, char2id, label2id)
test_input = prepare_input(test_data, word2id, char2id, label2id)

vocabulary_size = len(id2word)
word_embedding = initializeEmbedding(vocabulary_size, word_dim)

if pre_emb:
    # load pre-trained embeddings
    new_weights = word_embedding.embedding
    print('Loading pretrained embeddings from %s...' % pre_emb)
    pretrained = load_pretrained(pre_emb)
    # if emb_invalid > 0:
    #     print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    # Lookup table initialization
    for i in range(len(id2word)):
        word = id2word[i]
        if word in pretrained:
            new_weights[i] = pretrained[word]
            c_found += 1
        elif word.lower() in pretrained:
            new_weights[i] = pretrained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pretrained:
            new_weights[i] = pretrained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    word_embedding.embedding = new_weights
    print('Loaded %i pretrained embeddings.' % len(pretrained))
    print('%i / %i (%.4f%%) words have been initialized with '
          'pretrained embeddings.' % (
              c_found + c_lower + c_zeros, len(id2word),
              100. * (c_found + c_lower + c_zeros) / len(id2word)
          ))
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
              c_found, c_lower, c_zeros
          ))

char_embedding = initializeEmbedding(len(char2id), char_dim)

#
# export train_data, test_data. mapping and initial embeddings
#
mapping = {
    'id2word': id2word,
    'word2id': word2id,
    'id2char': id2char,
    'char2id': char2id,
    'id2label': id2label,
    'label2id': label2id
}
mapping_file = io.open(os.path.join(model_dir, 'mapping.pkl'), 'wb')
pickle.dump(mapping, mapping_file)

word_embedding_file = open(os.path.join(model_dir, 'word_embedding.csv'), 'w')
for row in word_embedding:
    word_embedding_file.write(','.join([str(item) for item in row]) + '\n')
char_embedding_file = open(os.path.join(model_dir, 'char_embedding.csv'), 'w')
for row in char_embedding:
    char_embedding_file.write(','.join([str(item) for item in row]) + '\n')

for k, v in train_input.items():





