import pickle
import json
import codecs
import os

heads = list()
desc = list()
# perpare data
maxLines = 10000
i = 0
infile = codecs.open('../bytecup2018/bytecup.corpus.train.0.txt', 'r', encoding='utf-8', errors='ignore')
for line in infile.readlines():
    lineParsed = json.loads(line.strip('\n'))
    heads.append(lineParsed['title'].strip())
    desc.append(lineParsed['content'].strip())

    i = i + 1
    if (i >= maxLines):
        break
infile.close()

# # Pickle the data
# tupleToPickle = (heads, descs, None)
#
# if not os.path.exists("data"):
#     os.makedirs("data")
#
# file = open("data/tokens.pkl", "wb")
# pickle.dump(tupleToPickle, file)
# file.close()

FN = 'vocabulary-embedding'
seed=42
vocab_size = 40000
embedding_dim = 100
lower = False # dont lower case the text

# FN0 = 'tokens'  # this is the name of the data file which I assume you already have
# with open('data/%s.pkl' % FN0, 'rb') as fp:
#     heads, desc, keywords = pickle.load(fp)     # keywords are not used in this project

if lower:
    heads = [h.lower() for h in heads]
    desc = [h.lower() for h in desc]

i = 0
# print(heads[i])
# print(desc[i])

print('heads len:', len(heads), len(set(heads)))
print('desc len:', len(desc), len(set(desc)))

from collections import Counter
from itertools import chain

def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = list(map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1])))
    return vocab, vocabcount

vocab, vocabcount = get_vocab(heads+desc)

# print(vocab[:50])
print('vocab len:', len(vocab))

import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot([vocabcount[w] for w in vocab]);
plt.gca().set_xscale("log", nonposx='clip')
plt.gca().set_yscale("log", nonposy='clip')
plt.title('word distribution in headlines and discription')
plt.xlabel('rank')
plt.ylabel('total appearances')
# plt.show()

empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word

def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx + start_idx) for idx, word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos

    idx2word = dict((idx, word) for word, idx in word2idx.items())

    return word2idx, idx2word

word2idx, idx2word = get_idx(vocab, vocabcount)

# fname = 'glove.6B.%dd.txt'%embedding_dim
# import os
# datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
# if not os.access(datadir_base, os.W_OK):
#     datadir_base = os.path.join('/tmp', '.keras')
# datadir = os.path.join(datadir_base, 'datasets')
# glove_name = os.path.join(datadir, fname)
# if not os.path.exists(glove_name):
#     path = 'glove.6B.zip'
#     path = get_file(path, origin="http://nlp.stanford.edu/data/glove.6B.zip")
#     !unzip {datadir}/{path}

glove_name = './glove.6B/glove.6B.%dd.txt' % embedding_dim
file = open(glove_name, 'r', encoding='utf-8')
glove_n_symbols = len(file.readlines())
file.close()

# glove_n_symbols = !wc -l {glove_name}
# glove_n_symbols = int(glove_n_symbols[0].split()[0])
print('glove_n_symbols:', glove_n_symbols)

import numpy as np
glove_index_dict = {}
# matrix 400000 * 100
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
globale_scale = .1
with open(glove_name, 'r') as fp:
    i = 0
    for l in fp:
        l = l.strip().split()
        w = l[0]
        glove_index_dict[w] = i
        glove_embedding_weights[i, :] = list(map(float, l[1:]))
        i += 1
glove_embedding_weights *= globale_scale
print('glove_embedding_weights.std:', glove_embedding_weights.std())

for w, i in glove_index_dict.items():
    w = w.lower()
    if w not in glove_index_dict:
        glove_index_dict[w] = i

# generate random embedding with same scale as glove
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
print('shape', shape)
scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
embedding = np.random.uniform(low=-scale, high=scale, size=shape)
print('random-embedding/glove scale', scale, 'std', embedding.std())

# copy from glove weights of words that appear in our short vocabulary (idx2word)
c = 0
for i in range(vocab_size):
    w = idx2word[i]
    g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    # print(w, g)
    if g is None and w.startswith('#'): # glove has no hastags (I think...)
        w = w[1:]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
    if g is not None:
        embedding[i, :] = glove_embedding_weights[g, :]
        c += 1
print('number of tokens, in small vocab, found in glove and copied to embedding', c, c/float(vocab_size))

glove_thr = 0.5
word2glove = {}
for w in word2idx:
    if w in glove_index_dict:
        g = w
    elif w.lower() in glove_index_dict:
        g = w.lower()
    elif w.startswith('#') and w[1:] in glove_index_dict:
        g = w[1:]
    elif w.startswith('#') and w[1:].lower() in glove_index_dict:
        g = w[1:].lower()
    else:
        continue
    word2glove[w] = g
print('word2glove:', len(word2glove))
normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight, gweight)) for gweight in embedding])[:, None]

nb_unknown_words = 100

glove_match = []
for w,idx in word2idx.items():
    if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
        gidx = glove_index_dict[word2glove[w]]
        gweight = glove_embedding_weights[gidx,:].copy()
        # find row in embedding that has the highest cos score with gweight
        gweight /= np.sqrt(np.dot(gweight,gweight))
        score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
        while True:
            embedding_idx = score.argmax()
            s = score[embedding_idx]
            if s < glove_thr:
                break
            if idx2word[embedding_idx] in word2glove :
                glove_match.append((w, embedding_idx, s))
                break
            score[embedding_idx] = -1
print('nb_unknown_words:', nb_unknown_words)

glove_match.sort(key=lambda x: -x[2])
print('# of glove substitutes found', len(glove_match))

for orig, sub, score in glove_match[-5:]:
    print(score, orig, '=>', idx2word[sub])
glove_idx2idx = dict((word2idx[w], embedding_idx) for w, embedding_idx, _ in glove_match)
print('glove_idx2idx:', len(glove_idx2idx))

Y = [[word2idx[token] for token in headline.split()] for headline in heads]
print('Y len:', len(Y))
# plt.hist(list(map(len, Y)), bins=50)

X = [[word2idx[token] for token in d.split()] for d in desc]
print('X len:', len(X))
# plt.hist(list(map(len, X)), bins=50)

with open('data/%s.pkl' % FN, 'wb') as fp:
    pickle.dump((embedding, idx2word, word2idx, glove_idx2idx), fp, -1)

with open('data/%s.data.pkl' % FN, 'wb') as fp:
    pickle.dump((X, Y), fp, -1)

