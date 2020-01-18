import argparse
import os
import re
from struct import pack

from flair.models import SequenceTagger

parser = argparse.ArgumentParser(description='Training NER baseline')
parser.add_argument('-task', type=str, default='wnut17')
parser.add_argument('-file', help='file', type=str, default='wnut17/best-model.pt')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from flair import datasets

corpus = None
if args.task == 'ner03':
    corpus = datasets.CONLL_03()
elif args.task == 'wnut17':
    corpus = datasets.WNUT_17()
elif args.task == 'chunk':
    corpus = datasets.CONLL_2000()
model = SequenceTagger.load(args.file)

# save vocabulary for the development set and test set
print('saving text vocab...')
tokens = []
for s in corpus.dev.sentences + corpus.test.sentences:
    for t in s:
        tokens.append(t.text)
tokens = set(tokens)
with open(args.task + '.vocab', 'w') as f:
    f.write('{}\n'.format(len(tokens)))
    for i, t in enumerate(tokens):
        f.write('{}\t{}\n'.format(t, i + 1))

# save vocabulary and embeddings
# multiple words can share the same embeddings
print('saving embeddings and their vocabs...')
embeddings = model.embeddings.embeddings
for emb in embeddings:
    key = emb.precomputed_word_embeddings.vocab
    word_2_id = {}
    unique_words = []
    id = 1
    for word in tokens:
        if word in key:
            word_2_id[word] = id
            unique_words.append(word)
            id += 1
        elif word.lower() in key:
            word_2_id[word] = id
            word_2_id[word.lower()] = id
            unique_words.append(word.lower())
            id += 1
        elif re.sub(r"\d", "#", word.lower()) in key:
            word_2_id[word] = id
            word_2_id[re.sub(r"\d", "#", word.lower())] = id
            unique_words.append(re.sub(r"\d", "#", word.lower()))
            id += 1
        elif re.sub(r"\d", "0", word.lower()) in key:
            word_2_id[word] = id
            word_2_id[re.sub(r"\d", "0", word.lower())] = id
            unique_words.append(re.sub(r"\d", "0", word.lower()))
            id += 1
    with open(args.task + '{}.emb.vocab'.format(emb.embeddings), 'w') as f:
        f.write('{}\n'.format(len(word_2_id)))
        f.write('{}\t{}\n'.format('<PAD>', 0))
        for k, v in word_2_id.items():
            f.write('{}\t{}\n'.format(k, v))
    with open(args.task + '{}.emb'.format(emb.embeddings), 'wb') as f:
        f.write(pack('Q', id))
        emb_size = emb.precomputed_word_embeddings.vector_size
        f.write(pack('Q', emb_size))

        f.write(pack('f' * emb_size, *([0.0] * emb_size)))
        for w in unique_words:
            f.write(pack('f' * emb_size, *(emb.precomputed_word_embeddings[w].tolist())))