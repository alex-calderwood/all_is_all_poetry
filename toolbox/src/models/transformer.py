# Author: Alex

import torch
from torch import nn
import math


def forward(func):
    """Decorator to print name of function call
    """
    def wrapper(*args, **kwargs):
        print(f'{func.__qualname__}')
        return func(*args, **kwargs)
    return wrapper


class Transformer(nn.Module):
    def __init__(self, dictionary, d_model=512, heads=8, dk_dv=64, p_drop=0.1, dff=2048):
        super(Transformer, self).__init__()

        # For now we are going to make these the same, but this is not necessary
        dv = dk = dk_dv

        assert heads * dv == d_model

        self.dictionary = dictionary
        self.d_model = d_model
        self.multi_head_attn = MultiHeadAttention(heads, d_model, dk, dv, p_drop, dff)

    @forward
    def forward(self, x):
        pos_en = PositionalEncoding(x, self.d_model, self.dictionary)
        out = self.multi_head_attn(pos_en.embedding)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dk, dv, p_drop, dff):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.heads = [AttentionHead(d_model, dk, dv, p_drop) for _ in range(n_heads)]

        # Learned attention output matrix
        self.WO = torch.randn(self.n_heads * dk, d_model)

        # Layer norm 1
        self.layer_norm1 = nn.LayerNorm(d_model)

        # Set activation function
        self.activation = nn.functional.relu

        # Feed forward 1
        ff_shape = (d_model, dff)
        self.W1 = torch.randn(ff_shape[0], ff_shape[1])
        self.b1 = torch.randn(ff_shape[1])

        # Feed forward 2
        head_out_shape = (ff_shape[1], d_model)
        self.W2 = torch.randn(head_out_shape[0], head_out_shape[1])
        self.b2 = torch.randn(d_model)

        # Layer norm 2
        self.layer_norm2 = nn.LayerNorm(d_model)

    @forward
    def forward(self, x):

        Z = []
        for head in self.heads:
            # Pass the encoded vector to each head
            head_out = head(x)
            # Save
            Z.append(head_out)

        # Concatenate the outputs of all heads
        Z1 = torch.cat(Z, dim=1)

        # Multiply by learned embedding matrix WO
        Z1 = Z1.matmul(self.WO)

        # Connect residuals and apply layer norm
        Z1 = self.layer_norm1(Z1 + x)

        # Feed forward x 2  # TODO does it make sense that there is no activation applied to the last one?
        Z2 = self.activation(Z1.matmul(self.W1) + self.b1)
        Z2 = Z2.matmul(self.W2) + self.b2

        # Connect second residual and apply layer norm
        Z2 = self.layer_norm2(Z2 + Z1)

        return Z2


class AttentionHead(nn.Module):
    def __init__(self, d_model, dk, dv, p_drop):
        super(AttentionHead, self).__init__()
        self.d_model = d_model
        self.dk = dk
        self.dv = dv
        self.p_drop = p_drop

        # Initialize Weight Matrices
        self.WQ = torch.randn(d_model, dk)
        self.WK = torch.randn(d_model, dk)
        self.WV = torch.randn(d_model, dv)

    @forward
    def forward(self, E):
        """
        :param E: An embedding matrix of dimension (seq_len, d_model)
        :return:
        """

        Q = E.matmul(self.WQ)
        K = E.matmul(self.WK)
        V = E.matmul(self.WV)

        att = Q.matmul(K.T) / math.sqrt(self.dk)
        Z = torch.softmax(att, dim=1).matmul(V)
        return Z


class PositionalEncoding():
    def __init__(self, seq, d_model, dictionary):
        self.seq = seq
        self.d_model = d_model

        # Create the original sequence Matrix (seq_len x vocab)
        self.seq = torch.tensor([dictionary.onehot(word) for word in seq])

        self.embedding = self.embed(self.seq)

        # Add the positional encoding information
        self.embedding = self.add_position(self.embedding)

    def embed(self, seq):
        # Turn the sequence into an embedding matrix (seq_len x d_model)
        embedding = torch.empty(seq.shape[0], self.d_model)
        for i in range(seq.shape[0]):
            word = seq[i, :]

            # TODO actually embed, right now it just cuts the vector to the right size
            embedding[i] = word[:512]

        return embedding

    def add_position(self, E):
        """
        Section 3.5 - Attention is All You Need
        :param E: word embedding matrix
        """

        encoding_info = torch.zeros((E.shape[0], self.d_model))

        for pos, word in enumerate(E):
            for i in range(len(word)):
                if i % 2 == 0:
                    encoding_info[pos, i] = math.sin(pos / (10000 ** (i / self.d_model)))
                else:
                    encoding_info[pos, i] = math.cos(pos / (10000 ** ((i - 1) / self.d_model)))

            # TODO double check this math
            # print(encoding_info[pos, :])

        # Add positional encoding to original encoding
        return E + encoding_info


class Dictionary:
    import pickle
    from collections import Counter

    UNK = '<unk>'

    def __init__(self, file=None):
        """
        :param file: either a corpus or a pickle
        """

        # Read in a corpus file or a saved dictionary pickle
        ext = file.split('.')[-1]
        if ext == 'pickle':
            self.vocab = self._load(file)
        elif ext == 'txt':
            self.vocab = self._read_corpus(file)

        self.size = len(self.vocab.keys())

        # Create the lookup tables
        self.word_to_id, self.id_to_word = self._make_id_lookup(self.vocab)

    def save(self, file):
        self.pickle.dump(self.vocab, open(file, 'wb'))

    def _load(self, file):
        return self.pickle.load(open(file, 'rb'))

    def _read_corpus(self, file):
        with open(file, 'r') as f:
            corpus = self.Counter()

            # Add tokens from the corpus file
            for line in f.readlines():
                for word in self._tokenize(line):
                    corpus[word] += 1

            # Add additional tokens
            corpus[self.UNK] = 0

            return corpus

    def _make_id_lookup(self, vocab):
        word_to_id = {}
        id_to_word = {}

        for i, (word, count) in enumerate(vocab.items()):
            word_to_id[word] = i
            id_to_word[i] = word

        return word_to_id, id_to_word

    @staticmethod
    def _tokenize(text):
        return text.split(' ')

    def id(self, word):
        return self.word_to_id[word] if self.word_to_id.get(word) else self.word_to_id[self.UNK]

    def word(self, id=None, onehot=None):
        assert((onehot is not None or id is not None)
               and not (onehot is None and id is None))

        if onehot is not None:
            for i, x in enumerate(onehot):
                if x:
                    id = i

        return self.id_to_word[id]

    def onehot(self, word=None, id=None):
        """
        Encode the given word or id in onehot form.
        Enter only a word or an id, not both
        """""
        assert((word is not None or id is not None)
               and not (word is None and id is None))

        if word:
            id = self.id(word)

        vec = [0] * self.size
        vec[id] = 1
        return vec

    def __str__(self):
        return '<Dictionary> {}'.format([word for word in self.vocab][:1000])


if __name__ == '__main__':
    dictionary = Dictionary('/Users/alexcalderwood/Documents/all_is_all_poetry/toolbox/corpus/litbank-master/original/4300_ulysses.txt')
    # dictionary.save('corpus.pickle')
    # print(dictionary.onehot(word='<unk>'))


    # dictionary = Dictionary('corpus.pickle')
    # print(dictionary)
    net = Transformer(dictionary)
    sentence = 'The cat jumped over the hat and died.'.split(' ')
    net.forward(sentence)