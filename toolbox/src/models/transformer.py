# Author: Alex

import torch
from torch import nn
import math


def forward(func):
    """Decorator to print name of function call
    """
    def wrapper(*args, **kwargs):

        arg_lengths = [args[i].shape if isinstance(args[i], torch.Tensor) else len(args[i]) for i in range(1, len(args))]

        print(f'{func.__qualname__}', arg_lengths)
        return func(*args, **kwargs)
    return wrapper


class Transformer(nn.Module):
    def __init__(self, dictionary, d_model=512, encoding_layers=6, decoding_layers=6, heads=8, dk_dv=64, p_drop=0.1, dff=2048):
        super(Transformer, self).__init__()

        # For now we are going to make these the same, but this is not necessary
        dv = dk = dk_dv

        assert heads * dv == d_model

        self.dictionary = dictionary
        self.d_model = d_model

        self.encoding_layers = [MultiHeadAttentionEncodingLayer(heads, d_model, dk, dv, p_drop, dff) for _ in range(encoding_layers)]
        self.decoding_layers = [MultiHeadAttentionDecodingLayer(heads, d_model, dk, dv, p_drop, dff) for _ in range(decoding_layers)]

    @forward
    def forward(self, x):
        pos_en = PositionalEncoding(x, self.d_model, self.dictionary)

        # Pass the encoding sequentially to each layer of the encoding
        out = pos_en.embedding
        for multi_head_encoding_layer in self.encoding_layers:
            out = multi_head_encoding_layer(out)

        return out


class MultiHeadAttentionEncodingLayer(nn.Module):

    def __init__(self, n_heads, d_model, dk, dv, p_drop, dff):
        super(MultiHeadAttentionEncodingLayer, self).__init__()

        # Multi head self attention x 6
        self.multi_head_attention = MultiheadAttention(n_heads, d_model, dk, dv, p_drop, dff)
        self.norm1 = nn.LayerNorm(d_model)

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
    def forward(self, src):
        attn_out = self.multi_head_attention(src)
        attn_out = self.norm1(attn_out + src)

        # Feed forward x 2  # TODO does it make sense that there is no activation applied to the last one?
        ff_out = self.activation(attn_out.matmul(self.W1) + self.b1)
        ff_out = ff_out.matmul(self.W2) + self.b2

        # Connect second residual and apply layer norm
        ff_out = self.layer_norm2(ff_out + attn_out)

        return ff_out


class MultiHeadAttentionDecodingLayer(nn.Module):

    def __init__(self, n_heads, d_model, dk, dv, p_drop, dff):
        super(MultiHeadAttentionDecodingLayer, self).__init__()

        # Multi head self attention x 6
        self.masked_multi_head_attention = MultiheadAttention(n_heads, d_model, dk, dv, p_drop, dff)
        self.norm1 = nn.LayerNorm(d_model)

        self.multi_head_attention = MultiheadAttention(n_heads, d_model, dk, dv, p_drop, dff)
        self.norm2 = nn.LayerNorm(d_model)

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
    def forward(self, src, target):

        # TODO implement mask
        output = self.mask(target)

        # TODO are we supposed to be using the embedding Q, K, V???

        # First attention layer
        attn_out_1 = self.masked_multi_head_attention(output)
        attn_out_1 = self.norm1(attn_out_1 + output)

        attn_out_2 = self.multi_head_attention(attn_out_1)


        # Feed forward x 2  # TODO does it make sense that there is no activation applied to the last one?
        ff_out = self.activation(attn_out.matmul(self.W1) + self.b1)
        ff_out = ff_out.matmul(self.W2) + self.b2

        # Connect second residual and apply layer norm
        ff_out = self.layer_norm2(ff_out + attn_out)

        return ff_out

    def mask(self, output):
        return output


class MultiheadAttention(nn.Module):

    def __init__(self, n_heads, d_model, dk, dv, p_drop, dff):
        super(MultiheadAttention, self).__init__()
        self.n_heads = n_heads
        self.heads = [AttentionHead(d_model, dk, dv, p_drop) for _ in range(n_heads)]

        # Learned attention output matrix
        self.WO = torch.randn(self.n_heads * dk, d_model)

    @forward
    def forward(self, src):
        """
        :param src: either the input from the previous layer of the transformer, or the
        original embedding sequence.
        """
        Z = []
        for head in self.heads:
            # Pass the encoded vector to each head
            head_out = head(src, src, src)
            # Save
            Z.append(head_out)

        # Concatenate the outputs of all heads
        out = torch.cat(Z, dim=1)

        # Multiply by learned embedding matrix WO
        out = out.matmul(self.WO)

        return out


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
    def forward(self, queries, keys, values):
        """
        queries, keys, and values will be multiplied by their respective parameter
        matrices to make the q, k, v matrices used in attention.

        :param queries: An embedding matrix of dimension (output_len, d_model)
        :param keys: An embedding matrix of dimension (seq_len, d_model)
        :param values: An embedding matrix of dimension (seq_len, d_model)

        In a self-attention layer all of the keys, values and queries come
        from the previous layer in the encoder.
        """

        q = queries.matmul(self.WQ)
        k = keys.matmul(self.WK)
        v = values.matmul(self.WV)

        att = q.matmul(k.T) / math.sqrt(self.dk)
        z = torch.softmax(att, dim=1).matmul(v)

        return z


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
    sentence = 'The cat jumped over the hat and died I think.'.split(' ')
    net.forward(sentence)