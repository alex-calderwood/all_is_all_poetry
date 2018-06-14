# Written by Alex Calderwood on June 6th, 2018

from collections import Counter

END = '~END~'
PREFIX = '~P-{}~'

class Token():

    def __init__(self, features):
        self.n = len(features)
        self.features = features


class Sequence():
    """
    Represents a sequence of words or feature vectors. Can also be thought of as a sentence. Each sequence is augmented
    by extra tokens on the beginning and end for use by a prediction algorithm.
    """
    def __init__(self, list=list(), prefix_size=0):
        """
        Initialize a sequence object.
        :param list: The list (of words, feature vectors, etc.) to be made into a sequence.
        :param prefix_size: The size of the prefix sequence, i.e. the number of trailing prefix tokens (of the form '<PRE-{}>').
        """
        self.prefix = self.make_prefix(prefix_size)
        self.list = list
        self.postfix = [END]

    def append(self, token):
        self.list.append(token)

    def __getitem__(self, key):
        return self.list[key]

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return str(self.list)

    def augmented(self):
        """
        :return: the sequence augmented by the prefix <and suffix
        """
        return self.prefix + self.list + self.postfix

    @staticmethod
    def make_prefix(n):
        return [PREFIX.format(i) for i in reversed(range(1, n + 1))]


class Corpus():
    def __init__(self, filename, N=0):
        self.N = N
        self.n_grams = None

        with open(filename, 'r') as f:
            self._sequences_ = [Sequence(line.strip().split(' '), N) for line in f.readlines()]


    @staticmethod
    def from_file(filename, N=0):
        # TODO
        pass

    @staticmethod
    def from_df(dataframe):
        # TODO
        pass

    def __iter__(self):
        return iter(self._sequences_)

    def __getitem__(self, val):
        return self._sequences_[val]

    def set_sequences(self, sequences):
        self._sequences_ = sequences

    def __str__(self):
        return str(self._sequences_)

    def preprocessed(self):
        """
        Should be True if self.count_ngrams() has been called.
        :return: Bool corresponding to whether pre-processing (n_gram computation) has been done on the corpus.
        """

        if self.n_grams is None:
            raise RuntimeError('Attempted to access variable \'n_grams\' before assignemnt. '
                               'Must call Corpus.count_ngrams() first.')

        return self.n_grams is not None

    def p(self, y, x):
        """
        p(y|x) according to estimations given by the corpus.
        :return conditional probability of y given x
        """

        # Throw an exception if there is no n_gram info
        self.preprocessed()

        u = len(x)
        v = len(y)
        w = u + v

        return float(self.n_grams[w][tuple(x) + tuple(y)]) / float(self.n_grams[u][x])

    def count_ngrams(self):
        """
        Make n-grams from a corpus.
        """

        # Initialize n_grams, appending an empty '0'-gram dictionary
        self.n_grams = [{}]

        # Add each ngram for n in {1..N} to the counter
        for n in range(1, self.N + 1):
            self.n_grams.append(Corpus._count_ngrams_(n, self))

    @staticmethod
    def _count_ngrams_(n, corpus):
        """
        Iterate through the corpus and add every n-gram, for some specific n, to the running count.
        :param n: The current value of n.
        :param corpus: The corpus to count.
        :return: The counts of all n-grams.
        """
        n_grams = Counter()

        for sequence in corpus:
            seq = sequence.augmented()

            for n_gram in Corpus.generate_ngrams(seq, n):
                n_grams[tuple(n_gram)] += 1

        return n_grams


    @staticmethod
    def generate_ngrams(sequence, n=2):
        sequence_length = len(sequence)
        for token_index in range(sequence_length):
            if token_index + n > sequence_length:
                return

            # Yield the current n_gram
            yield sequence[token_index: token_index + n]

    def print_counts(self):

        self.preprocessed()

        for n in range(1, self.N + 1):
            print(self.name_grams(n))
            for ngram, count in sorted(self.n_grams[n].items(), reverse=True):
                print(ngram, count)

    @staticmethod
    def name_grams(n):
        if n == 0:
            return 'none-grams'
        elif n == 1:
            return 'unigrams'
        elif n == 2:
            return 'bigrams'
        elif n == 3:
            return 'trigrams'

        return '{}-grams'.format(n)