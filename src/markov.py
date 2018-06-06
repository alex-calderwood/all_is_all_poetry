import nltk
from collections import Counter

END = '<END>'
PREFIX = '<PRE-{}>'

class Token():

    def __init__(self, features):
        self.n = len(features)
        self.features = features

class Sequence():
    def __init__(self, list):
        self.__list = list

    def __getitem__(self, key):
        if isinstance(key, slice):
            step = 1 if key.step is None else key.step
            return [self.__getitem__(i) for i in range(key.start, key.stop, step)]

        if key < 0:
            return PREFIX.format(-key)
        elif key == len(self.__list):
            return END
        else:
            return self.__list[key]

    def __len__(self):
        return len(self.__list)
    
class Corpus():
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.__sequences = [Sequence(line.strip().split(' ')) for line in f.readlines()]
        
    def __iter__(self):
        return iter(self.__sequences)

    def __getitem__(self, val):
        return self.__sequences[val]
        
    def set_sequences(self, sequences):
        self.__sequences = sequences

    def __str__(self):
        return str(self.__sequences)


class MarkovChain():
    
    def __init__(self, corpus, N):
        """
        Reads a corpus, a list of sentences. A sentence is a list of tokens. Each token is of an arbitrary number of features.
        :param N: the length of the markov chain, e.g. 2. 
        """

        self.N = N

        self.ngrams = self.__count_ngrams(corpus)

        for thing in self.ngrams:
            print(thing)

    def __count_ngrams(self, corpus):
        """
        Untested. Make ngrams from a corpus.
        """

        ngrams = Counter()

        # Make each ngram for n in {1..N}
        for sequence in corpus:
            for token_index in range(len(sequence) + 1):
                # for n in reversed(range(1, self.N)):
                for n in [1]:
                    ngram = sequence[token_index - n: token_index]
                    print(token_index - n, token_index, ngram)
                    ngrams[tuple(ngram)] += 1

        return ngrams


if __name__ == '__main__':
    corpus = Corpus('testcorpus.txt')
    mk = MarkovChain(corpus, 2)

