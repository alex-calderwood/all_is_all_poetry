import nltk
from collections import Counter

class Token():

    def __init__(features):
        self.n = len(features)
        self.features = features
    
class Corpus():
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.__sequences = [line.strip().split(' ') for line in f.readlines()]
        
    def __iter__(self):
        return iter(self.__sequences)

    def __getitem__(self, val):
        return self.__sequences[val]
        
    def set_sequences(self, sequences):
        self.__sequences = sequences

    def __str__(self):
        return str(self.__sequences)


class MarkovChain():

    END = '<END>'
    PREFIX = '<PRE-{}>'
    
    def __init__(self, corpus, N):
        """
        Reads a corpus, a list of sentences. A sentence is a list of tokens. Each token is of an arbitrary number of features.
        :param N: the length of the markov chain, e.g. 2. 
        """

        self.N = N

        self.ngrams = self.__count_ngrams(corpus)

        for key, value in self.ngrams:
             print(key, value)

    def __count_ngrams(self, corpus):
        """
        Untested. Make ngrams from a corpus.
        """

        ngrams = Counter()

        # Make each ngram for n in {1..N}
        for sequence in corpus:
            for token_index in len(sequence):
                for i in reversed(range(1, self.N)):
                    ngram = sequence[token_index - i: token_index]
                    ngrams[ngram] += 1

        return ngrams

    def preprocess_corpus(self, corpus):
        """
        Add the prefix and suffix tokens for each sequence in the corpus.
        """
        sequences = []
        for sequence in corpus:
            sequence = [PREFIX.format(i) for i in range(1, self.N + 1)] + sequence + [END]

        corpus.set_sequences(seqeunces)


if __name__ == '__main__':
    corpus = Corpus('testcorpus.txt')
    print(corpus)
    mk = MarkovChain(2, corpus)

