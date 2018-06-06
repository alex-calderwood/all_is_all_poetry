import nltk
from collections import Counter
from src.corpus import  Corpus

END = '<END>'
PREFIX = '<PRE-{}>'

class Token():

    def __init__(self, features):
        self.n = len(features)
        self.features = features


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

