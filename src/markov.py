import nltk
from collections import Counter
from src.corpus import  Corpus


class PoemEngine():
    def train(self):
        pass

    def predict(self):
        pass


class MarkovChain(PoemEngine):
    
    def __init__(self, N):
        """
        Reads a corpus, a list of sentences. A sentence is a list of tokens. Each token is of an arbitrary number of features.
        :param N: the length of the markov chain, e.g. 2. 
        """

        self.N = N


    def train(self, corpus):
        self.ngrams = self.count_ngrams(corpus)


    def count_ngrams(self, corpus):
        """
        Make ngrams from a corpus.
        """

        ngrams = Counter()

        # Add each ngram for n in {1..N} to the counter
        for n in range(1, self.N + 1):
            self.__count_ngrams(n, ngrams, corpus)

        return ngrams

    def __count_ngrams(self, n, ngrams, corpus):
        """
        For every new ngram seen, add it to the running count.
        :param n: The current value of n.
        :param ngrams: The running counter.
        :param corpus: The corpus to count.
        :return:
        """

        for sequence in corpus:
            sequence_length = len(sequence.augmented())
            for token_index in range(sequence_length):

                if token_index + n > sequence_length:
                    continue

                # Get the current ngram
                ngram = sequence.augmented()[token_index: token_index + n]

                # Add the ngram to the running count
                ngrams[tuple(ngram)] += 1

    def print_counts(self):
        for ngram, count in self.ngrams.items():
            print(ngram, count)


if __name__ == '__main__':
    corpus = Corpus('testcorpus.txt', 2)
    mk = MarkovChain(2)
    mk.train(corpus)
    mk.print_counts()

