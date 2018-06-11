from collections import Counter
from src.corpus import Corpus, Sequence, END


class PoemEngine():
    def train(self, corpus):
        pass

    def generate(self):
        pass


class NaiveBayes(PoemEngine):
    
    def __init__(self, N):
        """
        Reads a corpus, a list of sentences. A sentence is a list of tokens. Each token is of an arbitrary number of features.
        :param N: the length of the markov chain, e.g. 2. 
        """

        self.N = N
        self.n_grams = []

    def generate(self):

        # Initialize the sequence
        sequence = Sequence(prefix_size=self.N - 1)

        prior = tuple(sequence.prefix)
        index = 0
        while index < 100:
            token = max(self.n_grams[1], key=lambda y: self.p(y, prior))[0]
            if token == END:
                break
            sequence.append(token)
            prior = prior[1:] + (token,)
            index += 1

        return sequence

    def prob(self, sequence):
        """
        Return the probability of the given sequence.
        """
        p = 1.0

        for trigram in self.generate_n_grams(sequence, n=3):
            p *= p(trigram[2], trigram[:-1])

        return p

    def p(self, y, x):
        """
        p(y|x) according to estimations given by the corpus.
        :return conditional probability of y given x
        """
        u = len(x)
        v = len(y)
        w = u + v

        return float(self.n_grams[w][tuple(x) + tuple(y)]) / float(self.n_grams[u][x])

    def train(self, corpus):
        self.n_grams = self.count_ngrams(corpus)

    def count_ngrams(self, corpus):
        """
        Make n-grams from a corpus.
        """

        # Initialize n_grams, appending an empty '0'-gram dictionary
        n_grams = [{}]

        # Add each ngram for n in {1..N} to the counter
        for n in range(1, self.N + 1):
            n_grams.append(self.__count_ngrams(n, corpus))

        return n_grams

    @staticmethod
    def generate_n_grams(sequence, n=2):
        sequence_length = len(sequence)
        for token_index in range(sequence_length):
            if token_index + n > sequence_length:
                return

            # Yield the current n_gram
            yield sequence[token_index: token_index + n]

    @staticmethod
    def __count_ngrams(n, corpus):
        """
        Iterate through the corpus and add every n-gram to the running count.
        :param n: The current value of n.
        :param corpus: The corpus to count.
        :return: The counts of all n-grams.
        """
        n_grams = Counter()

        for sequence in corpus:
            seq = sequence.augmented()

            for n_gram in NaiveBayes.generate_n_grams(seq, n):
                n_grams[tuple(n_gram)] += 1

        return n_grams

    def print_counts(self):

        for n in range(1, self.N + 1):
            print('{}-grams'.format(n))
            for ngram, count in sorted(self.n_grams[n].items(), reverse=True):
                print(ngram, count)


if __name__ == '__main__':
    corpus = Corpus('testcorpus.txt', 3)
    bayes = NaiveBayes(3)
    bayes.train(corpus)
    bayes.print_counts()

    print(bayes.generate())
