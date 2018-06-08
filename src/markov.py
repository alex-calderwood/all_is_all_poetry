import nltk
from collections import Counter
from src.corpus import  Corpus


class PoemEngine():
    def train(self):
        pass

    def predict(self):
        pass


class NaiveBayes(PoemEngine):
    
    def __init__(self, N):
        """
        Reads a corpus, a list of sentences. A sentence is a list of tokens. Each token is of an arbitrary number of features.
        :param N: the length of the markov chain, e.g. 2. 
        """

        self.N = N

    def predict():
        poss_sequences = []
        
        return Math.argmax([(s, prob(s)) for s in poss_sequences])

    def prob(sequence):
        """
        Return the probability of the given sequence.
        """
        p = 1.0

        for trigram in iterate_ngrams(sequence, n=3):
            p *= p(trigram[2], trigram[:-1])

        return p

    def max_y(x):
        max_p = 0
        for y in Y:
            p = p(y, x)
            max_y = None
            if p > max_p:
                max_p = p
                max_y = y
        return max_y

    def p(y, x):
        """
        :return conditional probability of y given x P(y|x) according to estimations given by the corpus. 
        """
        return float(ngrams((x,y))) / float(ngrams(x)) 

    def train(self, corpus):
        self.ngrams = self.count_ngrams(corpus)


    def count_ngrams(self, corpus):
        """
        Make n-grams from a corpus.
        """

        n_grams = Counter()

        # Add each ngram for n in {1..N} to the counter
        for n in range(1, self.N + 1):
            self.__count_ngrams(n, n_grams, corpus)

        return n_grams

    def __count_ngrams(self, n, n_grams, corpus):
        """
        Iterate through the corpus and add every n-gram to the running count.
        :param n: The current value of n.
        :param ngrams: The running counter.
        :param corpus: The corpus to count.
        :return:
        """

        for sequence in corpus:
            # Get the sequence tat is ready to be processed
            seq = sequence.augmented()
            sequence_length = len(seq)

            for token_index in range(sequence_length):

                if token_index + n > sequence_length:
                    continue

                # Get the current n_gram
                n_gram = seq[token_index: token_index + n]

                # Add the n_gram to the running count
                n_grams[tuple(n_gram)] += 1

    def print_counts(self):
        for ngram, count in sorted(self.ngrams.items()):
            print(' '.join(ngram), count)


if __name__ == '__main__':
    corpus = Corpus('testcorpus.txt', 2)
    bayes = NaiveBayes(2)
    bayes.train(corpus)
    bayes.print_counts()

    bayes.proba(Sequence('All is all'))
