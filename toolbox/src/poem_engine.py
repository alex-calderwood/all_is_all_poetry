from collections import Counter
from src.corpus import NgramCorpus, Sequence, END
from abc import ABC, abstractmethod


class PoemEngine(ABC):

    """
                                _   _
                               ( \_/ )
                  _   _       __) _ (__
          _   _  ( \_/ )  _  (__ (_) __)
         ( \_/ )__) _ (__( \_/ )) _ (
        __) _ ((__ (_) __)) _ ((_/ \_)
       (__ (_) __)) _ ((__ (_) __)
          ) _ (  (_/ \_)  ) _ (
         (_/ \_)         (_/ \_)

    An engine that can crank out poems.

    """

    @abstractmethod
    def turn(self, corpus):
        """
        Turn the gears, crank the engine. Aka, train the engine on the given corpus.
        """
        pass

    @abstractmethod
    def generate(self):
        """
        Create something new.
        :return: a new Sequence() object.
        """
        pass


class Grammar():
    def check(self):
        pass


class PoemCritic(ABC):

    @abstractmethod
    def judge(self):
        """
        :return: a score between 0 and 1
        """
        pass


class NaiveBayes(PoemEngine):

    # Maximum number of words to generate in a single call to NaiveBayes.generate()
    MAX_ITER = 100
    
    def __init__(self, N):
        """
        Reads a corpus, a list of sentences. A sentence is a list of tokens. Each token is of an arbitrary number of features.
        :param N: the length of the markov chain, e.g. 2. 
        """

        self.N = N
        self.corpus = None

    def turn(self, corpus):
        self.corpus = corpus

        if self.corpus.N < 1:
            raise RuntimeError('Cannot compute NaiveBayes without N >= 1.')

        if self.N > corpus.N:
            raise RuntimeError('Cannot compute NaiveBayes with n = {} with corpus of smaller n (= {})'.format(self.N, corpus.N))

        self.corpus.count_ngrams()

    def generate(self):

        if self.corpus == None:
            raise RuntimeError('Error. Method PoemEngine.generate() cannot be called before PoemEngine.turn(corpus)')

        sequence = Sequence(prefix_size=self.N - 1)

        prior = tuple(sequence.prefix)
        index = 0
        while index < NaiveBayes.MAX_ITER:
            token = max(self.corpus.n_grams[1], key=lambda y: self.corpus.p(y, prior))[0]
            if token == END:
                break
            sequence.append(token)
            prior = prior[1:] + (token,)
            index += 1

        return sequence


class SimpleNet(PoemEngine):

    def turn(self, corpus):
        pass

    def generate(self):
        pass

    def __init__(self):
        pass


if __name__ == '__main__':
    """
    Example usage.
    """

    # Read in the test corpus.
    corpus = NgramCorpus('../corpus/testcorpus.txt', 3)

    # Instantiate a naive bayes engine
    engine = NaiveBayes(3)

    # Give it a few cranks on the corpus
    engine.turn(corpus)
    corpus.print_counts()

    # Generate new text
    new_line = engine.generate()
    print(new_line)
