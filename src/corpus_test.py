import unittest
from src.corpus import Corpus
from src.corpus import Sequence
from src.poem_engine import NaiveBayes


class TestStringMethods(unittest.TestCase):

    def test_iter(self):
        corpus = Corpus('testcorpus.txt', prefix_padding_size=2)

        assert([s.augmented() for s in corpus] == [
            ['~P-2~', '~P-1~', 'and', 'all', 'is', 'all', '~END~'],
            ['~P-2~', '~P-1~', 'and', 'each', 'is', 'all', '~END~'],
            ['~P-2~', '~P-1~', 'and', 'infinite', 'the', 'glory', '~END~']
        ])

        assert ([str(s) for s in corpus] == [
            str(['and', 'all', 'is', 'all']),
            str(['and', 'each', 'is', 'all']),
            str(['and', 'infinite', 'the', 'glory'])
        ])

        corpus = Corpus('testcorpus.txt')

        assert ([s.augmented() for s in corpus] == [
            ['and', 'all', 'is', 'all', '~END~'],
            ['and', 'each', 'is', 'all', '~END~'],
            ['and', 'infinite', 'the', 'glory', '~END~']
        ])

    def test_get(self):
        corpus = Corpus('testcorpus.txt', prefix_padding_size=0)
        sequence = corpus[0]

        self.assertEquals(sequence.list, ['and', 'all', 'is', 'all'])
        self.assertEquals(sequence[2], 'is')

    def test_slice(self):
        corpus = Corpus('testcorpus.txt', prefix_padding_size=0)
        sequence = corpus[0]

        self.assertEquals(sequence.list, sequence[0: len(sequence)])


class TestBayes(unittest.TestCase):
    def test_p(self):
        corpus = Corpus('testcorpus.txt', 3)
        bayes = NaiveBayes(3)
        bayes.turn(corpus)
        p = bayes.p(('is',), ('all',))

        self.assertEquals(1/3.0, p)

        self.assertListEqual(['and', 'all', 'is', 'all'], bayes.generate().list)


if __name__ == '__main__':
    unittest.main()