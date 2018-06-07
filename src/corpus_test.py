import unittest
from src.corpus import Corpus
from src.corpus import Sequence


class TestStringMethods(unittest.TestCase):

    def test_iter(self):
        corpus = Corpus('testcorpus.txt', prefix_padding_size=2)

        assert([s.augmented() for s in corpus] == [
            ['<PRE-2>', '<PRE-1>', 'and', 'all', 'is', 'all', '<END>'],
            ['<PRE-2>', '<PRE-1>', 'and', 'each', 'is', 'all', '<END>'],
            ['<PRE-2>', '<PRE-1>', 'and', 'infinite', 'the', 'glory', '<END>']
        ])

        assert ([str(s) for s in corpus] == [
            str(['and', 'all', 'is', 'all']),
            str(['and', 'each', 'is', 'all']),
            str(['and', 'infinite', 'the', 'glory'])
        ])

        corpus = Corpus('testcorpus.txt')

        assert ([s.augmented() for s in corpus] == [
            ['and', 'all', 'is', 'all', '<END>'],
            ['and', 'each', 'is', 'all', '<END>'],
            ['and', 'infinite', 'the', 'glory', '<END>']
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


if __name__ == '__main__':
    unittest.main()