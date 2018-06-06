import unittest
from src.corpus import Corpus

class TestStringMethods(unittest.TestCase):

    def test_iter(self):
        corpus = Corpus('testcorpus.txt')
        print(corpus)

        for s in corpus:
            for i in s:
                print(i)

if __name__ == '__main__':
    unittest.main()