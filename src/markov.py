import nltk
from collections import Counter

class Token():

    def __init__(features, n):
        self.features = features
    

class MarkovChain():
    
    def __init__(corpus, n):
        """
        Reads a corpus, a list of sentences. A sentence is a list of tokens. Each token is of an arbitrary number of features.
        :param n: the number length of the markov chain, e.g. 2. 
        """

        self.n = n
        self.ngrams = Counter()
    
        for sentence in corpus:
            for w in range(len(senttence) + 1):
                # set token to the current word, or the end of feed character
                token = sentence[w] if w != len(sentence) else '<END>'
                
                ngrams = []
                for i in range(len(n)):
                    ngrams[i - 1] = make_ngrams(i corpus)

                # set u, v to the previous words or the end 
                u = sentence[w - 2] if w > 1 else '<START-2>'
                v = sentence[w - 1] if w > 0 else '<START-1>'
                
                # Record the ngram (add to the running count of the number of this n-gram seen)
                self.ngrams[(u, v)] += 1

        def make_ngrams(n, corpus):
            ngrams = Counter()

            # TODO Make this work for arbitrary n
                  # prev_n = []
                  # for i in range(n - 1):
                  #    token = sentence[w - (i + 1)]
                  #    ngram[i] = sentence[token]

            return ngrams 
