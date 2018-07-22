from corpus import VectorSpace, Corpus
import pickle
import os.path
from nltk.tokenize import word_tokenize
import argparse

PICKLE = 'gandhi_pickle.dat'


def embed_corpus_into_vector_space(file, save_to=None):
    """'
    Read all of the works of the given corpus and embed it into a vector space. Serialize the vector space into a
    """

    print('Embedding corpus into vector space...')

    corpus = Corpus(file, codec='iso-8859-1')

    space = VectorSpace(corpus)

    if save_to:
        pickle.dump(space, open(save_to, 'wb'))

    print(space.vector('all'))


def translate(string, model):

    string = word_tokenize(string)

    new_string = ''
    for word in string:

        similar_words = []

        try:
            similar_words = model.wv.most_similar(positive=[word])
        except KeyError:
            pass

        if len(similar_words) != 0:
            new_string += ' ' + similar_words[0][0]
        else:
            new_string += ' ' + word

    return new_string


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('phrase', help='The phrase to translate.')
    args = parser.parse_args()

    file = '../corpus/gandhi/gandhi.txt'
    # Check if the pre-loaded gandhi vector space model exists, or create one otherwise
    if not os.path.isfile(PICKLE):
        embed_corpus_into_vector_space(file, save_to=PICKLE)

    # Load the vector space pickle
    vector_space = pickle.load(open(PICKLE, 'rb')).model

    gandhi_speak = translate(args.phrase, vector_space)
    print(gandhi_speak)

    # print(gandhi_speak)
    # p = None
    # p = ['home']
    # n = None
    # n = ['building']

    # sim = vector_space.wv.most_similar(positive=p, negative=n)
    # print(sim)


