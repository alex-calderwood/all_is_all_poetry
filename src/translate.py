from corpus import VectorSpace, Corpus, tokenize, untokenize
import pickle
import os.path
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

    string = tokenize(string)
    seq = []
    for word in string:

        similar_words = []

        try:
            similar_words = model.wv.most_similar(positive=[word])
        except KeyError:
            pass

        if Corpus.is_stopword(word) or len(similar_words) == 0:
            new = word
        else:
            new = similar_words[0][0]

        seq.append(new)

    return untokenize(seq)


def use(name=None):
    return '''python gandhi.py \"And all is all and each all, and infinite the glory.\"
    Result: And all is all and each all, and limitless the lustre.
           '''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=use())
    parser.add_argument('phrase', help='The phrase to translate.')
    args = parser.parse_args()

    file = '../corpus/gandhi/gandhi.txt'
    # Check if the pre-loaded gandhi vector space model exists, or create one otherwise
    if not os.path.isfile(PICKLE):
        embed_corpus_into_vector_space(file, save_to=PICKLE)

    # Load the vector space pickle
    vector_space = pickle.load(open(PICKLE, 'rb')).model

    translated = translate(args.phrase, vector_space)
    print(translated)
