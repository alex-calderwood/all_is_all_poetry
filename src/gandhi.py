from corpus import VectorSpace, Corpus
import pickle
import os.path

g_pic = 'gandhi_pickle.dat'

def embed_gandhi_into_vector_space():
    ''''
    Read all of the works of gandhi and embed him into a vector space
    Must first run scrape_gandhi.py according to its usage directions. 
    '''

    print('Embedding Gandhi into vector space...')

    space = VectorSpace(Corpus('../corpus/gandhi/gandhi.txt', codec='iso-8859-1'))
    pickle.dump(space, open(g_pic, 'wb'))

    print(space.vector('all'))


def translate(string, model):

    string = string.split(' ')

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
    # Check if the preloaded gandhi vector space model exists, or create one otherwise
    if not os.path.isfile(g_pic):
        embed_gandhi_into_vector_space()

    # Load the vector space pickle
    vector_space = pickle.load(open(g_pic, 'rb')).model

    # a = vector_space.wv.most_similar(positive=['woman', 'king'], negative=['man'])

    gandhi_speak = translate('I think that would qualify as not smart , but genius .... and a very stable genius at that !', vector_space)
    print(gandhi_speak)