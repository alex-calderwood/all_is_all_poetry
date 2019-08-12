import os
from gensim import downloader  # Documentation here: https://github.com/RaRe-Technologies/gensim-data
from gensim.models.word2vec import Word2Vec # https://radimrehurek.com/gensim/models/word2vec.html
import dill as pickle

name = 'glove-wiki-gigaword-100'

print('retrieving model / corpus')

filename = os.path.join('./models/', name + '.pickle')

if os.path.exists(filename):
    print('loading cached model')
    model = pickle.load(open(filename, 'rb'))
else:
    print('downloading model')
    model = downloader.load(name)
    pickle.dump(model, open(filename, 'wb'))


# v = model.most_similar(positive=['king', 'queen'], negative=['man'], topn=1)[0]
# model.most_similar_to_given('president', ['truck', 'running', 'jumped'])
# print(v)