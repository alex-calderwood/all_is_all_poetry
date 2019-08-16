import os
from gensim import downloader  # Documentation here: https://github.com/RaRe-Technologies/gensim-data
from gensim.models.word2vec import Word2Vec # https://radimrehurek.com/gensim/models/word2vec.html
import dill as pickle

"""
Look how simple it is to use gensim! 
This code will download a GLOVE model trained on wikipedia, 
and print out some simple word vector arithmetic results.
"""

name = 'glove-wiki-gigaword-100'

print('retrieving model / corpus')

filename = os.path.join('./assets/', name + '.pickle')

if os.path.exists(filename):
    print('loading cached model')
    model = pickle.load(open(filename, 'rb'))
else:
    print('downloading model')
    model = downloader.load(name)
    pickle.dump(model, open(filename, 'wb'))


v = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)[0]
print("king + woman - man = ", v[0], '(similarity', str(v[1]) + ')')

w = 'president'
sim = model.most_similar_to_given(w, ['human', 'leader', 'machine'])
print('most similar to', w, ':', sim)
