from gensim import downloader  # Documentation here: https://github.com/RaRe-Technologies/gensim-data
import os
import dill as pickle
import numpy as np
from scipy.spatial.distance import cosine
from scipy.interpolate import interp1d, interp2d
from gensim.models import KeyedVectors

# Original idea with TSNE:
# https://medium.com/@aneesha/using-tsne-to-plot-a-subset-of-similar-words-from-word2vec-bb8eeaea6229


def load_model():
    name = 'glove-wiki-gigaword-100'

    print('retrieving gensim model')
    try:
        print('looking for cached model')
        model = KeyedVectors.load(name)
    except Exception as e:
        print(e)
        print('couldn\'t lodad cached model. Downloading a fresh copy...')
        model = downloader.load(name)
        print('Saving', name)
        model.save(name)
    print('Model loaded.')

    return model


model = load_model()


def get_similar_matrix(word, n=25):
    sim = [(word, 1)] + model.most_similar(word, topn=n)
    matrix = np.empty((0, model.vector_size), dtype='f')
    word_labels = []
    for word in sim:
        try:
            word_labels.append(word[0])
            matrix = np.append(matrix, np.array([model[word[0]]]), axis=0)
        except ValueError:
            print(model[word[0]])

    print('vector_size', model.vector_size, 'shape', np.shape(matrix))

    return matrix, word_labels


# def plot(word):
#     word_matrix, words = get_similar_matrix(word)
#     Y = TSNE(n_components=2).fit_transform(word_matrix)
#
#     # print(Y)
#
#     plt.scatter(Y[:, 0], Y[:, 1], alpha=0)
#
#     for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
#         plt.annotate(label, xy=(x, y), xytext=(-3 * len(label), 0), textcoords='offset points')
#
#     plt.axis('off')
#     plt.title(word, fontsize=20)
#     plt.show()


def interpolate(w, v, n=10):
    interp = interp1d([0, 1], [w, v], axis=0)
    return interp(np.linspace(0, 1, n))


def interpolate2D(a, b, c, d, n=10):
    assert(a.shape == b.shape == c.shape == d.shape)
    dim = len(a)

    # Define the interpolated values between the vectors a to b and the vectors c to d
    x = np.linspace(a, b, num=n)
    y = np.linspace(c, d, num=n)

    # Define a square grid with interpolations from a1 to c1 down to b1 to d1
    grid = np.empty((n, n, dim))
    print('grid shape', grid.shape)
    for (i, (xi, yi)) in enumerate(zip(x, y)):
        between = np.linspace(xi, yi, n)
        for j in range(n):
            grid[i][j] = between[j]

    return grid


def similar_by_vector(vector, exclude):
    sim_words = [word for word, _ in model.similar_by_vector(vector, topn=10)]

    for word in sim_words:
        if word not in exclude:
            return word

    return ''


def words_between(vectors_between):
    words = [model.similar_by_vector(vec, topn=1)[0][0] for vec in vectors_between]
    return words


def words_between2D(vector_grid, exclude=None, true_corners=False):
    """
    Given an (n x m x d) grid of word vectors of dimension d, return a (n x m) matrix of their corresponding most similar words.
    :param vector_grid: the vector grid
    :param exclude: perform the "cheat" described here https://arxiv.org/pdf/1905.09866.pdf
        and in the gensim most_similar function. In other words, don't allow any of these words to be in the returned grid
    :return: the new word grid
    """

    if exclude is None:
        exclude = []

    # Define the four corner points
    corners = [
        (0, 0),
        (vector_grid.shape[0] - 1, 0),
        (0, vector_grid.shape[1] - 1),
        (vector_grid.shape[0] - 1, vector_grid.shape[1] - 1)
    ]

    word_grid = []
    for i in range(vector_grid.shape[0]):
        row = []
        for j in range(vector_grid.shape[1]):
            if not true_corners or (i, j) not in corners:
                # Append the most similar word other than the words to be excluded
                row.append(similar_by_vector(vector_grid[i][j], exclude))
            else:
                # Append the most similar word, which should be one of the four input words
                row.append(similar_by_vector(vector_grid[i][j], []))
        word_grid.append(row)

    return np.array(word_grid)


def cos(w, v):
    cosine(w, v)


def visualize_word(w):
    v = model[w]

    z = np.zeros(v.shape)

    pos_v = np.abs(v)
    max_index = np.argmax(pos_v, axis=0)
    print('max index', max_index)

    z[max_index] = v[max_index]

    sim = model.similar_by_vector(z, topn=3)
    nsim = model.similar_by_vector(-z, topn=3)
    print('sim', sim, 'not sim', nsim)


def walk_zero_space():
    for i in range(model.vector_size):
        z = np.zeros(model.vector_size)
        z[i] = 1
        sim = model.similar_by_vector(z, topn=3)
        nsim = model.similar_by_vector(-z, topn=3)
        print(i, 'sim', sim, 'not sim', nsim)


def compose_grid(w_up, w_down, w_left, w_right, extra_exclude=[], true_corners=False, output=None, n=7):

    # Get the four word vectors from the model
    w = model[w_up]
    v = model[w_down]
    x = model[w_left]
    y = model[w_right]

    print('plotting ({} {}) ({} {})'.format(w_up, w_down, w_left, w_right))

    # Create an (n x n x vocab_len) grid
    vector_grid = interpolate2D(w, v, x, y, n=n)

    # Use the vector grid to create an (n x n) grid of words
    word_grid = words_between2D(vector_grid, exclude=[w_up, w_down, w_left, w_right] + extra_exclude, true_corners=true_corners)

    # Rotote the grid 45 degrees counterclockwise for aesthetics
    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    coordinate_grid = np.array([np.array([rotation_matrix.dot(np.array([j, i])) for i in range(n)]) for j in range(n)])

    grid_string = ''

    for line in word_grid:
        grid_string += '\t\t'.join([word for word in line]) + '\n'

    if output:
        with open(output, 'w') as f:
            f.write(grid_string)

    # print(coordinate_grid)

    word_grid = word_grid.flatten().tolist()
    coordinate_grid = coordinate_grid.reshape((-1, 2)).tolist()
    return word_grid, coordinate_grid