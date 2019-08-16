from gensim import downloader  # Documentation here: https://github.com/RaRe-Technologies/gensim-data
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.manifold.t_sne import TSNE
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cosine
from scipy.interpolate import interp1d, interp2d

# Notes:
# https://medium.com/@aneesha/using-tsne-to-plot-a-subset-of-similar-words-from-word2vec-bb8eeaea6229

name = 'glove-wiki-gigaword-100'

print('retrieving model / corpus')

filename = os.path.join('./assets/', name + '.pickle')

if os.path.exists(filename):
    print('loading cached model')
    model = pickle.load(open(filename, 'rb'))
else:  # Train a new model
    print('downloading model')
    model = downloader.load(name)
    pickle.dump(model, open(filename, 'wb'))


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


def plot(word):
    word_matrix, words = get_similar_matrix(word)
    Y = TSNE(n_components=2).fit_transform(word_matrix)

    # print(Y)

    plt.scatter(Y[:, 0], Y[:, 1], alpha=0)

    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-3 * len(label), 0), textcoords='offset points')

    plt.axis('off')
    plt.title(word, fontsize=20)
    plt.show()


def interpolate(w, v, n=10):
    interp = interp1d([0, 1], [w, v], axis=0)
    return interp(np.linspace(0, 1, n))


def sample_between(a, b, n):

    g = np.linalg.norm

    # Compute distance
    # d = np.linalg.norm(b - a)
    # Vector pointing towards b
    c = (b - a)

    # Halfway between a and b
    h = (c / 2) + a
    points = []
    for i in range(n):
        points.append(g(c / 2) + h)

    plt.scatter(points, [0] * n)


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


def words_between2D(vector_grid, exclude=[]):
    """
    Given an (n x m x d) grid of word vectors of dimension d, return a (n x m) matrix of their corresponding most similar words.
    :param vector_grid: the vector grid
    :param exclude: perform the "cheat" described here https://arxiv.org/pdf/1905.09866.pdf
        and in the gensim most_similar function. In other words, don't allow any of these words to be in the returned grid
    :return: the new word grid
    """
    word_grid = []
    for i in range(vector_grid.shape[0]):
        row = []
        for j in range(vector_grid.shape[1]):
            row.append(similar_by_vector(vector_grid[i][j], exclude))
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


def plot_along_space(w1, w2, n=25):

    visualize_word(w1)
    visualize_word(w2)

    w = model[w1]
    v = model[w2]

    vectors_between = interpolate(w, v, n=n)
    words = words_between(vectors_between)

    interesting_words = [word for word in words if word not in (w1, w2)]

    if len(interesting_words) > 0:
        print(interesting_words)
    else:
        print("no interesting words")

    matrix = np.array(vectors_between)
    Y = PCA(n_components=2).fit_transform(matrix)
    print(Y.shape)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('')

    ax1.set_title(w1 + ' - ' + w2, fontsize=20)
    ax1.bar(range(model.vector_size), w)
    ax1.bar(range(model.vector_size), v)

    ax2.scatter(Y[:, 0], Y[:, 1], alpha=0)

    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        ax2.annotate(label, xy=(x, y), xytext=(-3.5 * len(label), 0), textcoords='offset points')

    ax2.axis('off')
    ax2.set_title('interpolation', fontsize=20)
    plt.show()

    return matrix


def word_grid(w_up, w_down, w_left, w_right, extra_exclude=[], output=None, plot=False, n=7):

    w = model[w_up]
    v = model[w_down]
    x = model[w_left]
    y = model[w_right]

    print('plotting ({} {}) ({} {})'.format(w_up, w_down, w_left, w_right))

    # Create an (n x n x vocab_len) grid
    vector_grid = interpolate2D(w, v, x, y, n=n)
    word_grid = words_between2D(vector_grid, exclude=[w_up, w_down, w_left, w_right] + extra_exclude)

    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])

    coordinate_grid = np.array([np.array([rotation_matrix.dot(np.array([j, i])) for i in range(n)]) for j in range(n)])

    if plot:
        plt.scatter(range(n), range(n), alpha=0)

        for i in range(n):
            for j in range(n):
                label = word_grid[i][j]
                plt.annotate(label, xy=(i, j), xytext=(-2.5 * len(label), 0), textcoords='offset points')

        plt.axis('off')
        # plt.set_title('interpolation', fontsize=20)
        plt.show()

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