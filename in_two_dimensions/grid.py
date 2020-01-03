from gensim import downloader  # Documentation here: https://github.com/RaRe-Technologies/gensim-data
import numpy as np
import dill as pickle
from scipy.interpolate import interp1d, interp2d
from gensim.models import KeyedVectors
import math
from nltk.stem.snowball import SnowballStemmer

# Original idea with TSNE:
# https://medium.com/@aneesha/using-tsne-to-plot-a-subset-of-similar-words-from-word2vec-bb8eeaea6229

phrase = False

stemmer = SnowballStemmer(language='english')

def load_model():
    # Pick a model
    if phrase:
        name = "word2vec-google-news-300"
    else:
        name = 'glove-wiki-gigaword-100'

    heroku_file_path = ''

    pickle_name = heroku_file_path + name + '.pickle'

    print('retrieving gensim model', name)
    try:
        print('looking for cached model')
        # model = KeyedVectors.load(name)
        model = pickle.load(open(pickle_name, 'rb'))
    except Exception as e:
        print(e)
        print('couldn\'t lodad cached model. Downloading a fresh copy...')
        model = downloader.load(name)
        print('Saving', name)
        # model.save(name)
        pickle.dump(model, open(pickle_name, 'wb'))
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


def interpolate(w, v, n=10):
    interp = interp1d([0, 1], [w, v], axis=0)
    return interp(np.linspace(0, 1, n))


def grid_2D_interpolate(a, b, c, d, n=10):
    assert(a.shape == b.shape == c.shape == d.shape)
    dim = len(a)

    # Define the interpolated values between the vectors a to b and the vectors c to d
    x = np.linspace(a, d, num=n)
    y = np.linspace(c, b, num=n)

    # Define a square grid with interpolations from each x[i] to each y[i]
    grid = np.empty((n, n, dim))
    for (i, (xi, yi)) in enumerate(zip(x, y)):
        between = np.linspace(xi, yi, n)
        for j in range(n):
            grid[i][j] = between[j]

    return grid


def triangle_interoplate(a, b, c, n=3):
    """
    Uses Barycentric Coordinates
    https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    :param a, b, c: numpy word vectors returned from gensim
    """
    assert(a.shape == b.shape == c.shape)

    # Define the 2D points of the final triangle to be drawn
    tri = [
        (0, 0),
        (0, 1),
        (0.5, math.sqrt(1.5)),
    ]

    def barycentric_weights(x, y):
        # Calulate the weights for each point using the 2D barycentric equation
        # https://codeplea.com/triangular-interpolation
        w1 = (tri[1][1] - tri[2][1]) * (x - tri[2][0]) + \
             (tri[2][0] - tri[1][0]) * (y - tri[2][1])
        w2 = (tri[2][1] - tri[0][1]) * (x - tri[2][0]) + \
             (tri[0][0] - tri[2][0]) * (y - tri[2][1])
        d = (tri[1][1] - tri[2][1]) * (tri[0][0] - tri[2][0]) + \
            (tri[2][0] - tri[1][0]) * (tri[0][1] - tri[2][1])
        w1 = w1 / d
        w2 = w2 / d
        w3 = 1 - w1 - w2

        # Todo: this will be sped up (I'm gussing) by using scipy
        #  https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points
        return w1, w2, w3

    def point_in_triangle(w1, w2, w3):
        # 0 <= w1, w2, w3 <= 1
        # w1 + w2 + w3 = 1
        return w1 * a + w2 * b + w3 * c

    min_bound = min([p[0] for p in tri]), min(p[1] for p in tri)
    max_bound = max([p[0] for p in tri]), max(p[1] for p in tri)

    N = 20

    pts = []
    for i in range(N):
        w1, w2, w3 = -1, -1, -1
        while (w1 < 0 or w2 < 0 or w3 < 0): # check that the points are within the triangle
            x = math.random(min_bound[0], max_bound[0])
            y = math.random(min_bound[1], max_bound[1])
            w1, w2, w3 = barycentric_weights(x, y)

        pts.append((x, y, w1, w2, w3))

    # # Simply interpolate between the three to get points on the edges of the triangle
    # ab = np.linspace(a, b, num=n)
    # bc = np.linspace(b, c, num=n)
    # ca = np.linspace(c, a, num=n)

    # each entry in this list will be
    # ((x, y), interpolated_word_vector)
    return [((p[0], p[1]), point_in_triangle(p[2], p[3], p[4])) for p in pts]


def similar_by_vector(vector, exclude):
    sim_words = [word for word, _ in model.similar_by_vector(vector, topn=10)]

    for word in sim_words:
        if word not in exclude and stemmer.stem(word) not in exclude:
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


def compose_grid(w_up, w_down, w_left, w_right, extra_exclude=[], true_corners=False, output=None, n=7, rotation=None):
    # Get the four word vectors from the model
    vec_up = model[w_up]
    vec_down = model[w_down]
    vec_left = model[w_left]
    vec_right = model[w_right]

    print('plotting ({} {}) ({} {})'.format(w_up, w_down, w_left, w_right))

    # Set up the list of words to exclude from the grid, including lemmas of the four corner words
    exclude = [w_up, w_down, w_left, w_right] + extra_exclude
    exclude = [stemmer.stem(word) for word in exclude] + exclude

    # Create an (n x n x vocab_len) matrix
    vector_grid = grid_2D_interpolate(vec_up, vec_down, vec_left, vec_right, n=n)

    # Use the vector grid to create an (n x n) grid of words
    word_grid = words_between2D(vector_grid, exclude=exclude, true_corners=true_corners)

    # Create a coordinate matrix, rotated in space 45 degrees counterclockwise for aesthetics
    theta = 2 * np.pi * rotation / 360 + np.pi/4 if rotation else np.pi/4
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s], [s, c]])
    coordinate_grid = np.array([np.array([rotation_matrix.dot(np.array([j, i])) for i in range(n)]) for j in range(n)])
    normalize(coordinate_grid, n)

    if output:
        grid_string = ''
        for line in word_grid:
            grid_string += '\t\t'.join([word for word in line]) + '\n'
        with open(output, 'w') as f:
            f.write(grid_string)

    words = word_grid.flatten().tolist()
    coordinates = coordinate_grid.reshape((-1, 2)).tolist()

    if phrase:
        words = [word.replace('_', ' ') for word in words]

    return words, coordinates


def normalize(coordinate_grid, n):
    """Normalize all values between the max and min of the coordinates"""
    min_x = math.inf
    min_y = math.inf
    max_x = - math.inf
    max_y = - math.inf
    for i in range(n):
        for j in range(n):
            c = coordinate_grid[i][j]
            if c[0] > max_x:
                max_x = c[0]
            if c[0] < min_x:
                min_x = c[0]
            if c[1] > max_y:
                max_y = c[1]
            if c[1] < min_y:
                min_y = c[1]

    range_x = max_x - min_x
    range_y = max_y - min_y

    for i in range(n):
        for j in range(n):
            c = coordinate_grid[i][j]
            coordinate_grid[i][j] = np.array([
                (c[0] - min_x) / range_x,
                (c[1] - min_y) / range_y
            ])
