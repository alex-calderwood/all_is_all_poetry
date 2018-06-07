# Written by Alex Calderwood on June 6th, 2018

END = '<END>'
PREFIX = '<PRE-{}>'

class Token():

    def __init__(self, features):
        self.n = len(features)
        self.features = features

class Sequence():
    """
    Represents a sequence of words or feature vectors. Can also be thought of as a sentence. Each sequence is augmented
    by extra tokens on the beginning and end for use by a prediction algorithm.
    """
    def __init__(self, list, prefix_size=0):
        """
        Initialize a sequence object.
        :param list: The list (of words, feature vectors, etc.) to be made into a sequence.
        :param prefix_size: The size of the prefix sequence, i.e. the number of trailing prefix tokens (of the form '<PRE-{}>').
        """
        self.prefix = [PREFIX.format(i) for i in reversed(range(1, prefix_size + 1))]
        self.list = list
        self.postfix = [END]

    def __getitem__(self, key):
        return self.list[key]

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return str(self.list)

    def augmented(self):
        """
        :return: the sequence augmented by the prefix <and suffix
        """
        return self.prefix + self.list + self.postfix

    def update_prefix_size(self, new_size):
        self.prefix = [PREFIX.format(i) for i in reversed(range(1, new_size + 1))]


class Corpus():
    def __init__(self, filename, prefix_padding_size=0):
        with open(filename, 'r') as f:
            self.__sequences = [Sequence(line.strip().split(' '), prefix_padding_size) for line in f.readlines()]

    def __iter__(self):
        return iter(self.__sequences)

    def __getitem__(self, val):
        return self.__sequences[val]

    def set_sequences(self, sequences):
        self.__sequences = sequences

    def __str__(self):
        return str(self.__sequences)