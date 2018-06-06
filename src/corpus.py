class Sequence():
    def __init__(self, list):
        self.__list = list

    def __getitem__(self, key):
        if isinstance(key, slice):
            step = 1 if key.step is None else key.step
            return [self.__getitem__(i) for i in range(key.start, key.stop, step)]

        if key < 0:
            return PREFIX.format(-key)
        elif key == len(self.__list):
            return END
        else:
            return self.__list[key]

    def __len__(self):
        return len(self.__list)


class Corpus():
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.__sequences = [Sequence(line.strip().split(' ')) for line in f.readlines()]

    def __iter__(self):
        return iter(self.__sequences)

    def __getitem__(self, val):
        return self.__sequences[val]

    def set_sequences(self, sequences):
        self.__sequences = sequences

    def __str__(self):
        return str(self.__sequences)