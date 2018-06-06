# from six import text_type
from nltk.tokenize.nist import NISTTokenizer

CMU_DICT = "../corpus/cmudict/cmudict.dict"


def tokenize(text):
    """
    Turn a standard english text into a list of words."
    """

    # If the following gives an error, you need to download the correct corpus as in the error message
    tokens = NISTTokenizer().tokenize(text, lowercase=False)
    return tokens


def load_dict(file_name):
    """
    Load the CMU pronunciation dictionary.
    TODO: Add support for second (2) definitions.
    """

    with open(file_name, 'r') as file:
        lines = file.readlines()
        pronunciation_dict = {}
        for line in lines:
            line = line.split(' ')
            pronunciation_dict[line[0]] = line[1:]

        return pronunciation_dict


def get_pronunciations(words):
    """
    Map from words to their pronunciations
    TODO: Figure out a better way to handle None result in dictionary.
    """

    pronounce = load_dict(CMU_DICT)

    pronunciations = []
    for word in words:
        lowercase = word.lower()
        pronunciations.append(pronounce.get(lowercase))

    return pronunciations


if __name__ == "__main__":
    tokens = tokenize("Hello World.")
    print(tokens)

    syllables = get_pronunciations(tokens)

    print(syllables)
