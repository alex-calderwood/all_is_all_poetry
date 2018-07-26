from corpus import tokenize
from collections import defaultdict
import re
import subprocess

"""
Created by Alex Calderwood in 2018

Turns a list of english word tokens into a list of pronunciations according to the CMU pronunciation dictionary.
"""


CMU_DICT = "../corpus/cmudict/cmudict.dict"

SUFFIX = re.compile('\([0-9]+\)')


def load_dict(file_name):
    """
    Load the CMU pronunciation dictionary.
    :return a dictionary `pronunce` such that
        pronounce[word] = a list of all valid ARPAbet pronunciations
    """

    with open(file_name, 'r') as file:
        lines = file.readlines()
        pronunce = defaultdict(list)
        for line in lines:
            line = line.split(' ')
            word = re.sub(SUFFIX, '', line[0])
            pronunce[word].append(line[1:])

        return pronunce


def say(text):
    subprocess.call(['say', str(text)])


def get_pronunciations(words, pronounce):
    """
    Map from words to their pronunciations
    TODO: Figure out a better way to handle None result in dictionary.
    """

    pronunciations = []
    for word in words:
        lowercase = word.lower()
        pronunciation = pronounce.get(lowercase)
        pronunciation = [phoneme.strip() for phoneme in pronunciation[0]] \
            if pronunciation is not None and len(pronunciation) > 0 \
            else []
        pronunciations.append(pronunciation)

    return pronunciations


if __name__ == "__main__":

    text = "Hello World."
    tokens = tokenize(text)
    print('Input:', tokens)

    syllables = get_pronunciations(tokens, load_dict(CMU_DICT))

    print('Pronunciation:', syllables)