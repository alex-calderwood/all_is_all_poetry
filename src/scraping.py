import re, codecs
import requests
import time
import sys
import subprocess
from collections import Counter
import os
import utils

verbose = False

base_url = 'http://www.gandhiashramsevagram.org/gandhi-literature/mahatma-gandhi-collected-works-volume-{}.pdf'
base_gandhi_file = '../corpus/gandhi/pdf/volume_{}.pdf'
base_gandhi_text = '../corpus/gandhi/text/volume_{}.txt'
base_clean_file = '../corpus/gandhi/text/clean_volume_{}.txt'
gandhi = '../corpus/gandhi/gandhi.txt'


def donwload_pdf(url, save_name):
    response = requests.get(url)

    with open(save_name, 'wb') as f:
        f.write(response.content)


def download_gandhi():
    # Download each of the 98 pdfs from the site http://www.gandhiashramsevagram.org/
    for i in range(1, 99):
        save_file = base_gandhi_file.format(i)

        if verbose:
            print('Downloading', save_file + '...')

        donwload_pdf(base_url.format(i), save_file)


def pdf_to_text():
    for i in range(1, 99):

        gandhi_text_file = base_gandhi_text.format(i)

        if verbose:
            print('Converting PDF to', gandhi_text_file + '...')

        # Convert the ith pdf into a txt file and save it in the txt/ directory
        result = subprocess.run(['pdftotext', base_gandhi_file.format(i), gandhi_text_file])
        print(result)


class CorpusCleaner():

    def __init__(self):
        self.cleaners = []
        self.line_cleaners = []

    def add(self, cleaner):
        self.cleaners.append(cleaner)

    def addline(self, cleaner):
        """
        Add a line cleaner to the list of cleaners to use for each line in the input file.
        The cleaner should be a function that returns the cleaned line, or '' if the line should be removed.
        """
        self.line_cleaners.append(cleaner)

    @staticmethod
    def make_simple_line_cleaner(pattern, type):
        pattern = re.compile(pattern)

        def cleaner(text, pattern=pattern):
            if type == 'match':
                return '' if re.match(pattern, text) else text
            if type == 'fullmatch':
                return '' if re.fullmatch(pattern, text) else text
            elif type == 'search':
                return '' if re.search(pattern, text) else text
            else:
                raise NotImplementedError

        return cleaner

    @staticmethod
    def check_regex(regex, text):

        matches = re.finditer(regex, text)

        for match in matches:
            print(match.group(0))
            print('---')

    @staticmethod
    def _re_line(pattern, br='[\n]+[\s]*', end=''):
        """
        Match: a line with all surrounding whitespace.
        """

        return re.compile(br + pattern + br + end)

    def clean_text(self, text):

        if verbose:
            print('Cleaning...')

        # Successively remove all of the text matched by our global patterns
        if verbose:
            print('Global cleaners')
        for cleaner in self.cleaners:
            text, count = re.subn(cleaner, '\n', text)

            if verbose:
                print(cleaner, 'deletions:', count)

        # Successively remove each line that is matched by a line cleaner
        if verbose:
            print('Line cleaners')
        text = text.split('\n')
        clean_text = ''
        deletions = Counter()
        for line in text:

            # print('line', line, end="")
            for cleaner in self.line_cleaners:
                line = cleaner(line)

                if line == '':
                    deletions[cleaner] += 1
                    break

            if line != '':
                clean_text += line + '\n'

        if verbose:
            for cleaner, count in deletions.items():
                print(cleaner, 'deletions', count)

        return clean_text


def make_gandhi_cleaner(volume):
    cleaner = CorpusCleaner()

    cleaner.add(CorpusCleaner._re_line('[0-9]*[\n\s]*THE COLLECTED WORKS OF.*[\n\s]*', end='\f'))  # collected works
    cleaner.add(re.compile('VOL\.\s*{}.*[\n\s]*[0-9]*[\n\s]*\f'.format(volume)))  # vol page break
    cleaner.add(re.compile('\n[a-z|0-9].*[\n\s]+'))  # headers and footers

    simple_line = CorpusCleaner.make_simple_line_cleaner
    cleaner.addline(simple_line('.*photostat.*', 'search')) # description of text source
    cleaner.addline(simple_line('[A-Z0-9 \_,\.\\\-]+', 'fullmatch'))  # only uppercase and numbers
    cleaner.addline(lambda line: '' if len(line) < 50 else line)  # short text

    return cleaner


def gandhi_clean():

    total = 99
    for i in range(1, total):
        # Read the ith text file
        gandhi_text_file = codecs.open(base_gandhi_text.format(i), 'r', 'iso-8859-1')
        text = gandhi_text_file.read()

        cleaner = make_gandhi_cleaner(i)
        clean_text = cleaner.clean_text(text)

        with open(base_clean_file.format(i), 'w', encoding='utf-8') as clean_file:
            clean_file.write(clean_text)

        if verbose:
            print('wrote clean file to', base_clean_file.format(i), end="")

        if not verbose:
            utils.progress_bar(i, total, message='wrote clean file to ' + base_clean_file.format(i), size=75)


def one_big_gandhi():
    # put everything into one big clean file

    total = 99

    # Delete the previous one big clean file
    try:
        os.remove(gandhi)
    except FileNotFoundError:
        print('No previous {} file'.format(gandhi))

    # print('Cleaning text files')
    # gandhi_clean()
    # print('\rDone')

    gandhi_file = open(gandhi, 'w', encoding='utf-8')

    for i in range(1, total):
        clean_text = open(base_clean_file.format(i), 'r', encoding='utf-8').read()
        print(clean_text[:10], end=' ')
        gandhi_file.write(clean_text)

        print('Added all of', base_clean_file.format(i), 'to', gandhi)

    gandhi_file.close()

    for i in range(1, total):
        os.remove(base_clean_file.format(i))

    print('Removed temp clean files.')
    print('Wrote to {}'.format(gandhi))


if __name__ == '__main__':
    one_big_gandhi()