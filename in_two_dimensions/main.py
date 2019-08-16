from flask import Flask, render_template, url_for, request, redirect, flash, g
import json
from .grid import compose_grid
app = Flask(__name__)

previous_words = ['north', 'west', 'east', 'south']

@app.route('/')
def base_grid():
    # Initial value for the 4 word vector directions
    global previous_words
    return compose_and_render(previous_words)


@app.route('/', methods=['POST'])
def grid_post():
    try:
        n = int(request.form['n'])
        print('heres n', n)

        return compose_and_render([
            request.form['word1'],
            request.form['word2'],
            request.form['word3'],
            request.form['word4']
        ], n=n)

    except KeyError as e:
        print("Word not in dictionary!\n", e)
        return redirect(url_for('base_grid'))


def compose_and_render(words, n=5):
    global previous_words

    assert(len(words) == 4)

    for i, word in enumerate(words):
        if word.strip() == '':
            print(i, word, "is the same")
            words[i] = previous_words[i]

    # Get the generated words (an nxn grid) and their corresponding coordinates on the page.
    # Each word is linearly interpolated between the 4 passed in
    grid, coordinates = compose_grid(words[0], words[1], words[2], words[3], n=n)
    print(grid)
    grid, coordinates = json.dumps(grid), json.dumps(coordinates)

    previous_words = words

    return render_template('grid.html', **locals())