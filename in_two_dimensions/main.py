from flask import Flask, render_template, url_for, request
import json
from .grid import word_grid
app = Flask(__name__)

@app.route('/')
def grid():
    words = [word1, word2, word3, word4] = ['north', 'west', 'east', 'south']
    return do_grid(words)

@app.route('/', methods=['POST'])
def grid_post():
    return do_grid([
        request.form['word1'],
        request.form['word2'],
        request.form['word3'],
        request.form['word4']
    ]);

def do_grid(words):
    assert(len(words) == 4)
    grid, coordinates = word_grid(words[0], words[1], words[2], words[3], n=3)
    grid, coordinates = json.dumps(grid), json.dumps(coordinates)
    return render_template('grid.html', **locals())