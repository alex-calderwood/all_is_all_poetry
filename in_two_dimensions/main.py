from flask import Flask, render_template, url_for
import json
from .grid import word_grid
app = Flask(__name__)

@app.route('/')
def grid():
    words = [word1, word2, word3, word4] = ['word', 'another', 'yet', 'another']
    grid, coordinates = word_grid(word1, word2, word3, word4)
    grid, coordinates = json.dumps(grid), json.dumps(coordinates)
    print('grid', grid)
    print('coordinates', coordinates)
    return render_template('grid.html', **locals())