from flask import Flask, render_template, url_for
from .grid import word_grid
app = Flask(__name__)

@app.route('/')
def grid():
    words = [word1, word2, word3, word4] = ['word', 'another', 'yet', 'another']
    grid = word_grid(word1, word2, word3, word4)
    return render_template('grid.html', **locals())
    return grid