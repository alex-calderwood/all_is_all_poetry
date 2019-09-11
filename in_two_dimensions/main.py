from flask import Blueprint, render_template, url_for, request, redirect
from . import app
import json
from .grid import compose_grid

prev_corners = ['north', 'west', 'east', 'south']

# Blueprint question https://stackoverflow.com/questions/15583671/flask-how-to-architect-the-project-with-multiple-apps
# Blueprint doc https://flask.palletsprojects.com/en/1.1.x/blueprints/
# Uncomment if this is being used as a module in a seperate flask app (also see __init__.py)
# two_dimensions = Blueprint('two_dimensions', __name__, url_prefix='/two_dimensions', template_folder='templates', static_folder='static')

@app.route('/')
def base_grid():
    # Initial value for the 4 word vector directions
    global prev_corners
    return compose_and_render(prev_corners)


@app.route('/', methods=['POST'])
def grid_post():
    try:
        # Set the size of the grid based on the form drop down menu
        n = int(request.form['n'])

        # Set a flag based on the form checkbox that determines if the corner 'cheat' should be deactivated
        true_corners = request.form.get('true-corners')

        return compose_and_render([
            request.form['word1'],
            request.form['word2'],
            request.form['word3'],
            request.form['word4']
        ], n=n, true_corners=true_corners)

    except KeyError as e:
        print("Word not in dictionary!\n", e)
        return redirect(url_for('base_grid'))


def compose_and_render(words, n=3, true_corners=True):
    global prev_corners

    assert(len(words) == 4)

    for i, word in enumerate(words):
        if word.strip() == '':
            words[i] = prev_corners[i]

    # Get the generated words (an nxn grid) and their corresponding coordinates on the page.
    # Each word in the grid is then linearly interpolated between the 4 passed in
    grid, coordinates = compose_grid(words[0], words[1], words[2], words[3], n=n, true_corners=true_corners)
    grid, coordinates = json.dumps(grid), json.dumps(coordinates)

    prev_corners = words

    true_corners_checked = 'checked' if true_corners else ''

    # Pass the grid to the template through 'locals'
    return render_template('grid.html', **locals())