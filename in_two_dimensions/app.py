from flask import Flask, session
from flask_session import Session
from flask import Blueprint, render_template, url_for, request, redirect
# from . import app
import json
from grid import compose_grid
from random import choice

app = Flask(__name__)

app.secret_key = 'asdfasdfasdfasdf1111'

# Set up the user session object
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

# Set the version number
# This is used when deciding when to update the javascript file grid.js
VERSION = '1.0.8'


def seed_words():
    return choice([
        ['north', 'south', 'west', 'east'],
        ['up', 'down', 'left', 'right']
    ])

# corners = seed_words()
# n = 3
# rotation = None

# Blueprint question https://stackoverflow.com/questions/15583671/flasks-how-to-architect-the-project-with-multiple-apps
# Blueprint doc https://flask.palletsprojects.com/en/1.1.x/blueprints/
# Uncomment if this is being used as a module in a seperate flask app (also see __init__.py)
# two_dimensions = Blueprint('two_dimensions', __name__, url_prefix='/two_dimensions', template_folder='templates', static_folder='static')

def init_session():
    if not session.get('initialized'):
        session['corners'] = seed_words()
        session['n'] = 3
        session['rotation'] = None
        session['true_corners'] = True

        session['is_initialized'] = True
        print('New Session', session.sid)


@app.route('/')
def base_grid():
    # Initialize a user session
    init_session()

    # corners = [up, down, left, right] = session.get('corners')
    corners = [
        request.args.get('up', default=None,type=str),
        request.args.get('down', default=None,type=str),
        request.args.get('left', default=None,type=str),
        request.args.get('right', default=None,type=str),
    ]
    print('init corners', corners)

    n = request.args.get('n', default=session.get('n'), type=int)
    true_corners = session.get('true_corners')

    if all(corners):
        print('YESS')
        try: 
            return compose_and_render(corners, n=n, rotation=session.get('rotation'), true_corners=true_corners)
        except KeyError as e:
            print('args', e.args)
            bad_word = e.args[0].split('\'')[1]
            print("Word not in dictionary!\n", bad_word)

            print('before', corners)

            # Get rid of the error word
            for i in range(len(corners)):
                if corners[i] == str(bad_word):
                    print('replace', corners[i])
                    corners[i] = session.get('corners')[i]
                else:
                    print('not replace', corners[i], str(bad_word))
                print(corners[i])

            [up, down, left, right] = corners

            print('after', corners)
            print('session', session.get('corners'))

            return redirect(url_for('grid_post', up=up, down=down, left=left, right=right))
    else:
        # If not all of the corner arguments have been suplied
        print('NOOOO')
        prev_corners = session.get('corners')
        corners = [up, down, left, right] = [corners[i] if corners[i] else prev_corners[i] for i in range(len(corners))]
        return redirect(url_for('grid_post', up=up, down=down, left=left, right=right, n=n))

@app.route('/', methods=['POST'])
def grid_post():
    init_session()

    # Set the size of the grid based on the form drop down menu
    n = int(request.form['n'])

    # Set a flag based on the form checkbox that determines if the corner 'cheat' should be deactivated
    session['true_corners'] = request.form.get('true_corners')

    # Not currently used
    session['rotation'] = int(request.form['rotation'])

    corners = [
        request.form['up_word'],
        request.form['down_word'],
        request.form['left_word'],
        request.form['right_word'],
    ]

    prev_corners = session.get('corners')

    corners = [up, down, left, right] = [corners[i] if corners[i] else prev_corners[i] for i in range(len(corners))]

    print('form', up, down, left, right)

    return redirect(url_for('base_grid',up=up, down=down, left=left, right=right, n=n))


    # try:
    #     # Set the size of the grid based on the form drop down menu
    #     n = int(request.form['n'])

    #     # Set a flag based on the form checkbox that determines if the corner 'cheat' should be deactivated
    #     true_corners = request.form.get('true-corners')
        
    #     return compose_and_render([
    #         request.form['up_word'],
    #         request.form['down_word'],
    #         request.form['left_word'],
    #         request.form['right_word'],
    #     ], n=n, true_corners=true_corners, rotation=int(request.form['rotation']))

    # except KeyError as e:
    #     print("Word not in dictionary!\n", e)
    #     return redirect(url_for('base_grid'))


def compose_and_render(words, n=3, true_corners=True, rotation=None):
    
    # Get the version number (passed through **locals to the template)
    global VERSION
    version_num = VERSION

    prev_corners = session.get('corners')

    assert(len(words) == 4)

    for i, word in enumerate(words):
        if word.strip() == '':
            words[i] = prev_corners[i]
        else:
            words[i] = words[i].lower()

    # Get the generated words (an nxn grid) and their corresponding coordinates on the page.
    # Each word in the grid is then linearly interpolated between the 4 passed in
    grid_words, coordinates = compose_grid(words[0], words[1], words[2], words[3], n=n, true_corners=true_corners, rotation=rotation)
    grid_words, coordinates = json.dumps(grid_words), json.dumps(coordinates)

    # Save data from this query as session variables
    session['corners'] = words
    session['n'] = n
    session['rotation'] = rotation


    true_corners_checked = 'checked' if true_corners else ''

    # Pass the grid to the template through 'locals'
    return render_template('grid.html', **locals())
