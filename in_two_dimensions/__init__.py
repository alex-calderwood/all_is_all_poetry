from flask import Flask

from .main import two_dimensions
app = Flask(__name__)

# This should be added to the parent app if running as a blueprint to another flask app
app.register_blueprint(two_dimensions)