from flask import Flask
from .main import two_dimensions
app = Flask(__name__)
app.register_blueprint(two_dimensions)