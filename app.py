import os
import logging
from flask import Flask
from api.routes import api_bp

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

# Import models after app creation to avoid circular imports
from models import *

# Default route
@app.route('/')
def index():
    from flask import render_template
    return render_template('index.html')

# Documentation route
@app.route('/docs')
def docs():
    from flask import render_template
    return render_template('docs.html')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    from flask import jsonify
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    from flask import jsonify
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize modules
logger.info("Initializing application modules...")
