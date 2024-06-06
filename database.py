from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_app(app):
    """
    This function binds the previously created SQLAlchemy object ('db')
    to a specific Flask application ('app').
    It should be called from the main application module after the Flask app has been created.
    """
    db.init_app(app)

    # Optionally, you might also want to create tables automatically when you run your app
    with app.app_context():
        db.create_all()  # This will create all tables according to your models definitions