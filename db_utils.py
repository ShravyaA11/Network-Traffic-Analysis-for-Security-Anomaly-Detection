# utils/db_utils.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Initialize SQLAlchemy object
db = SQLAlchemy()

# ============================
# MODELS
# ============================

class User(db.Model):
    _tablename_ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def _repr_(self):
        return f"<User {self.email}>"

class Alert(db.Model):
    _tablename_ = 'alerts'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100), default=datetime.utcnow)
    attack_type = db.Column(db.String(100))
    src_ip = db.Column(db.String(50))
    dst_ip = db.Column(db.String(50))

    def _repr_(self):
        return f"<Alert {self.attack_type}>"

# ============================
# DATABASE INITIALIZER
# ============================

def init_db(app):
    """Bind SQLAlchemy to Flask app and create tables."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")