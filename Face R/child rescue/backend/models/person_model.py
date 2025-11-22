from extensions import db

class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    location = db.Column(db.String(100))
    status = db.Column(db.String(10), default="missing")
    images = db.Column(db.Text)
