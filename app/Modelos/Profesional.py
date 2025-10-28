from app import db
from .Historial import Historial

class Profesional(db.Model):
    __tablename__ = 'profesional'
    rut_profesional = db.Column(db.String(12), primary_key=True, unique=True, nullable=False)
    nombre = db.Column(db.String(100), nullable=False)
    app_paterno = db.Column(db.String(100), nullable=False)
    app_materno = db.Column(db.String(100))
    especialidad = db.Column(db.String(100), nullable=False)
    edad = db.Column(db.Integer, nullable=False)

    # Relación con Historial
    historiales = db.relationship('Historial', backref='profesional', lazy=True)

    def __init__(self, rut_profesional, Nombre, App_Paterno, App_Materno, Especialidad, Edad):
        self.rut_profesional = rut_profesional
        self.nombre = Nombre
        self.app_paterno = App_Paterno
        self.app_materno = App_Materno
        self.especialidad = Especialidad
        self.edad = Edad

    def __repr__(self):
        return f"<Profesional {self.Nombre} {self.App_Paterno}>"