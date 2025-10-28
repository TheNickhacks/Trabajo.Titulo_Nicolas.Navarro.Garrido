from app import db
from flask_login import UserMixin
from datetime import datetime

class Usuario(UserMixin, db.Model):
    __tablename__ = 'usuario'

    id_user = db.Column(db.Integer, primary_key=True)
    id_profesional = db.Column(db.String(12), db.ForeignKey('profesional.rut_profesional'), unique=True, nullable=True)
    correo_electronico = db.Column(db.String(100), unique=True, nullable=False)
    contrasena = db.Column(db.String(255), nullable=False)
    tipo_usuario = db.Column(db.String(50), nullable=False, default='usuario')
    hora_ingreso = db.Column(db.DateTime, default=datetime.utcnow)

    # Relación con Profesional (un usuario puede estar asociado a un profesional)
    # profesional = db.relationship('Profesional', backref='usuarios', lazy=True)

    def __init__(self, id_profesional, correo_electronico, contrasena, tipo_usuario='usuario', hora_ingreso=None):
        self.id_profesional = id_profesional
        self.correo_electronico = correo_electronico
        self.contrasena = contrasena
        self.tipo_usuario = tipo_usuario
        self.hora_ingreso = hora_ingreso if hora_ingreso is not None else datetime.utcnow()

    def get_id(self):
        return str(self.id_user)

    def __repr__(self):
        return f"<Usuario {self.correo_electronico}>"
