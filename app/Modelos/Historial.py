from app import db

class Historial(db.Model):
    __tablename__ = 'historial'
    id_consulta = db.Column(db.Integer, primary_key=True)
    rut_profesional = db.Column(db.String(12), db.ForeignKey('profesional.rut_profesional'), nullable=False)
    fecha = db.Column(db.Date, nullable=False)
    hora = db.Column(db.Time, nullable=False)
    archivo_img = db.Column(db.LargeBinary) # BYTEA en PostgreSQL se mapea a LargeBinary en SQLAlchemy
    diagnostico = db.Column(db.String(255))
    diagnostico_2 = db.Column(db.String(255))
    edad_paciente = db.Column(db.Integer)
    sexo = db.Column(db.String(20))
    lugar_lesion = db.Column(db.String(255))
    mapa_calor = db.Column(db.LargeBinary)
    explicacion = db.Column(db.Text)

    def __repr__(self):
        return f"<Historial {self.id_consulta}>"