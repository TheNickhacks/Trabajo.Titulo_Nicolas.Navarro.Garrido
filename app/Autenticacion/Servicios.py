from app import db
from app.Modelos.Usuario import Usuario
import datetime
from werkzeug.security import check_password_hash, generate_password_hash
# Consulta para obtener un usuario por su nombre de usuario
def get_user_by_rut_user(rut_user):
    return Usuario.query.filter_by(rut_user=rut_user).first()

def get_user_by_email(correo_electronico):
    return Usuario.query.filter_by(correo_electronico=correo_electronico).first()

def add_user(rut_user, correo_electronico, hashed_password):
    new_user = Usuario(rut_user=rut_user, correo_electronico=correo_electronico, contrasena=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return new_user

def user_exists(correo_electronico):
    return Usuario.query.filter_by(correo_electronico=correo_electronico).first() is not None
