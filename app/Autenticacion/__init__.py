from flask import Blueprint

# Crear el blueprint para el módulo de autenticación
auth_bp = Blueprint('Autenticacion', __name__)

from . import Rutas  # Importar las rutas después de definir el blueprint
