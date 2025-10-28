from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from dotenv import load_dotenv
import os

# Inicializar SQLAlchemy
db = SQLAlchemy()

# Inicializar LoginManager
login_manager = LoginManager()

def create_app():
    # Cargar variables de entorno
    load_dotenv()

    app = Flask(__name__)

    # Configuración de la aplicación
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'una_clave_secreta_por_defecto_que_deberias_cambiar')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Inicializar extensiones
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'Autenticacion.login'

    # Registrar blueprints
    from app.Autenticacion import auth_bp
    from app.Diagnosticos import analisis_bp
    from app.Historial import historial_bp
    from app.main import main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(analisis_bp)
    app.register_blueprint(historial_bp)
    app.register_blueprint(main_bp)

    # Importar modelos y crear tablas en la BD (Postgres) al inicio
    from app.Modelos.Usuario import Usuario
    from app.Modelos.Profesional import Profesional
    from app.Modelos.Historial import Historial

    with app.app_context():
        db.create_all()

    return app

    # Manejadores de errores
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errores/404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('errores/500.html'), 500

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('errores/403.html'), 403

    return app
