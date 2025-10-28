from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

class Config:
    # Obtener la cadena de conexión a la base de datos desde la variable de entorno
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Desactivar el seguimiento de modificaciones, ya que no es necesario
    SECRET_KEY = os.getenv('SECRET_KEY', 'una_clave_secreta_muy_dificil_de_adivinar') # Clave secreta para protección CSRF y sesiones