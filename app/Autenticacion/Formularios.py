from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from app.Modelos.Usuario import Usuario

# Formulario de inicio de sesión
class LoginForm(FlaskForm):
    correo_electronico = StringField('Correo Electrónico', validators=[DataRequired(), Email()])
    password = PasswordField('Contraseña', validators=[DataRequired()])
    remember_me = BooleanField('Recuérdame')
    submit = SubmitField('Iniciar Sesión')

# Formulario de registro de usuario
class RegisterForm(FlaskForm):
    id_profesional = StringField('ID Profesional', validators=[DataRequired(), Length(min=4, max=12)])

    correo_electronico = StringField('Correo Electrónico', validators=[DataRequired(), Length(min=6, max=100)])
    nombre = StringField('Nombre', validators=[DataRequired(), Length(min=2, max=50)])
    app_paterno = StringField('Apellido Paterno', validators=[DataRequired(), Length(min=2, max=50)])
    app_materno = StringField('Apellido Materno', validators=[DataRequired(), Length(min=2, max=50)])
    especialidad = StringField('Especialidad', validators=[DataRequired(), Length(min=2, max=50)])
    edad = StringField('Edad', validators=[DataRequired()])
    password = PasswordField('Contraseña', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirmar contraseña', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Registrar')

    # Validación para asegurarse de que el ID de profesional no esté ya registrado
    def validate_id_profesional(self, id_profesional):
        user = Usuario.query.filter_by(id_profesional=id_profesional.data).first()
        if user:
            raise ValidationError('Este ID de profesional ya está en uso. Elige otro.')

    def validate_correo_electronico(self, correo_electronico):
        user = Usuario.query.filter_by(correo_electronico=correo_electronico.data).first()
        if user:
            raise ValidationError('Este correo electrónico ya está en uso. Elige otro.')

