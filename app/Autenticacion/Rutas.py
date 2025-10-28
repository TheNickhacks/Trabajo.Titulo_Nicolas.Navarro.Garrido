from flask import render_template, redirect, url_for, flash, request
from app.Autenticacion import auth_bp
from app.Autenticacion.Servicios import get_user_by_email, add_user, user_exists
from werkzeug.security import check_password_hash, generate_password_hash
from app.Autenticacion.Formularios import LoginForm, RegisterForm
from app.Modelos.Usuario import Usuario
from app.Modelos.Profesional import Profesional
from flask_login import login_user, logout_user, current_user
from app import login_manager, db

# Ruta para la página de inicio de sesión
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('Analisis.nuevo_analisis'))

    form = LoginForm()
    if form.validate_on_submit():
        user = get_user_by_email(form.correo_electronico.data)
        if user and check_password_hash(user.contrasena, form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            flash('Inicio de sesión exitoso.', 'success')
            return redirect(next_page or url_for('Analisis.nuevo_analisis'))
        else:
            flash('Inicio de sesión fallido. Por favor, verifica tu correo electrónico y contraseña.', 'danger')
    return render_template('Autenticacion/login.html', title='Iniciar Sesión', form=form)

@auth_bp.route('/logout')
def logout():
    logout_user()
    flash('Has cerrado sesión exitosamente.', 'info')
    return redirect(url_for('Autenticacion.login'))

# Ruta para la página de registro
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if user_exists(form.correo_electronico.data):
            flash('El correo electrónico ya está en uso', 'danger')
            return redirect(url_for('Autenticacion.register'))
        hashed_password = generate_password_hash(form.password.data)

        # Crear nuevo usuario
        new_user = Usuario(
            id_profesional=form.id_profesional.data,
            correo_electronico=form.correo_electronico.data,
            contrasena=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()

        # Crear nuevo profesional asociado al usuario
        new_profesional = Profesional(
            rut_profesional=form.id_profesional.data,
            Nombre=form.nombre.data,
            App_Paterno=form.app_paterno.data,
            App_Materno=form.app_materno.data,
            Especialidad=form.especialidad.data,
            Edad=form.edad.data
        )
        db.session.add(new_profesional)
        db.session.commit()

        flash('¡Registro exitoso! Ya puedes iniciar sesión.', 'success')
        return redirect(url_for('Autenticacion.login'))
    return render_template('Autenticacion/register.html', form=form)

@login_manager.user_loader
def load_user(user_id):
    return Usuario.query.get(int(user_id))
