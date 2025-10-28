from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.Modelos.Historial import Historial
from app.Modelos.Profesional import Profesional
import base64
import json
from app.Historial.ExplicabilidadServicios import niveles_abcd

historial_bp = Blueprint('historial', __name__, template_folder='templates')

@historial_bp.route('/historial')
@login_required
def historial():
    if current_user.is_authenticated and hasattr(current_user, 'id_profesional') and current_user.id_profesional:
        rut_profesional = current_user.id_profesional
        registros_historial = Historial.query.filter_by(rut_profesional=rut_profesional).order_by(Historial.fecha.desc(), Historial.hora.desc()).all()

        # Construir miniaturas (preferir imagen reescalada; usar mapa de calor si no está)
        for r in registros_historial:
            try:
                if getattr(r, 'archivo_img', None):
                    r.thumb_img = 'data:image/jpeg;base64,' + base64.b64encode(r.archivo_img).decode('utf-8')
                elif getattr(r, 'mapa_calor', None):
                    r.thumb_img = 'data:image/jpeg;base64,' + base64.b64encode(r.mapa_calor).decode('utf-8')
                else:
                    r.thumb_img = None
            except Exception as e:
                print(f"Error generando miniatura para historial {r.id_consulta}: {e}")
                r.thumb_img = None
        
        print(f"Rut Profesional: {rut_profesional}")
        print(f"Total registros historial: {len(registros_historial)}")
        return render_template('historial/historial.html', historiales=registros_historial)
    else:
        # Handle cases where user is not authenticated or id_profesional is not available
        return render_template('historial/historial.html', historiales=[])

# Nueva ruta para ver detalle de un historial específico
@historial_bp.route('/historial/<int:id_consulta>')
@login_required
def detalle_historial(id_consulta):
    registro = Historial.query.get_or_404(id_consulta)

    imagen_reescalada = None
    if registro.archivo_img:
        try:
            imagen_reescalada = 'data:image/jpeg;base64,' + base64.b64encode(registro.archivo_img).decode('utf-8')
        except Exception as e:
            print(f"Error al decodificar imagen de historial {id_consulta}: {e}")

    mapa_calor_img = None
    if registro.mapa_calor:
        try:
            mapa_calor_img = 'data:image/jpeg;base64,' + base64.b64encode(registro.mapa_calor).decode('utf-8')
        except Exception as e:
            print(f"Error al decodificar mapa de calor {id_consulta}: {e}")

    # Parsear explicación JSON si existe para mostrar ABCD
    abcd_mensajes = None
    abcd_justificacion = None
    abcd_scores = None
    abcd_niveles = None
    alerta_diametro = None
    try:
        if registro.explicacion:
            exp = json.loads(registro.explicacion)
            abcd_mensajes = exp.get('mensajes')
            abcd_justificacion = exp.get('justificacion')
            abcd_scores = exp.get('scores')
            alerta_diametro = exp.get('alerta_diametro')
            if abcd_scores:
                abcd_niveles = niveles_abcd(abcd_scores, {"asimetria": 0.65, "bordes": 0.45, "color": 0.50})
    except Exception as e:
        print(f"Error al parsear explicación del historial {id_consulta}: {e}")

    return render_template(
        'historial/detalle.html',
        historial=registro,
        imagen_reescalada=imagen_reescalada,
        mapa_calor_img=mapa_calor_img,
        abcd_mensajes=abcd_mensajes,
        abcd_justificacion=abcd_justificacion,
        abcd_scores=abcd_scores,
        abcd_niveles=abcd_niveles,
        alerta_diametro=alerta_diametro,
    )