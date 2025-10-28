from flask import render_template, request, flash, redirect, url_for, Blueprint
import os
from .Servicios import cargar_y_reescalar_imagen, obtener_modelos_h5, generar_prediccion, generar_grad_cam
import io
from flask_login import login_required
import base64
from datetime import datetime
from flask_login import current_user
from app import db
from app.Modelos.Historial import Historial
from app.Historial.ExplicabilidadServicios import analizar_explicabilidad_abcd
import json
from PIL import Image
import numpy as np

analisis_bp = Blueprint('Analisis', __name__, url_prefix='/analisis')

@analisis_bp.route('/nuevo', methods=['GET', 'POST'])
@login_required
def nuevo_analisis():
    modelos_disponibles = obtener_modelos_h5(os.path.dirname(__file__))
    if request.method == 'POST':
        if 'imagen' not in request.files:
            flash('No se seleccionó ningún archivo de imagen', 'danger')
            return redirect(request.url)
        
        imagen_cargada = request.files['imagen']
        
        if imagen_cargada.filename == '':
            flash('No se seleccionó ningún archivo de imagen', 'danger')
            return redirect(request.url)
        
        if imagen_cargada:
            # Leer la imagen como bytes
            imagen_bytes = io.BytesIO(imagen_cargada.read())
            
            # Cargar y reescalar la imagen
            imagen_reescalada_pil, imagen_np = cargar_y_reescalar_imagen(imagen_bytes)
            
            if imagen_reescalada_pil:
                # Convertir la imagen reescalada a bytes para Base64
                buffered = io.BytesIO()
                imagen_reescalada_pil.save(buffered, format="JPEG")
                reescalada_imagen_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Reiniciar el puntero de imagen_bytes para el original
                imagen_bytes.seek(0)
                original_imagen_base64 = base64.b64encode(imagen_bytes.getvalue()).decode('utf-8')

                nombre_modelo_seleccionado = request.form['modeloIA']
                prediccion = None
                if nombre_modelo_seleccionado and imagen_np is not None:
                    try:
                        prediccion_principal_raw, prediccion_secundaria_raw = generar_prediccion(imagen_np, nombre_modelo_seleccionado, os.path.dirname(__file__))

                        # Lógica para la predicción principal
                        diagnostico_principal_label = "" # Inicializar la etiqueta
                        if len(prediccion_principal_raw) > 0 and len(prediccion_principal_raw[0]) > 1:
                            if prediccion_principal_raw[0][1] > prediccion_principal_raw[0][0] and prediccion_principal_raw[0][1] > 0.4:
                                prediccion_principal = float(prediccion_principal_raw[0][1]) # Benigna
                                diagnostico_principal_label = "Maligna"
                            else:
                                prediccion_principal = float(prediccion_principal_raw[0][0]) # Maligna
                                diagnostico_principal_label = "Benigna"


                        # Lógica para la predicción secundaria: valor mayor de todos los valores
                        diagnostico_secundario_label = "" # Inicializar la etiqueta
                        if prediccion_secundaria_raw is not None and len(prediccion_secundaria_raw) > 0 and len(prediccion_secundaria_raw[0]) > 0:
                            # Asumiendo que prediccion_secundaria_raw es una lista de listas
                            max_prob = max(prediccion_secundaria_raw[0])
                            prediccion_secundaria = float(max_prob)
                            # Aquí necesitas la lista de etiquetas en el orden correcto
                            # La etiqueta secundaria será 'Maligna' si el primer valor de prediccion_principal_raw es mayor, 'Benigna' en caso contrario.
                            # Asumiendo que prediccion_principal_raw[0] contiene las probabilidades para [Benigna, Maligna]
                            if prediccion_principal_raw[0][0] > prediccion_principal_raw[0][1]:
                                diagnostico_secundario_label = "Benigna"
                            else:
                                diagnostico_secundario_label = "Maligna"
                        else:
                            prediccion_secundaria = 0.0 # Default value in case of error
                            diagnostico_secundario_label = "Desconocido" # Etiqueta por defecto

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        prediccion_principal = 0.0 # Default value in case of error
                        diagnostico_principal_label = "Error en predicción"
                        prediccion_secundaria = 0.0 # Default value in case of error
                        diagnostico_secundario_label = "Error en predicción"

                # Generar Grad-CAM para el modelo seleccionado (overlay para mostrar)
                grad_cam_pil = None
                try:
                    grad_cam_pil = generar_grad_cam(imagen_np, nombre_modelo_seleccionado, os.path.dirname(__file__))
                    buf_heat = io.BytesIO()
                    grad_cam_pil.save(buf_heat, format="JPEG")
                    grad_cam_base64 = base64.b64encode(buf_heat.getvalue()).decode('utf-8')
                except Exception as e:
                    print(f"Error al generar Grad-CAM: {e}")
                    buf_heat = io.BytesIO()
                    grad_cam_base64 = None

                # Ejecutar pipeline de explicabilidad ABCD (usa heatmap interno, no otra predicción)
                explicabilidad_res = {}
                try:
                    explicabilidad_res = analizar_explicabilidad_abcd(
                        imagen_np=imagen_np,
                        nombre_modelo=nombre_modelo_seleccionado,
                        base_path=os.path.dirname(__file__),
                        umbrales={"asimetria": 0.65, "bordes": 0.45, "color": 0.50},
                        d_mm=None,
                        target_layer_name=None,
                        es_maligna=(diagnostico_principal_label == "Maligna")
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    explicabilidad_res = {}

                # Convertir máscara de lesión a imagen base64 para previsualización
                lesion_mask_base64 = None
                try:
                    mask = explicabilidad_res.get('mask') if explicabilidad_res else None
                    if mask is not None:
                        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
                        buf_mask = io.BytesIO()
                        mask_img.save(buf_mask, format="PNG")
                        lesion_mask_base64 = base64.b64encode(buf_mask.getvalue()).decode('utf-8')
                except Exception as e:
                    print(f"Error al convertir máscara de lesión: {e}")
                    lesion_mask_base64 = None

                flash('Imagen cargada y reescalada exitosamente', 'success')

                edad_str = request.form.get('edad', '0')
                try:
                    edad = int(edad_str)
                except ValueError:
                    edad = 0 # Default to 0 or handle error as appropriate
                sexo = request.form.get('sexo', 'No especificado')
                lugar_lesion = request.form.get('lugar_lesion', 'No especificado')
                fecha_hora_prediccion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Guardar en la base de datos (explicacion como JSON de evidencia Grad-CAM)
                explicacion_json = None
                try:
                    if explicabilidad_res:
                        explicacion_json = json.dumps({
                            "justificacion": explicabilidad_res.get("justificacion"),
                            "mensajes": explicabilidad_res.get("mensajes"),
                            "scores": explicabilidad_res.get("scores"),
                            "cuantificacion": explicabilidad_res.get("cuantificacion"),
                            "feature_scores": explicabilidad_res.get("feature_scores"),
                            "alerta_diametro": explicabilidad_res.get("alerta_diametro"),
                            "es_maligna": explicabilidad_res.get("es_maligna"),
                        }, ensure_ascii=False)
                except Exception:
                    explicacion_json = None

                nueva_consulta = Historial(
                    rut_profesional=current_user.id_profesional, # Asume que current_user.id_profesional contiene el RUT
                    fecha=datetime.now().date(),
                    hora=datetime.now().time(),
                    archivo_img=buffered.getvalue(), # La imagen reescalada en bytes
                    diagnostico=diagnostico_principal_label, # Guardar la etiqueta
                    diagnostico_2=diagnostico_secundario_label, # Guardar la etiqueta
                    edad_paciente=edad, # Asegúrate de que edad sea un entero
                    sexo=sexo,
                    lugar_lesion=lugar_lesion,
                    mapa_calor=buf_heat.getvalue() if 'buf_heat' in locals() else None,
                    explicacion=explicacion_json or f"Grad-CAM generado con {nombre_modelo_seleccionado}"
                )
                db.session.add(nueva_consulta)
                db.session.commit()
                flash('Diagnóstico guardado exitosamente en la base de datos.', 'success')

                modelos_disponibles = obtener_modelos_h5(os.path.dirname(__file__))
                return render_template('Analisis/diagnostico.html',
                                       original_imagen='data:image/jpeg;base64,' + original_imagen_base64,
                                       edad=edad,
                                       sexo=sexo,
                                       lugar_lesion=lugar_lesion,
                                       fecha_hora_prediccion=fecha_hora_prediccion,
                                       prediccion_principal=prediccion_principal, # Mantener para mostrar el valor numérico si es necesario
                                       prediccion_secundaria=prediccion_secundaria, # Mantener para mostrar el valor numérico si es necesario
                                       diagnostico_principal_label=diagnostico_principal_label, # Pasar la etiqueta al template
                                       diagnostico_secundario_label=diagnostico_secundario_label, # Pasar la etiqueta al template
                                       mapa_calor_img=('data:image/jpeg;base64,' + grad_cam_base64) if grad_cam_base64 else None,
                                       lesion_mask_img=('data:image/png;base64,' + lesion_mask_base64) if lesion_mask_base64 else None,
                                       abcd_mensajes=explicabilidad_res.get('mensajes') if explicabilidad_res else None,
                                       abcd_justificacion=explicabilidad_res.get('justificacion') if explicabilidad_res else None,
                                       abcd_scores=explicabilidad_res.get('scores') if explicabilidad_res else None,
                                       abcd_cuantificacion=explicabilidad_res.get('cuantificacion') if explicabilidad_res else None,
                                       abcd_feature_scores=explicabilidad_res.get('feature_scores') if explicabilidad_res else None,
                                       abcd_umbrales=explicabilidad_res.get('umbrales') if explicabilidad_res else None,
                                       alerta_diametro=explicabilidad_res.get('alerta_diametro') if explicabilidad_res else None,
                                       abcd_niveles=explicabilidad_res.get('niveles') if explicabilidad_res else None,
                                       diametro_equivalente_px=explicabilidad_res.get('diametro_equivalente_px') if explicabilidad_res else None,
                                       modelos=modelos_disponibles)
            else:
                flash('Error al procesar la imagen', 'danger')
                return redirect(request.url)
        else:
            flash('No se seleccionó ningún archivo de imagen', 'danger')
            return redirect(request.url)
    return render_template('Analisis/diagnostico.html', modelos=modelos_disponibles)