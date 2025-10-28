import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def obtener_modelos_h5(ruta_base):
    modelos = []
    ruta_modelos = os.path.join(ruta_base, 'modelos_ia')
    if os.path.exists(ruta_modelos):
        for archivo in os.listdir(ruta_modelos):
            if archivo.endswith('.h5'):
                modelos.append(archivo)
    return modelos
import numpy as np
import os

def cargar_y_reescalar_imagen(imagen_bytes):
    try:
        # Abrir la imagen desde bytes
        imagen = Image.open(imagen_bytes)
        
        # Reescalar la imagen a 224x224 píxeles
        imagen_reescalada = imagen.resize((224, 224))
        
        # Asegurarse de que la imagen tenga 3 canales (RGB)
        if imagen_reescalada.mode != 'RGB':
            imagen_reescalada = imagen_reescalada.convert('RGB')

        # Convertir a arreglo numpy
        imagen_np = np.array(imagen_reescalada)
        
        return imagen_reescalada, imagen_np
    except Exception as e:
        print(f"Error al cargar o reescalar la imagen: {e}")
        return None, None

def generar_prediccion(imagen_np, nombre_modelo_primario, base_path):
    try:
        # Cargar el primer modelo
        ruta_modelo_primario = os.path.join(base_path, 'modelos_ia', nombre_modelo_primario)
        modelo_primario = load_model(ruta_modelo_primario)

        # Preprocesar la imagen para la predicción
        imagen_procesada = np.expand_dims(imagen_np, axis=0)
        imagen_procesada = imagen_procesada / 255.0  # Normalización de píxeles a [0, 1]

        # Realizar la primera predicción con el modelo primario
        prediccion_primaria = modelo_primario.predict(imagen_procesada)

        # Determinar el diagnóstico basado en la primera predicción
        # La lógica de "maligno" vs "benigno" debería ser consistente con la usada en diagnostico.html
        es_maligno = prediccion_primaria[0][1] > prediccion_primaria[0][0] and prediccion_primaria[0][1] > 0.4

        # Cargar el segundo modelo según el diagnóstico
        nombre_modelo_secundario = "Modelo_2.h5" if es_maligno else "Modelo_1_CapaPlana.h5"
        
        # Verificar si el modelo secundario existe
        ruta_modelo_secundario = os.path.join(base_path, 'modelos_ia', nombre_modelo_secundario)
        if not os.path.exists(ruta_modelo_secundario):
            return prediccion_primaria # Retorna la predicción primaria si el modelo secundario no existe

        modelo_secundario = load_model(ruta_modelo_secundario)

        # Realizar la predicción final con el modelo secundario
        prediccion_final = modelo_secundario.predict(imagen_procesada)
        return prediccion_primaria, prediccion_final
        
        # Preprocesar la imagen para la predicción
        # Para mostrar las predicciones en pantalla, el resultado 'resultado_prediccion' debe ser pasado al template renderizado.
        # Luego en el template (ej: diagnostico.html) se puede mostrar asi: {{ prediccion }}
        imagen_procesada = np.expand_dims(imagen_np, axis=0)
        imagen_procesada = imagen_procesada / 255.0  # Normalización de píxeles a [0, 1]

        prediccion = modelo.predict(imagen_procesada)
        # Devolvemos el array completo de predicción para considerar ambos valores
        return prediccion
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def generar_grad_cam(imagen_np, nombre_modelo, base_path, target_layer_name=None):
    try:
        ruta_modelo = os.path.join(base_path, 'modelos_ia', nombre_modelo)
        model = load_model(ruta_modelo)

        # Asegurar base en RGB para overlay
        base_img = imagen_np
        if base_img.ndim == 2:
            base_img = np.stack([base_img] * 3, axis=-1)
        elif base_img.ndim == 3 and base_img.shape[-1] == 1:
            base_img = np.concatenate([base_img] * 3, axis=-1)

        # Entrada normalizada para forward
        img_input = np.expand_dims(base_img, axis=0).astype(np.float32)
        img_input = img_input / 255.0

        # Predicción inicial para clase objetivo
        preds = model.predict(img_input)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        class_index = int(np.argmax(preds[0]))

        # Identificar última capa conv
        last_conv = None
        last_conv_name = None
        from tensorflow.keras.layers import Conv2D
        if target_layer_name:
            try:
                last_conv = model.get_layer(target_layer_name)
                last_conv_name = last_conv.name
            except Exception:
                last_conv = None
        if last_conv is None:
            for layer in reversed(model.layers):
                if isinstance(layer, Conv2D):
                    last_conv = layer
                    last_conv_name = layer.name
                    break
        if last_conv is None:
            print("No se encontró una capa convolucional para Grad-CAM")
            return Image.fromarray(base_img.astype(np.uint8))

        # Reconstruir grafo funcional sin depender de model.input/model.output
        inputs = tf.keras.Input(shape=img_input.shape[1:])
        x = inputs
        outputs_by_name = {}
        for layer in model.layers:
            try:
                x = layer(x)
                outputs_by_name[layer.name] = x
            except Exception:
                # Algunas capas pueden requerir argumentos; si fallan, continuar
                pass
        predictions_tensor = x
        if last_conv_name not in outputs_by_name:
            print("No se pudo obtener la salida de la capa conv seleccionada en el grafo reconstruido")
            return Image.fromarray(base_img.astype(np.uint8))

        grad_model = tf.keras.models.Model(inputs=inputs, outputs=[outputs_by_name[last_conv_name], predictions_tensor])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap_max = tf.reduce_max(heatmap)
        heatmap = heatmap / (heatmap_max + 1e-8)
        heatmap_np = heatmap.numpy()

        # Ajustar tamaño del heatmap al de la imagen base
        heatmap_img = Image.fromarray(np.uint8(heatmap_np * 255))
        heatmap_img = heatmap_img.resize((base_img.shape[1], base_img.shape[0]))
        heatmap_arr = np.array(heatmap_img).astype(np.float32) / 255.0  # Normalizar a [0,1]

        # Colormap tipo jet: azul -> cian -> verde -> amarillo -> rojo
        def jet_colormap(values):
            # values debe estar en [0,1]
            r = np.zeros_like(values)
            g = np.zeros_like(values)
            b = np.zeros_like(values)
            
            # Azul a cian (0.0 - 0.25)
            mask1 = (values >= 0.0) & (values < 0.25)
            b[mask1] = 1.0
            g[mask1] = values[mask1] * 4.0
            
            # Cian a verde (0.25 - 0.5)
            mask2 = (values >= 0.25) & (values < 0.5)
            g[mask2] = 1.0
            b[mask2] = 1.0 - (values[mask2] - 0.25) * 4.0
            
            # Verde a amarillo (0.5 - 0.75)
            mask3 = (values >= 0.5) & (values < 0.75)
            g[mask3] = 1.0
            r[mask3] = (values[mask3] - 0.5) * 4.0
            
            # Amarillo a rojo (0.75 - 1.0)
            mask4 = (values >= 0.75) & (values <= 1.0)
            r[mask4] = 1.0
            g[mask4] = 1.0 - (values[mask4] - 0.75) * 4.0
            
            return np.stack([r, g, b], axis=-1) * 255.0

        heatmap_color = jet_colormap(heatmap_arr)

        # Atenuar la imagen base para que el heatmap sea más prominente
        base_dimmed = base_img.astype(np.float32) * 0.4  # Atenuar imagen base
        alpha = 0.7  # Heatmap prominente pero no tanto como antes
        overlay = np.clip(alpha * heatmap_color + (1 - alpha) * base_dimmed, 0, 255).astype(np.uint8)

        overlay_pil = Image.fromarray(overlay)
        return overlay_pil
    except Exception as e:
        print(f"Error generando Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return Image.fromarray(imagen_np if imagen_np.ndim == 3 else np.stack([imagen_np]*3, axis=-1))