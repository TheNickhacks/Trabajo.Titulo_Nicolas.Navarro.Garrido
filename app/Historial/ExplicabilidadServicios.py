import numpy as np
from typing import Dict, Optional, Tuple
import os

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    h = np.maximum(heatmap, 0)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    return h


def generar_mapa_calor_normalizado(
    imagen_np: np.ndarray,
    nombre_modelo: str,
    base_path: str,
    target_layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Genera el mapa de calor Grad-CAM normalizado en [0,1] a partir de la imagen y el modelo.
    Devuelve única y exclusivamente la matriz del mapa de calor (sin overlay),
    del mismo tamaño espacial que la imagen de entrada.
    """
    if tf is None or keras is None:
        raise RuntimeError("TensorFlow/Keras no disponible en el entorno actual")

    # Asegurar RGB y normalización [0,1]
    if imagen_np.ndim == 2:
        imagen_np = np.stack([imagen_np] * 3, axis=-1)
    elif imagen_np.shape[-1] == 4:
        imagen_np = imagen_np[..., :3]
    img = imagen_np.astype(np.float32)
    img = img / 255.0 if img.max() > 1.0 else img

    # Cargar el modelo desde la carpeta correcta
    model_dir = os.path.join(base_path, 'modelos_ia')
    candidate_paths = [
        os.path.join(model_dir, nombre_modelo),
        os.path.join(base_path, nombre_modelo),
    ]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError(f"Modelo no encontrado: {nombre_modelo} en {model_dir} o {base_path}")

    # Cargar modelo sin compilar para evitar errores con funciones de pérdida personalizadas
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Advertencia al cargar modelo: {e}")
        model = keras.models.load_model(model_path)

    # Inferencia inicial para obtener clase objetivo
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)
    preds = model(input_tensor)
    class_index = int(tf.argmax(preds[0]).numpy())

    # Reconstruir grafo funcional para obtener salidas intermedias sin depender de model.output
    inputs = tf.keras.Input(shape=img.shape)
    x = inputs
    outputs_by_name = {}
    last_conv_name = None
    for layer in model.layers:
        try:
            x = layer(x)
            outputs_by_name[layer.name] = x
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_name = layer.name
        except Exception:
            # Algunas capas pueden requerir argumentos; si fallan, continuar
            pass

    # Determinar capa objetivo
    if target_layer_name:
        if target_layer_name not in outputs_by_name:
            raise ValueError(f"La capa objetivo '{target_layer_name}' no está disponible en el grafo reconstruido")
        conv_output_tensor = outputs_by_name[target_layer_name]
    else:
        if last_conv_name is None or last_conv_name not in outputs_by_name:
            raise ValueError("No se encontró ninguna capa Conv2D en el modelo para Grad-CAM")
        conv_output_tensor = outputs_by_name[last_conv_name]

    predictions_tensor = x

    grad_model = keras.models.Model(inputs=inputs, outputs=[conv_output_tensor, predictions_tensor])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)

    # Pooling de gradientes canal-wise
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1).numpy()

    # Normalizar y redimensionar al tamaño original
    heatmap = _normalize_heatmap(heatmap)
    heatmap_resized = tf.image.resize(
        tf.expand_dims(tf.expand_dims(heatmap, axis=-1), axis=0),
        [img.shape[0], img.shape[1]],
    )[0, ..., 0].numpy()

    return heatmap_resized


def mascara_lesion_por_percentil(heatmap: np.ndarray, percentil: float = 75.0) -> np.ndarray:
    """Genera máscara binaria de la lesión usando un umbral por percentil del mapa de calor."""
    thr = np.percentile(heatmap, percentil)
    return (heatmap >= thr).astype(np.uint8)


def _vecinos_8(mask: np.ndarray) -> np.ndarray:
    """Devuelve una dilatación de 1 px usando vecindario-8 sin librerías externas."""
    h, w = mask.shape
    dil = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            y0 = max(0, -dy)
            y1 = h - max(0, dy)
            x0 = max(0, -dx)
            x1 = w - max(0, dx)
            dil[y0:y1, x0:x1] |= mask[y0+dy:y1+dy, x0+dx:x1+dx]
    return dil


def anillo_borde(mask: np.ndarray) -> np.ndarray:
    """Obtiene un anillo del borde: perímetro + vecinos externos inmediatos."""
    h, w = mask.shape
    # Perímetro: píxeles de máscara con al menos un vecino en 0
    perimetro = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 1:
                y0, y1 = max(0, y-1), min(h, y+2)
                x0, x1 = max(0, x-1), min(w, x+2)
                vec = mask[y0:y1, x0:x1]
                if np.any(vec == 0):
                    perimetro[y, x] = 1
    # Vecinos externos inmediatos
    dil = _vecinos_8(mask)
    externos = (dil & (~mask.astype(bool))).astype(np.uint8)
    return ((perimetro | externos) > 0).astype(np.uint8)


def centroides(mask: np.ndarray, heatmap: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Centroides (x,y): de la máscara y centro de masa del heatmap."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (heatmap.shape[1]/2.0, heatmap.shape[0]/2.0), (heatmap.shape[1]/2.0, heatmap.shape[0]/2.0)
    cx_mask = float(xs.mean())
    cy_mask = float(ys.mean())

    y_idx, x_idx = np.indices(heatmap.shape)
    w = heatmap + 1e-8
    cx_h = float((x_idx * w).sum() / w.sum())
    cy_h = float((y_idx * w).sum() / w.sum())
    return (cx_mask, cy_mask), (cx_h, cy_h)


def score_asimetria(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Puntúa asimetría: distancia normalizada entre centroides (0-1)."""
    (cx_m, cy_m), (cx_h, cy_h) = centroides(mask, heatmap)
    # Normalización por diagonal de bounding box de la máscara (fallback: imagen)
    ys, xs = np.where(mask > 0)
    if len(xs) >= 2:
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
    else:
        h, w = mask.shape
    diag = np.sqrt(w*w + h*h) + 1e-8
    dist = np.sqrt((cx_m - cx_h)**2 + (cy_m - cy_h)**2)
    return float(np.clip(dist / diag, 0.0, 1.0))


def score_bordes(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Puntúa bordes: intensidad media del mapa de calor en el anillo del borde (0-1)."""
    ring = anillo_borde(mask)
    if ring.sum() == 0:
        return 0.0
    return float(np.clip(heatmap[ring > 0].mean(), 0.0, 1.0))


def score_color(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Puntúa color: varianza interna normalizada por el valor máximo teórico (0.25)."""
    vals = heatmap[mask > 0]
    if len(vals) < 2:
        return 0.0
    var = float(vals.var())
    return float(np.clip(var / 0.25, 0.0, 1.0))


def alerta_diametro(d_mm: Optional[float]) -> bool:
    """Alerta binaria por diámetro clínico (True si > 6mm)."""
    if d_mm is None:
        return False
    return bool(d_mm > 6.0)

# Nuevo: diámetro equivalente en píxeles (DEP) desde el área de la máscara
def diametro_equivalente_pixeles(mask: np.ndarray) -> float:
    area_lesion = float(np.sum(mask > 0))
    if area_lesion <= 0.0:
        return 0.0
    return float(2.0 * np.sqrt(area_lesion / np.pi))

# Nuevo: niveles concisos para criterios ABCD
def niveles_abcd(scores: Dict[str, float], umbrales: Dict[str, float]) -> Dict[str, str]:
    UA = float(umbrales.get('asimetria', 0.65))
    UB = float(umbrales.get('bordes', 0.45))
    UC = float(umbrales.get('color', 0.50))
    sA = float(scores.get('asimetria', 0.0))
    sB = float(scores.get('bordes', 0.0))
    sC = float(scores.get('color', 0.0))

    def nivel(s: float, u: float) -> str:
        if s >= u:
            return 'alta'
        elif s >= 0.8 * u:
            return 'moderada'
        else:
            return 'baja'
    return {
        'asimetria': nivel(sA, UA),
        'bordes': nivel(sB, UB),
        'color': nivel(sC, UC),
    }


def construir_justificacion(
    scores: Dict[str, float],
    umbrales: Dict[str, float],
    es_maligna: Optional[bool] = None,
    d_mm: Optional[float] = None,
) -> str:
    """
    Genera un resumen clínico conciso basado en la activación del Grad-CAM
    (Regla ABCD). Resume cuántos criterios superaron sus umbrales técnicos.
    """
    # Puntuaciones
    sA = float(scores.get('asimetria', 0.0))
    sB = float(scores.get('bordes', 0.0))
    sC = float(scores.get('color', 0.0))
    sD = float(scores.get('diametro', 0.0))

    # Umbrales (no se muestran en UI de justificación, solo se usan internamente)
    UA = float(umbrales.get('asimetria', 0.65))
    UB = float(umbrales.get('bordes', 0.45))
    UC = float(umbrales.get('color', 0.50))
    UD = float(umbrales.get('diametro_px', 0.25))

    # Criterios activados estrictamente (>= umbral)
    criterios = []
    if sA >= UA:
        criterios.append('Asimetría')
    if sB >= UB:
        criterios.append('Bordes Irregulares')
    if sC >= UC:
        criterios.append('Variación de Color')
    if sD >= UD:
        criterios.append('Diámetro Relativo Elevado')

    num = len(criterios)

    # Nivel de evidencia en función del conteo
    if num >= 3:
        nivel = 'alta'
    elif num == 2:
        nivel = 'moderada'
    elif num == 1:
        nivel = 'leve'
    else:
        nivel = 'baja'

    if num > 0:
        if num == 1:
            lista = criterios[0]
        elif num == 2:
            lista = f"{criterios[0]} y {criterios[1]}"
        else:
            lista = ", ".join(criterios[:-1]) + f" y {criterios[-1]}"
        resumen = f"Evidencia {nivel} según ABCD: {num} criterio(s) superan umbral ({lista})."
    else:
        # Criterio más cercano al umbral (máximo ratio)
        ratios = {
            'Asimetría': sA / UA if UA > 0 else 0,
            'Bordes Irregulares': sB / UB if UB > 0 else 0,
            'Variación de Color': sC / UC if UC > 0 else 0,
            'Diámetro Relativo Elevado': sD / UD if UD > 0 else 0,
        }
        prox = max(ratios, key=ratios.get)
        resumen = f"Evidencia baja según ABCD: ningún criterio supera umbral; el más cercano es {prox}."

    # Información clínica opcional por diámetro en mm (si aplica)
    extra = ""
    if d_mm and d_mm > 6.0:
        extra = f" Adicionalmente, el diámetro estimado ({d_mm:.1f} mm) supera los 6 mm."

    return resumen + extra


def score_diametro_px(dep_px: float, image_shape: Tuple[int, int]) -> float:
    """Score de diámetro relativo a la imagen: DEP dividido por lado menor.
    Devuelve valor normalizado en [0,1]."""
    h, w = int(image_shape[0]), int(image_shape[1])
    base = float(min(h, w))
    if base <= 0.0:
        return 0.0
    return float(np.clip(dep_px / base, 0.0, 1.0))


def mensajes_tipo_abcd(
    scores: Dict[str, float],
    umbrales: Dict[str, float],
    cuant: Dict[str, float],
    feats: Dict[str, float],
    d_mm: Optional[float],
    dep_px: Optional[float] = None,
) -> Dict[str, str]:
    UA = float(umbrales.get('asimetria', 0.65))
    UB = float(umbrales.get('bordes', 0.45))
    UC = float(umbrales.get('color', 0.50))
    UD_rel = float(umbrales.get('diametro_px', 0.25))  # umbral relativo al lado menor
    sA = float(scores.get('asimetria', 0.0))
    sB = float(scores.get('bordes', 0.0))
    sC = float(scores.get('color', 0.0))
    sD = float(scores.get('diametro', 0.0))

    def etiqueta_sospecha(s: float, u: float) -> str:
        if s >= u:
            return 'Alta Sospecha'
        elif s >= 0.8 * u:
            return 'Sospecha Moderada'
        elif s >= 0.5 * u:
            return 'Baja Sospecha'
        else:
            return 'Ausencia de Sospecha'

    eA = etiqueta_sospecha(sA, UA)
    eB = etiqueta_sospecha(sB, UB)
    eC = etiqueta_sospecha(sC, UC)
    eD = etiqueta_sospecha(sD, UD_rel)

    # A - Asimetría (solo score entre paréntesis)
    sim_desc = 'esencialmente simétrica' if sA < 0.5 * UA else 'moderadamente desbalanceada'
    msgA = (
        f"A - Asimetría (Score={sA:.2f}): La activación del Grad-CAM respecto al centro de la lesión fue {sim_desc}, "
        f"aportando {'evidencia significativa' if sA >= UA else 'evidencia limitada'} de irregularidad estructural."
    )

    # B - Bordes (solo score entre paréntesis)
    borde_act = 'baja' if sB < 0.5 * UB else ('moderada' if sB < 0.8 * UB else 'alta')
    msgB = (
        f"B - Bordes (Score={sB:.2f}): El modelo enfatizó el contorno periférico de la lesión, con {borde_act} activación en el anillo de borde. "
        f"{'Sugiere irregularidad' if sB >= UB else 'No se considera fuertemente irregular'} bajo el análisis de la IA."
    )

    # C - Color (solo score entre paréntesis)
    color_act = 'mínima' if sC < 0.5 * UC else ('moderada' if sC < 0.8 * UC else 'alta')
    msgC = (
        f"C - Color (Score={sC:.2f}): Se observó una variación del foco interno del Grad-CAM, compatible con diversidad cromática {color_act}. "
        f"Este criterio {'contribuye' if sC >= UC else 'tiene contribución limitada'} a la puntuación de riesgo (C x 0.50 = {feats['C']:.2f})."
    )

    # D - Diámetro (solo score entre paréntesis)
    dep_text = f"{(dep_px or 0.0):.1f}" if dep_px is not None else "No disponible"
    msgD = (
        f"D - Diámetro (Score={sD:.2f}): El tamaño de la lesión, medido como DEP, es {dep_text} píxeles. "
        f"Se reporta como métrica de referencia debido a la falta de una escala física fiable."
    )

    return {'A': msgA, 'B': msgB, 'C': msgC, 'D': msgD}


def analizar_explicabilidad_abcd(
    imagen_np: np.ndarray,
    nombre_modelo: str,
    base_path: str,
    umbrales: Optional[Dict[str, float]] = None,
    d_mm: Optional[float] = None,
    target_layer_name: Optional[str] = None,
    es_maligna: Optional[bool] = None,
) -> Dict[str, object]:
    """
    Pipeline completo: genera heatmap, máscara y puntúa ABCD.
    Devuelve dict con scores, máscara, heatmap normalizado y texto de justificación.
    """
    umbrales = umbrales or {"asimetria": 0.65, "bordes": 0.45, "color": 0.50, "diametro_px": 0.25}
    heatmap = generar_mapa_calor_normalizado(
        imagen_np, nombre_modelo, base_path, target_layer_name
    )
    # Usar máscara refinada (Otsu ∩ Grad-CAM adaptativo)
    mask = mascara_lesion_refinada(imagen_np, heatmap)

    s_asim = score_asimetria(heatmap, mask)
    s_bord = score_bordes(heatmap, mask)
    s_color = score_color(heatmap, mask)
    alerta = alerta_diametro(d_mm)

    # Nuevo: diámetro equivalente en píxeles (DEP) y su score relativo
    dep_px = diametro_equivalente_pixeles(mask)
    h, w = mask.shape
    s_diametro = score_diametro_px(dep_px, (h, w))

    scores = {"asimetria": s_asim, "bordes": s_bord, "color": s_color, "diametro": s_diametro}
    cuant, feats = cuantificar_y_pesar(scores, d_mm)
    mensajes = mensajes_tipo_abcd(scores, umbrales, cuant, feats, d_mm, dep_px)
    justificacion = construir_justificacion(scores, umbrales, es_maligna=es_maligna, d_mm=d_mm)

    return {
        "scores": scores,
        "cuantificacion": cuant,
        "feature_scores": feats,
        "mensajes": mensajes,
        "alerta_diametro": alerta,
        "mask": mask,
        "heatmap": heatmap,
        "justificacion": justificacion,
        "umbrales": umbrales,
        "es_maligna": bool(es_maligna) if es_maligna is not None else False,
        "diametro_equivalente_px": dep_px,
        "niveles": niveles_abcd(scores, umbrales),
    }


def _to_grayscale(imagen_np: np.ndarray) -> np.ndarray:
    """Convierte la imagen a escala de grises float32 en [0,1]."""
    if imagen_np.ndim == 2:
        gray = imagen_np.astype(np.float32)
    else:
        img = imagen_np.astype(np.float32)
        if img.shape[-1] == 4:
            img = img[..., :3]
        # Conversión estándar a gris (luma)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray.astype(np.float32)


def mascara_lesion_por_otsu(imagen_np: np.ndarray) -> np.ndarray:
    """
    Genera máscara binaria de la lesión usando Otsu sobre la imagen de entrada.
    Evita depender del Grad-CAM para segmentar la lesión.
    """
    gray = _to_grayscale(imagen_np)
    # Histograma en 256 bins (0..1)
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    total = gray.size
    sum_total = float(np.dot(hist, np.arange(256)))

    weight_bg = 0.0
    sum_bg = 0.0
    max_between = -1.0
    thr_idx = 127
    for i in range(256):
        weight_bg += float(hist[i])
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += float(i * hist[i])
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            thr_idx = i

    thr = thr_idx / 255.0
    mask = (gray >= thr).astype(np.uint8)
    area = mask.sum() / mask.size
    # Si el área es degenerada, intentar invertir para obtener foreground más razonable
    if area < 0.05 or area > 0.95:
        inv = (gray <= thr).astype(np.uint8)
        inv_area = inv.sum() / inv.size
        if abs(inv_area - 0.5) < abs(area - 0.5):
            mask = inv
    return mask


# Restaurar: cuantificar y pesar (A,B,C,D)
def cuantificar_y_pesar(scores: Dict[str, float], d_mm: Optional[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    sA = float(scores.get('asimetria', 0.0))
    sB = float(scores.get('bordes', 0.0))
    sC = float(scores.get('color', 0.0))
    A = float(np.clip(round(2.0 * sA), 0, 2))
    B = float(np.clip(round(8.0 * sB), 0, 8))
    C = float(np.clip(round(1.0 + 5.0 * sC), 1, 6))
    Dmm = float(d_mm or 0.0)
    feature_scores = {
        'A': A * 1.3,
        'B': B * 0.10,
        'C': C * 0.50,
        'D': Dmm * 0.05,
    }
    cuant = {'A': A, 'B': B, 'C': C, 'D_mm': Dmm}
    return cuant, feature_scores


def mascara_lesion_refinada(
    imagen_np: np.ndarray,
    heatmap: np.ndarray,
    base_percentil: float = 85.0,
    max_percentil: float = 97.0,
) -> np.ndarray:
    """Genera una máscara robusta combinando Otsu con Grad-CAM por percentil.
    - Parte de Otsu para delimitar la lesión.
    - Intersecta con una máscara de Grad-CAM por percentil (adaptativo) para evitar cubrir toda la imagen.
    - Ajusta el percentil si el área resultante es demasiado pequeña o demasiado grande.
    - Aplica una limpieza simple eliminando pixeles aislados.
    """
    try:
        h, w = heatmap.shape[:2]
    except Exception:
        raise ValueError("Heatmap inválido: se espera matriz 2D")

    otsu = mascara_lesion_por_otsu(imagen_np).astype(np.uint8)

    def intersect_with_percentil(p: float) -> np.ndarray:
        hm = mascara_lesion_por_percentil(heatmap, percentil=p).astype(np.uint8)
        inter = (otsu & hm).astype(np.uint8)
        return inter

    # Búsqueda adaptativa de percentil
    candidatos = [base_percentil, 80.0, 75.0, 70.0, 65.0, 60.0]
    mask = None
    for p in candidatos:
        inter = intersect_with_percentil(p)
        area_ratio = float(inter.sum()) / float(h * w)
        if 0.01 <= area_ratio <= 0.60:  # área razonable
            mask = inter
            break
    if mask is None:
        # Si sigue siendo muy pequeño, bajar más el umbral; si es muy grande, subirlo
        inter_low = intersect_with_percentil(50.0)
        inter_high = intersect_with_percentil(max_percentil)
        ar_low = float(inter_low.sum()) / float(h * w)
        ar_high = float(inter_high.sum()) / float(h * w)
        if ar_low >= 0.01 and ar_low <= 0.70:
            mask = inter_low
        else:
            mask = inter_high
            # Si aún es muy grande, tensar la máscara reduciendo borde de Otsu
            iter_shrink = 0
            while float(mask.sum()) / float(h * w) > 0.60 and iter_shrink < 3:
                ring = anillo_borde(otsu)
                otsu = (otsu & (~ring)).astype(np.uint8)
                mask = (otsu & intersect_with_percentil(max_percentil)).astype(np.uint8)
                iter_shrink += 1

    # Limpieza: eliminar pixeles aislados (sin vecinos-8)
    vecinos = _vecinos_8(mask)
    pruned = (mask & vecinos).astype(np.uint8)
    if pruned.sum() > 0:
        mask = pruned

    return mask.astype(np.uint8)