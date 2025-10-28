# Hoja de Ruta: Módulo de Explicabilidad Clínica (Regla ABCD)

El siguiente escrito sintetiza la hoja de ruta detallada para la implementación de la explicabilidad clínica, enfocándose en la Ingeniería de la Aceptación y el Mapeo Conceptual de la Regla ABCD.

La implementación del Módulo de Explicabilidad Clínica se aborda bajo el principio de Ingeniería de la Aceptación, cuyo objetivo es traducir la evidencia visual de la IA (Grad-CAM) al marco descriptivo de la Regla ABCD para generar confianza en el profesional. Este proceso se divide en tres fases secuenciales:

---

## 1. Preparación de la Evidencia Cuantificable (Segmentación)

Esta fase inicial es crítica para aislar la lesión y generar inputs precisos.

- Generación y Normalización de Grad-CAM: Obtener la matriz de intensidades (0 a 1) que indica el foco del modelo.
- Aislamiento de la Lesión (Máscara): Mediante procesamiento de imagen (ej., Otsu o K-means), se genera una máscara que delimita la lesión.
- Objetivo: Asegurar que el análisis de Bordes y Asimetría no se contamine con ruido de fondo.

---

## 2. Cuantificación de los Criterios ABCD

La fase de cuantificación traduce la matriz de Grad-CAM a puntuaciones numéricas (\(S_A, S_B, S_C\)) que representan la fuerza de la evidencia. El backend realiza un análisis geométrico detallado:

- Asimetría (\(S_{\text{Asimetría}}\)): Se calcula la desviación del Centro de Masa del Mapa de Calor respecto al centroide de la lesión.
- Bordes (\(S_{\text{Bordes}}\)): Se mide la Intensidad Promedio del Grad-CAM dentro de un "anillo" alrededor del borde de la lesión. Una alta intensidad indica que la CNN se fijó en la irregularidad periférica.
- Color (\(S_{\text{Color}}\)): Se cuantifica la dispersión o varianza del Grad-CAM sobre la superficie interna de la lesión, indicando variación cromática.
- Diámetro (\(S_{\text{Diámetro}}\)): Se utiliza el dato opcional de entrada para establecer una alerta binaria si el diámetro supera los 6 mm.

---

## 3. Construcción del Diccionario de Mapeo (Justificación Final)

Finalmente, se integra la lógica de decisión para generar la Justificación Clínica:

- Umbrales de Activación (\(U_{\text{activación}}\)) para cada criterio (\(S_A, S_B, S_C\)).
- El backend itera sobre las puntuaciones para construir el mensaje.
- Si el **Contador de Sospecha** (criterios que superan el umbral) es de dos o más, y la predicción es **Maligna**, el sistema genera el texto de **Justificación Clínica**.
- Este texto correlaciona las puntuaciones altas con las frases descriptivas de la Regla ABCD (ej., "Fuerte evidencia en Bordes Irregulares"), proporcionando la evidencia auditada al profesional.

---

## Resultados Esperados

- Integración transparente entre la evidencia visual (Grad-CAM) y el lenguaje clínico ABCD.
- Mensajes claros, trazables y auditables que apoyen la toma de decisiones del profesional.
- Base para futuras mejoras (calibración de umbrales, validación clínica y ajuste por especialidad).