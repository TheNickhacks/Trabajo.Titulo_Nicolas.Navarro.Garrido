# Sistema de Diagnóstico de Lesiones Cutáneas con IA

Sistema web para el diagnóstico asistido por inteligencia artificial de lesiones cutáneas, desarrollado con Flask y TensorFlow. Utiliza modelos de redes neuronales convolucionales (CNN) para clasificar lesiones como benignas o malignas, con técnicas de explicabilidad (Grad-CAM y análisis ABCD) para proporcionar transparencia en las predicciones.

## 🎯 Características Principales

- **Autenticación de Usuarios**: Sistema de registro y login para profesionales de la salud
- **Análisis de Imágenes**: Carga y análisis de imágenes de lesiones cutáneas (224x224 px)
- **Diagnóstico Dual**: 
  - Predicción primaria: Clasificación inicial benigna/maligna
  - Predicción secundaria: Análisis especializado según diagnóstico primario
- **Explicabilidad (XAI)**:
  - Grad-CAM: Mapas de calor que muestran regiones de interés del modelo
  - Análisis ABCD: Evaluación de Asimetría, Bordes, Color y Diámetro
- **Historial Clínico**: Almacenamiento y consulta de diagnósticos previos
- **Base de Datos**: Soporte para PostgreSQL (producción) y SQLite (desarrollo)

## 🏗️ Arquitectura del Sistema

```
CNN_Tesis/
├── app/
│   ├── __init__.py              # Configuración de la aplicación Flask
│   ├── config.py                # Configuración de base de datos
│   ├── Autenticacion/           # Módulo de login/registro
│   │   ├── Formularios.py       # Formularios WTForms
│   │   ├── Rutas.py             # Endpoints de autenticación
│   │   └── Servicios.py         # Lógica de negocio
│   ├── Diagnosticos/            # Módulo de análisis de imágenes
│   │   ├── Rutas.py             # Endpoint de diagnóstico
│   │   ├── Servicios.py         # Carga de modelos, predicción, Grad-CAM
│   │   └── modelos_ia/          # Modelos .h5 de TensorFlow
│   │       ├── Modelo_1_CapaPlana.h5
│   │       ├── Modelo_2.h5
│   │       ├── Modelo_3.h5
│   │       └── Modelo_4.h5
│   ├── Historial/               # Módulo de historial clínico
│   │   ├── Rutas.py             # Consulta de historial
│   │   └── ExplicabilidadServicios.py  # Análisis ABCD y generación de heatmaps
│   ├── Modelos/                 # Modelos de base de datos (SQLAlchemy)
│   │   ├── Usuario.py
│   │   ├── Profesional.py
│   │   └── Historial.py
│   ├── Static/                  # CSS y JavaScript
│   └── Templates/               # Plantillas HTML (Jinja2)
├── instance/                    # Base de datos SQLite (desarrollo)
├── run.py                       # Punto de entrada de la aplicación
└── requerimientos.txt           # Dependencias del proyecto
```

## 📋 Requisitos del Sistema

- Python 3.8+
- PostgreSQL 12+ (producción) o SQLite (desarrollo)
- 4GB RAM mínimo (8GB recomendado para TensorFlow)
- Navegador web moderno

##  Instalación y Configuración

### 1. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requerimientos.txt
```

**Dependencias principales:**
- Flask 2.2.0
- Flask-SQLAlchemy 3.0.3
- TensorFlow 2.20.0
- Flask-Login 0.6.2
- Flask-WTF 1.0.1
- Pillow (procesamiento de imágenes)
- python-dotenv (variables de entorno)

### 4. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Desarrollo (SQLite)
DATABASE_URL=sqlite:///D:/CNN_Tesis/instance/app.db
SECRET_KEY=tu-clave-secreta-aqui

# Producción (PostgreSQL)
# DATABASE_URL=postgresql://usuario:contraseña@host:puerto/nombre_bd
```

### 5. Inicializar la base de datos

```bash
# El sistema crea las tablas automáticamente al iniciar
# Asegúrate de que la carpeta instance/ exista
mkdir instance
```

### 6. Ejecutar la aplicación

```bash
# Desarrollo
python run.py

# La aplicación estará disponible en http://127.0.0.1:5000
```

## 🧠 Modelos de IA

El sistema utiliza un enfoque de **diagnóstico en cascada**:

1. **Modelo Primario** (Modelo_3.h5 o Modelo_4.h5):
   - Clasificación inicial: Benigna vs Maligna
   - Umbral de decisión: >0.4 en clase maligna

2. **Modelos Secundarios**:
   - **Modelo_1_CapaPlana.h5**: Para lesiones benignas
   - **Modelo_2.h5**: Para lesiones malignas
   - Proporcionan clasificación especializada según resultado primario

### Grad-CAM (Gradient-weighted Class Activation Mapping)

- Genera mapas de calor visuales sobre la imagen original
- Identifica regiones que el modelo considera relevantes para el diagnóstico
- Utiliza gradientes de la última capa convolucional
- Colormap tipo jet: azul (baja activación) → rojo (alta activación)

### Análisis ABCD

Método dermatológico de evaluación cuantitativa:

- **A (Asimetría)**: Comparación entre mitades de la lesión
- **B (Bordes)**: Irregularidad del contorno
- **C (Color)**: Variación cromática
- **D (Diámetro)**: Tamaño equivalente de la lesión

Genera scores y mensajes interpretativos basados en umbrales configurables.

## 💾 Modelos de Base de Datos

### Usuario
- `id_user` (PK)
- `id_profesional` (FK → Profesional)
- `correo_electronico` (único)
- `contrasena` (hash)
- `tipo_usuario`
- `hora_ingreso`

### Profesional
- `rut_profesional` (PK)
- `Nombre`
- `App_Paterno`, `App_Materno`
- `Especialidad`
- `Edad`

### Historial
- `id_consulta` (PK)
- `rut_profesional` (FK → Profesional)
- `fecha`, `hora`
- `archivo_img` (BYTEA/LargeBinary)
- `diagnostico`, `diagnostico_2`
- `edad_paciente`, `sexo`, `lugar_lesion`
- `mapa_calor` (imagen Grad-CAM)
- `explicacion` (JSON con análisis ABCD)

## 🔒 Seguridad

- Contraseñas hasheadas con `werkzeug.security`
- Protección CSRF mediante Flask-WTF
- Sesiones seguras con `SECRET_KEY`
- Decorador `@login_required` para rutas protegidas
- Validación de formularios del lado del servidor

## 🎨 Interfaz de Usuario

- **Base Template**: Navegación consistente con Bootstrap
- **Login/Registro**: Formularios validados
- **Diagnóstico**: Interfaz de carga de imagen con previsualización
- **Resultados**: Visualización de predicciones, mapas de calor y análisis ABCD
- **Historial**: Tabla de diagnósticos previos con detalles expandibles
- **Manejo de Errores**: Páginas personalizadas 403, 404, 500

## 📊 Flujo de Diagnóstico

1. Usuario autenticado accede a `/analisis/nuevo`
2. Carga imagen de lesión (JPEG/PNG)
3. Sistema reescala a 224×224 px y convierte a RGB
4. Modelo primario realiza predicción inicial
5. Según resultado, se carga modelo secundario
6. Se genera Grad-CAM para explicabilidad visual
7. Se ejecuta análisis ABCD sobre la lesión segmentada
8. Resultados se almacenan en BD con metadata del paciente
9. Se muestra interfaz con diagnóstico, mapas de calor y justificación

## 🛠️ Comandos Útiles

```bash
# Activar entorno virtual
cd D:\CNN_Tesis # Windows

venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requerimientos.txt

# Ejecutar aplicación (desarrollo)
python run.py

# Ejecutar con gunicorn (producción)
gunicorn -w 4 -b 0.0.0.0:8000 run:app

# Ver logs en tiempo real
tail -f instance/app.log  # Linux/Mac
```

## 🐛 Solución de Problemas

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow==2.20.0
```

### Error: "unable to open database file"
```bash
# Asegurar que la carpeta instance/ existe
mkdir instance
# Usar ruta absoluta en DATABASE_URL
DATABASE_URL=sqlite:///D:/CNN_Tesis/instance/app.db
```

### Error: "TensorFlow/Keras no disponible"
- Verificar instalación: `pip show tensorflow`
- Reiniciar servidor después de instalar

### Problemas de memoria con TensorFlow
```python
# Añadir al inicio de run.py
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

## 📝 Notas de Desarrollo

- **Importación diferida de TensorFlow**: Los módulos importan TF dentro de funciones para evitar errores de inicio si no está instalado
- **Compatibilidad de BD**: El sistema soporta tanto PostgreSQL (producción) como SQLite (desarrollo local)
- **Versionado de modelos**: Los archivos .h5 deben estar en `app/Diagnosticos/modelos_ia/`
- **Imágenes en BD**: Se almacenan como BLOB/LargeBinary en formato JPEG


## 📄 Licencia

Este proyecto es parte de una investigación académica de tesis.


**Versión**: 1.0  
**Última actualización**: Noviembre 2025
