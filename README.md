# Proyecto Final de Conducción Autónoma

## Equipo 18 de la materia de Navegación Autónoma del MNA - Tecnológico de Monterrey.

## Índice de Contenidos
- [Descripción General](#descripción-general)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Controladores](#controladores)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Mundos de Simulación](#mundos-de-simulación)
- [Configuración del Entorno](#configuración-del-entorno)

## Descripción General
Este repositorio contiene el proyecto final de conducción autónoma, que incluye controladores para la recolección de datos de entrenamiento y para la conducción autónoma utilizando un modelo de IA preentrenado. El proyecto implementa un sistema end-to-end de conducción autónoma, desde la recolección de datos hasta el despliegue del modelo.

## Estructura del Repositorio
```
MR4010_Proyecto_Final_2025/
├── proyecto_final/
│   ├── test_vehicle_auto_drive.py  # Controlador de conducción autónoma
│   └── vehicle_controller.py       # Controlador para recolección de datos
├── worlds/
│   ├── city_traffic_2025_01.wbt   # Mundo de entrenamiento básico
│   └── city_traffic_2025_02.wbt   # Mundo con tráfico avanzado
├── datasets/                      # Directorio para datos de entrenamiento
├── models/                        # Directorio para modelos entrenados
└── training.ipynb                 # Notebook de entrenamiento del modelo
```

## Controladores

### [test_vehicle_auto_drive.py](MR4010_Proyecto_Final_2025/proyecto_final/test_vehicle_auto_drive.py)
Controlador principal para la conducción autónoma. Este módulo implementa:
- Detección de obstáculos usando múltiples radares
- Control de velocidad adaptativo
- Predicción de dirección usando modelo de IA
- Visualización en tiempo real del estado del vehículo

### [vehicle_controller.py](MR4010_Proyecto_Final_2025/proyecto_final/vehicle_controller.py)
Controlador para la recolección de datos de entrenamiento. Este módulo se encarga de:
- Captura de imágenes desde tres cámaras (central, izquierda, derecha)
- Preprocesamiento de imágenes para el modelo de IA
- Almacenamiento de datos en formato CSV
- Control manual del vehículo para generar datos de entrenamiento

## Entrenamiento del Modelo

### [training.ipynb](training.ipynb)
Notebook de Jupyter que contiene todo el proceso de entrenamiento del modelo de conducción autónoma:
- Carga y preprocesamiento de datos recolectados
- Arquitectura del modelo basada en NVIDIA End-to-End Learning
- Proceso de entrenamiento y validación
- Visualización de resultados y métricas
- Exportación del modelo para su uso en el controlador

### Estructura de Datos
Los datos de entrenamiento se organizan en el directorio `datasets/` con la siguiente estructura:
```
datasets/
├── YYYY-MM-DD_HH-MM-SS/         # Directorio de sesión de grabación
│   ├── imagen_central.jpg       # Imágenes de la cámara central
│   ├── imagen_izquierda.jpg     # Imágenes de la cámara izquierda
│   └── imagen_derecha.jpg       # Imágenes de la cámara derecha
└── YYYY-MM-DD_HH-MM-SS.csv      # Archivo CSV con ángulos de dirección
```

### Modelos Entrenados
Los modelos entrenados se guardan en el directorio `models/` con la siguiente estructura:
```
models/
└── best_model_cpu_v8.h5         # Modelo preentrenado actual
```

## Mundos de Simulación

### [city_traffic_2025_01.wbt](MR4010_Proyecto_Final_2025/worlds/city_traffic_2025_01.wbt)
Mundo básico de entrenamiento que incluye:
- Carreteras urbanas simples
- Condiciones de iluminación controladas
- Perfecto para recolección inicial de datos

### [city_traffic_2025_02.wbt](MR4010_Proyecto_Final_2025/worlds/city_traffic_2025_02.wbt)
Mundo avanzado con tráfico que incluye:
- Sistema de tráfico dinámico
- Múltiples tipos de vehículos
- Condiciones de conducción más desafiantes

## Configuración del Entorno

### Crear entorno de Anaconda

Crear el entorno en la terminal:

```bash
conda create --name autonomous-drive python=3.8
```

Activar el entorno:

```bash
conda activate autonomous-drive
```

### Instalar Dependencias

Instalar los requirements de Python en su entorno local de Anaconda:

```bash
pip install -r requirements.txt
```

### Instrucciones de Uso

1. Activar el entorno de Anaconda configurado
2. Cargar uno de los mundos de simulación (.wbt) en Webots
3. Ejecutar el controlador deseado desde el entorno de Anaconda:
   - Para recolección de datos: `vehicle_controller.py`
   - Para conducción autónoma: `test_vehicle_auto_drive.py`
4. Para entrenar un nuevo modelo:
   - Recolectar datos usando el controlador de recolección
   - Abrir `training.ipynb` en Jupyter Notebook
   - Seguir las instrucciones del notebook para entrenar el modelo

### Controles
- **Espacio**: Activar/Desactivar grabación o conducción autónoma
- **Flechas Arriba/Abajo**: Control de velocidad
- **Flechas Izquierda/Derecha**: Control de dirección en modo manual
