# Proyecto Final

## Crear entorno de Anaconda en caso de no tenerlo

Crear el entorno en la terminal

```
conda create --name autonomous-drive python=3.8
```

Activar el entorno una vez creado

```
conda activate autonomous-drive
```

## Instalar requirements de python en su entorno local de anaconda.

Desde dentro de su entorno de python en la terminal ejecuten:

```
pip install -r requirements.txt
```

## Archivos importantes

Dentro de la carpeta proyecto_final se encuentran el archivo Python que se debe ejecutar.

Dentro de la carpeta `MR4010_Proyecto_Final_2025` se encuentra el archivo `worlds/city_traffic_2025_01.wbt`. Este es el archivo de mundo de webots modificado para esta tarea de captura de datos. Favor de cargar este archivo.

Recuerden ejecutar el archivo `vehicle_controlle.py` desde el entorno de Anaconda que habían configurado en la primera práctica de webots.
