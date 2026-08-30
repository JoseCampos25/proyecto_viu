# Sesgo_espacial.py
# Script independiente para analizar el sesgo espacial y generar un mapa de calor a partir del CSV de avistamientos.

# Importa el módulo para interactuar con el sistema operativo (rutas, archivos).
import os
# Importa el módulo para acceder a parámetros y funciones del sistema (como argumentos de la línea de comandos).
import sys
# Importa el módulo para leer y escribir archivos en formato CSV.
import csv
# Importa la librería numérica para cálculos matriciales y científicos.
import numpy as np
# Importa la librería para generar mapas interactivos web.
import folium
# Importa la extensión de mapas de calor para Folium.
from folium.plugins import HeatMap

#----------------- OBTENER LA ESPECIE Y RUTA DEL CSV ----------------------
# Comprueba si se ha pasado al menos un argumento por la línea de comandos al ejecutar el script.
if len(sys.argv) > 1:
    # Recoge el primer argumento y reemplaza los guiones bajos por espacios para formar el nombre científico.
    especie = sys.argv[1].replace("_", " ")
    # Muestra por consola que la especie se ha obtenido de forma automática.
    print(f"Especie recibida automáticamente: {especie}")
else:
    # Si no hay argumentos, solicita interactivamente al usuario que introduzca el nombre científico y elimina espacios sobrantes.
    especie = input("Introduce el nombre científico de la especie (ej: Orcinus orca): ").strip()

# Obtiene la ruta absoluta del directorio donde se encuentra este script actual.
ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
# Sube un nivel en la jerarquía de directorios para obtener la ruta de la carpeta raíz del proyecto.
ruta_raiz = os.path.dirname(ruta_de_este_script)
# Reemplaza los espacios del nombre de la especie por guiones bajos para usarlo de forma segura en nombres de carpetas.
nombre_carpeta_especie = especie.replace(" ", "_")
# Construye la ruta completa hacia la carpeta específica de la especie dentro de la estructura de directorios del TFM.
carpeta_especie = os.path.join(ruta_raiz, "data", "carpeta_especies", nombre_carpeta_especie)

# Define la ruta completa del archivo CSV de avistamientos para esa especie específica.
csv_filename = os.path.join(carpeta_especie, f"avistamientos_{nombre_carpeta_especie}.csv")
# Define la ruta completa donde se guardará el archivo HTML resultante con el mapa de calor interactivo.
heatmap_filename = os.path.join(carpeta_especie, f"heatmapOBIS_{nombre_carpeta_especie}.html")

# Verifica si el archivo CSV de avistamientos no existe físicamente en el disco.
if not os.path.exists(csv_filename):
    # Imprime un mensaje advirtiendo que el archivo no fue hallado e indica su ruta esperada.
    print(f"No se encontró el archivo CSV en la ruta: {csv_filename}")
    # Sugiere al usuario ejecutar primero el script de descarga correspondiente.
    print("Ejecuta primero el script de descargas (Avistamientos.py) para esta especie.")
    # Finaliza la ejecución del script de manera abrupta.
    exit()

#----------------- LEER LOS REGISTROS DESDE EL CSV ----------------------
# Inicializa una lista vacía para almacenar los registros de avistamientos procesados.
registros_ordenados = []
# Abre el archivo CSV en modo lectura ('r') utilizando la codificación UTF-8.
with open(csv_filename, "r", encoding="utf-8") as f:
    # Crea un lector de diccionarios que interpreta la primera fila del CSV como las cabeceras de las columnas.
    reader = csv.DictReader(f)
    # Itera de forma secuencial sobre cada fila (diccionario) contenida en el archivo CSV.
    for row in reader:
        # Intenta convertir la latitud y longitud a números decimales (float); si el valor es texto "None", asigna None.
        try:
            lat = float(row["latitud"]) if row["latitud"] != "None" else None
            lon = float(row["longitud"]) if row["longitud"] != "None" else None
        # Captura errores de conversión numérica (ValueError) y asigna None a ambas coordenadas si falla.
        except ValueError:
            lat, lon = None, None
            
        # Añade un diccionario estructurado con fecha, hora, latitud y longitud limpios a la lista de registros.
        registros_ordenados.append({
            "fecha": row["fechas"],
            "hora": row["hora"],
            "lat": lat,
            "lon": lon
        })


#----------------- ANÁLISIS BÁSICO DE SESGO ESPACIAL ----------------------
# Define la función encargada de evaluar el sesgo espacial a partir de la lista de registros.
def evaluar_sesgo_espacial(registros_ordenados):
    # Extrae en una lista todas las latitudes que no sean valores nulos (None).
    lats = [r["lat"] for r in registros_ordenados if r["lat"] is not None]
    # Extrae en una lista todas las longitudes que no sean valores nulos (None).
    lons = [r["lon"] for r in registros_ordenados if r["lon"] is not None]
    
    # Comprueba si la lista de latitudes está vacía.
    if not lats:
        # Retorna un mensaje de advertencia si no existen coordenadas suficientes para realizar el análisis.
        return "No hay suficientes coordenadas para evaluar sesgo."
    
    # Calcula la amplitud o rango geográfico cubierto en latitud (diferencia entre el valor máximo y mínimo).
    lat_range = max(lats) - min(lats)
    # Calcula la amplitud o rango geográfico cubierto en longitud (diferencia entre el valor máximo y mínimo).
    lon_range = max(lons) - min(lons)
    
    # Imprime la cabecera del diagnóstico de sesgo de observación en la consola.
    print("\n--- DIAGNÓSTICO DE SESGO DE OBSERVACIÓN ---")
    # Imprime el nombre de la especie que está siendo analizada.
    print(f"Especie analizada: {especie}")
    # Imprime el recuento total de registros válidos analizados.
    print(f"Total de registros válidos analizados: {len(lats)}")
    # Imprime el rango de latitud calculado con dos decimales.
    print(f"Rango de latitud cubierto: {lat_range:.2f}°")
    # Imprime el rango de longitud calculado con dos decimales.
    print(f"Rango de longitud cubierto: {lon_range:.2f}°")
    
    

# Llama a la función de evaluación de sesgo espacial pasando la lista de registros leídos.
evaluar_sesgo_espacial(registros_ordenados)


#----------------- CREAR Y GUARDAR EL MAPA DE CALOR ----------------------
# Genera una lista de pares de coordenadas [latitud, longitud] filtrando exclusivamente aquellos registros donde ambas coordenadas son válidas.
coordenadas_calor = [[r['lat'], r['lon']] for r in registros_ordenados if r["lat"] is not None and r['lon'] is not None]

# Comprueba si la lista de coordenadas para el mapa de calor contiene elementos.
if coordenadas_calor:
    # Inicializa un objeto mapa base de Folium centrado en las coordenadas geográficas aproximadas [28, -20] con un nivel de zoom inicial de 5.
    mapa_calor = folium.Map(location=[28, -20], zoom_start=5)
    # Crea la capa de mapa de calor (HeatMap) utilizando las coordenadas, un radio, desenfoque y nivel máximo de zoom definidos, y la añade al mapa base.
    HeatMap(coordenadas_calor, radius=15, blur=10, max_zoom=1).add_to(mapa_calor)
    
    # Guarda el mapa interactivo generado como un archivo HTML en la ruta especificada por la variable correspondiente.
    mapa_calor.save(heatmap_filename)
    # Imprime un mensaje confirmando que el mapa de calor se guardó con éxito y mostrando su ruta.
    print(f"\nMapa de calor (Heatmap) guardado en: {heatmap_filename}\n")
else:
    # Imprime un mensaje de aviso indicando que no hay coordenadas válidas disponibles para generar el mapa de calor.
    print("No hay coordenadas válidas para generar el mapa de calor.")