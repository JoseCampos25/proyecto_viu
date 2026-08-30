# Importa el módulo del sistema operativo para gestionar rutas y directorios.
import os
# Importa la librería principal de visualización de datos matplotlib.
import matplotlib
# Configura el backend de matplotlib en modo 'Agg' para permitir la generación de gráficos sin interfaz gráfica (headless).
matplotlib.use('Agg')
# Importa xarray para trabajar con datasets multidimensionales (como archivos netCDF).
import xarray as xr
# Importa numpy para operaciones numéricas y matriciales eficientes.
import numpy as np
# Importa joblib para cargar y guardar modelos de Machine Learning entrenados de forma eficiente.
import joblib
# Importa pandas para la manipulación y análisis de datos tabulares (como archivos CSV).
import pandas as pd
# Importa pyplot desde matplotlib para crear y personalizar figuras y gráficos.
import matplotlib.pyplot as plt
# Importa el sistema de proyecciones cartográficas de Cartopy.
import cartopy.crs as ccrs
# Importa características geográficas predefinidas para Cartopy (como líneas de costa o continentes).
import cartopy.feature as cfeature
# Importa sys para acceder a los argumentos de la línea de comandos y parámetros del sistema.
import sys
# Importa cdist desde scipy para calcular distancias espaciales entre conjuntos de coordenadas.
from scipy.spatial.distance import cdist

# 1. Configuración
# Comprueba si se ha proporcionado algún argumento por la línea de comandos al ejecutar el script.
if len(sys.argv) > 1:
    # Toma el primer argumento, reemplaza los guiones bajos por espacios para restaurar el nombre científico de la especie.
    especie = sys.argv[1].replace("_", " ")
    # Imprime un mensaje indicando que la especie se ha obtenido automáticamente.
    print(f"Especie recibida automáticamente: {especie}")
else:
    # Si no hay argumentos, solicita interactivamente al usuario el nombre de la especie y elimina espacios sobrantes.
    especie = input("Nombre de la especie: ").strip()

# Crea una versión del nombre de la especie sustituyendo los espacios por guiones bajos para usarla en nombres de carpetas.
nombre_folder = especie.replace(" ", "_")
# Obtiene la ruta absoluta del directorio donde se encuentra este script.
ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
# Sube un nivel en la jerarquía de directorios para obtener la ruta de la carpeta raíz del proyecto.
ruta_base = os.path.dirname(ruta_de_este_script)

# Define la ruta completa del archivo que contiene el modelo Random Forest entrenado para la especie.
ruta_modelo = os.path.join(ruta_base, "data", "carpeta_especies", nombre_folder, f"modelo_RF_{nombre_folder}.pkl")
# Define la ruta completa del archivo NetCDF que contiene el cubo de variables ambientales de Macaronesia.
ruta_cubo = os.path.join(ruta_base, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")
# Define la ruta completa del archivo CSV con los avistamientos reales de la especie.
ruta_csv_obs = os.path.join(ruta_base, "data", "carpeta_especies", nombre_folder, f"avistamientos_{nombre_folder}.csv")

# Define la ruta de salida donde se guardará la imagen final del mapa de nicho e incertidumbre.
ruta_output_mapas = os.path.join(ruta_base, "results", "mapas", f"Mapa_Nicho_Incertidumbre_{nombre_folder}.png")
# Define la ruta de salida donde se guardará el gráfico de dispersión de validación espacial.
ruta_output_scatter = os.path.join(ruta_base, "results", "mapas", f"Scatter_Validacion_{nombre_folder}.png")

# Función de validación espacial corregida (Probabilidad vs Distancia a avistamientos reales)
# Define la función para generar un gráfico de dispersión evaluando la distancia de los píxeles a los puntos de avistamiento reales.
def generar_scatter_validacion_espacial(prob_media, lons_grid, lats_grid, especie):
    # Imprime un mensaje indicando el inicio de la generación del gráfico de validación espacial.
    print("Generando gráfico de validación espacial (Distancia a avistamientos)...")
    
    # Comprueba si el archivo CSV con las observaciones reales existe en el disco.
    if os.path.exists(ruta_csv_obs):
        # Lee el archivo CSV de observaciones utilizando pandas.
        df_obs = pd.read_csv(ruta_csv_obs)
        # Elimina aquellas filas del DataFrame que contengan valores nulos en latitud o longitud.
        df_obs = df_obs.dropna(subset=['latitud', 'longitud'])
        # Extrae las coordenadas de longitud y latitud de las observaciones reales como un arreglo numérico.
        coords_obs = df_obs[['longitud', 'latitud']].values
        
        # Muestreo aleatorio de celdas del grid para optimizar rendimiento del scatter
        # Combina las longitudes y latitudes de la malla en un único arreglo de coordenadas bidimensionales.
        coords_grid = np.column_stack((lons_grid, lats_grid))
        # Verifica si el número de puntos en la malla supera los 5000 para realizar un submuestreo.
        if len(coords_grid) > 5000:
            # Selecciona de forma aleatoria 5000 índices únicos de la malla sin reemplazo.
            indices = np.random.choice(len(coords_grid), 5000, replace=False)
            # Filtra las coordenadas de la malla usando los índices seleccionados.
            coords_grid_sub = coords_grid[indices]
            # Filtra las probabilidades medias correspondientes a esos índices submuestreados.
            prob_media_sub = prob_media[indices]
        else:
            # Si hay 5000 puntos o menos, utiliza la malla y las probabilidades completas sin submuestreo.
            coords_grid_sub = coords_grid
            prob_media_sub = prob_media
            
        # Calcular la distancia mínima de cada píxel marino al avistamiento real más cercano (en grados)
        # Calcula la matriz de distancias euclidianas entre la submuestra de la malla y los avistamientos reales, obteniendo la mínima para cada punto.
        distancias = np.min(cdist(coords_grid_sub, coords_obs, metric='euclidean'), axis=1)
        
        # Inicializa una nueva figura de matplotlib con un tamaño específico de 8x6 pulgadas.
        plt.figure(figsize=(8, 6))
        # Dibuja un gráfico de dispersión relacionando las distancias calculadas con la probabilidad media predicha.
        plt.scatter(distancias, prob_media_sub, alpha=0.2, c='teal', s=10)
        # Establece el título del gráfico incluyendo el nombre de la especie analizada.
        plt.title(f'Validación Espacial - Probabilidad vs Distancia: {especie}')
        # Etiqueta el eje X indicando la distancia mínima a los avistamientos reales.
        plt.xlabel('Distancia Mínima al Avistamiento Real más Cercano (grados)')
        # Etiqueta el eje Y indicando la probabilidad de presencia predicha.
        plt.ylabel('Probabilidad de Presencia Predicha')
        # Activa la cuadrícula en el gráfico con líneas discontinuas y cierta transparencia.
        plt.grid(True, linestyle='--', alpha=0.6)
        # Guarda la figura generada en la ruta de salida del scatter con una resolución de 300 DPI.
        plt.savefig(ruta_output_scatter, dpi=300)
        # Cierra la figura actual para liberar memoria.
        plt.close()
        # Imprime un mensaje confirmando que el gráfico de validación fue guardado con éxito.
        print(f"Gráfico de validación espacial guardado en {ruta_output_scatter}")
    else:
        # Imprime una advertencia si no se encuentra el archivo CSV de avistamientos para realizar el scatter.
        print(f"No se encontró el CSV en {ruta_csv_obs}. Se omite el scatter de validación espacial.")

# 2. Carga y Predicción
# Imprime un mensaje indicando que se está cargando el modelo y calculando las predicciones.
print("Cargando modelo y calculando predicciones...")
# Carga el modelo de Machine Learning previamente entrenado desde el archivo .pkl usando joblib.
modelo = joblib.load(ruta_modelo)

# Abre el dataset netCDF ambiental utilizando un gestor de contexto (with).
with xr.open_dataset(ruta_cubo) as ds:
    # Selecciona el primer paso temporal (time=0) del dataset espacial.
    ds_mapa = ds.isel(time=0)
    # Extrae los valores mínimos y máximos de longitud del dataset convirtiéndolos a float.
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    # Extrae los valores mínimos y máximos de latitud del dataset convirtiéndolos a float.
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    
    # Extraer coordenadas planas de la malla para el cálculo espacial
    # Genera una malla bidimensional de coordenadas de longitud y latitud a partir de las coordenadas del dataset.
    lons2d, lats2d = np.meshgrid(ds.longitude.values, ds.latitude.values)
    
    # Crea una máscara booleana basada en los valores no nulos de la profundidad (deptho) en el mapa.
    mask = ds_mapa['deptho'].notnull().squeeze()
    # Define la lista de variables ambientales requeridas por el modelo.
    variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho']
    # Apila los valores de las variables ambientales seleccionadas filtrándolos con la máscara de puntos válidos.
    datos = np.column_stack([ds_mapa[v].squeeze().values[mask.values] for v in variables])
    
    # Coordenadas válidas enmascaradas
    # Extrae las longitudes correspondientes únicamente a las posiciones válidas de la máscara.
    lons_validos = lons2d[mask.values]
    # Extrae las latitudes correspondientes únicamente a las posiciones válidas de la máscara.
    lats_validos = lats2d[mask.values]
    
    # Cálculo de incertidumbre y probabilidad media
    # Obtiene las probabilidades de predicción para la clase positiva de cada árbol individual que compone el Random Forest.
    all_probs = np.array([tree.predict_proba(datos)[:, 1] for tree in modelo.estimators_])
    # Calcula la probabilidad media predicha promediando las predicciones de todos los árboles a lo largo del eje 0.
    prob_media = all_probs.mean(axis=0)
    # Calcula la incertidumbre obteniendo la desviación estándar de las predicciones entre los árboles.
    incertidumbre = all_probs.std(axis=0)
    
    # Generar scatter plot de validación espacial real
    # Llama a la función para generar el gráfico de validación espacial con los datos calculados.
    generar_scatter_validacion_espacial(prob_media, lons_validos, lats_validos, especie)
    
    # Creación de grids corregidos (inicializados en NaN)
    # Define una función interna para reconstruir una cuadrícula espacial bidimensional rellenando los valores válidos sobre una matriz de NaNs.
    def crear_grid_corregido(valores, mask_array):
        # Crea una matriz llena de valores NaN con la misma forma geométrica del arreglo de la máscara.
        g = np.full(mask_array.shape, np.nan)
        # Asigna los valores calculados en las posiciones geográficas indicadas por la máscara booleana.
        g[mask_array.values] = valores
        # Voltea verticalmente la matriz resultante para corregir la orientación espacial al graficar.
        return np.flipud(g)

    # Reconstruye la cuadrícula espacial bidimensional para la probabilidad media de presencia.
    grid_prob = crear_grid_corregido(prob_media, mask)
    # Reconstruye la cuadrícula espacial bidimensional para la incertidumbre del modelo.
    grid_incert = crear_grid_corregido(incertidumbre, mask)

# 3. Renderizado Final de Mapas
# Imprime un mensaje indicando que se procederá a generar la renderización final de los mapas.
print( "Generando mapas...")
# Crea una figura con dos subgráficos (axes) lado a lado, configurados con la proyección cartográfica PlateCarree.
fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': ccrs.PlateCarree()})

# Define una función para configurar de forma automatizada los elementos visuales de cada mapa cartográfico.
def configurar_mapa(ax, grid, cmap_name, titulo, vmin, vmax):
    # Define la extensión espacial (límites geográficos) del mapa usando los valores mínimos y máximos de longitud y latitud.
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # Añade la capa de tierra firme al mapa con un color de relleno específico.
    ax.add_feature(cfeature.LAND, facecolor='#393731')
    # Dibuja las líneas de costa de los continentes e islas en color blanco.
    ax.coastlines(color='white')
    # Obtiene una copia del mapa de colores (colormap) especificado.
    cmap = plt.get_cmap(cmap_name).copy()
    # Asigna el color negro para representar los valores nulos o inválidos (bad).
    cmap.set_bad(color='black') 
    # Configura el color para los valores que excedan el límite superior de la escala.
    cmap.set_over(cmap(1.0))
    # Configura el color para los valores que queden por debajo del límite inferior de la escala.
    cmap.set_under(cmap(0.0))
    # Dibuja la cuadrícula de datos en la imagen utilizando los límites geográficos, la proyección y el mapa de colores configurado.
    im = ax.imshow(grid, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max], 
                    transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    # Establece el título correspondiente al subgráfico.
    ax.set_title(titulo)
    # Retorna el objeto de la imagen para su posterior uso en la barra de colores.
    return im

# Configura y renderiza el primer mapa correspondiente a la probabilidad de presencia usando la paleta 'magma'.
im1 = configurar_mapa(axes[0], grid_prob, 'magma', f'Probabilidad de Presencia: {especie}', 0.0, 1.0)
# Añade una barra de colores asociada al primer mapa indicando la escala de probabilidad.
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Probabilidad')

# Configura y renderiza el segundo mapa correspondiente a la incertidumbre usando la paleta 'inferno'.
im2 = configurar_mapa(axes[1], grid_incert, 'inferno', 'Incertidumbre (Desv. Estándar)', 0, 0.5)
# Añade una barra de colores asociada al segundo mapa indicando la escala de incertidumbre.
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Incertidumbre')

# Guarda la figura completa con ambos mapas en la ruta de salida definida, ajustando los márgenes y con alta resolución.
plt.savefig(ruta_output_mapas, dpi=300, bbox_inches='tight')
# Cierra todas las figuras abiertas de matplotlib para liberar los recursos de memoria del sistema.
plt.close('all')

# Imprime un mensaje final indicando que el análisis completo ha sido guardado con éxito en la ruta de resultados.
print(f" Análisis completo guardado en {ruta_base}\\results\\mapas")