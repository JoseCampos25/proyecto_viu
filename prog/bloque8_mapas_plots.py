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
# Define la ruta completa del dataset de entrenamiento para los rangos de referencia de la superficie MESS.
ruta_csv_train = os.path.join(ruta_base, "data", "carpeta_especies", nombre_folder, f"Dataset_presencias_ausencias_{nombre_folder}.csv")

# Define la ruta de salida donde se guardará la imagen final del mapa de nicho, incertidumbre y MESS.
ruta_output_mapas = os.path.join(ruta_base, "results", "mapas", f"Mapa_Nicho_Incertidumbre_MESS_{nombre_folder}.png")
# Define la ruta de salida donde se guardará el gráfico de dispersión de validación espacial.
ruta_output_scatter = os.path.join(ruta_base, "results", "mapas", f"Scatter_Validacion_{nombre_folder}.png")

def calcular_superficie_mess_robusta(datos_pred, datos_train_ref):
    """
    Cálculo riguroso del índice MESS (Elith et al., 2010).
    Compara los puntos de predicción frente a los rangos de referencia de entrenamiento.
    Devuelve valores entre -100 y 100. Negativos = condiciones novedosas (extrapolación).
    """
    # Obtiene el número de puntos a predecir y el número de variables ambientales.
    n_puntos, n_vars = datos_pred.shape
    # Inicializa una matriz de ceros para almacenar la similitud por cada variable.
    similitudes_por_var = np.zeros((n_puntos, n_vars))
    
    # Recorre cada variable ambiental.
    for v in range(n_vars):
        # Ordena de menor a mayor los datos de referencia de entrenamiento para la variable actual.
        ref_v = np.sort(datos_train_ref[:, v])
        # Obtiene la cantidad de datos de referencia.
        n_ref = len(ref_v)
        # Si no hay datos de referencia para la variable, salta a la siguiente.
        if n_ref == 0:
            continue
            
        # Extrae los datos de predicción para la variable actual.
        p = datos_pred[:, v]
        # Inicializa un array para almacenar las puntuaciones de similitud de esta variable.
        sim = np.zeros_like(p, dtype=float)
        
        # Identifica los valores mínimo y máximo en el conjunto de entrenamiento de referencia.
        min_ref = ref_v[0]
        max_ref = ref_v[-1]
        # Calcula el rango de la variable en los datos de entrenamiento.
        rango_ref = max_ref - min_ref
        # Previene división por cero en caso de que el rango sea cero.
        if rango_ref == 0:
            rango_ref = 1e-6
            
        # Evalúa el valor ambiental en cada punto de predicción.
        for i, val in enumerate(p):
            # Si el valor ambiental es un NaN, se asigna NaN a la similitud.
            if np.isnan(val):
                sim[i] = np.nan
                continue
                
            # Calcula la fracción de datos de referencia menores o iguales al valor actual.
            f = np.sum(ref_v <= val) / n_ref
            
            # Si el valor está por debajo del mínimo observado en el entrenamiento (extrapolación inferior).
            if val < min_ref:
                sim[i] = 100.0 * f
            # Si el valor está por encima del máximo observado en el entrenamiento (extrapolación superior).
            elif val > max_ref:
                sim[i] = 100.0 * (1.0 - f)
            # Si el valor está dentro del rango observado en el entrenamiento (interpolación).
            else:
                if f <= 0.5:
                    sim[i] = 200.0 * f
                else:
                    sim[i] = 200.0 * (1.0 - f)
                    
        # Guarda las puntuaciones de la variable actual en la matriz global.
        similitudes_por_var[:, v] = sim
        
    # Asigna la similitud mínima entre todas las variables para cada punto (criterio del 'factor más limitante').
    return np.min(similitudes_por_var, axis=1)

# Función de validación espacial corregida (Probabilidad vs Distancia a avistamientos reales)
def generar_scatter_validacion_espacial(prob_media, lons_grid, lats_grid, especie):
    # Notifica por consola la generación del gráfico de validación.
    print("Generando gráfico de validación espacial (Distancia a avistamientos)...")
    # Verifica si existe el archivo CSV de observaciones reales.
    if os.path.exists(ruta_csv_obs):
        # Carga el CSV de avistamientos observados.
        df_obs = pd.read_csv(ruta_csv_obs)
        # Limpia filas que tengan coordenadas faltantes.
        df_obs = df_obs.dropna(subset=['latitud', 'longitud'])
        # Extrae las coordenadas observadas en un array de numpy.
        coords_obs = df_obs[['longitud', 'latitud']].values
        
        # Empareja las longitudes y latitudes de la rejilla marina válida.
        coords_grid = np.column_stack((lons_grid, lats_grid))
        # Si la cantidad de puntos supera los 5000, toma una muestra aleatoria para no saturar el cálculo ni la figura.
        if len(coords_grid) > 5000:
            indices = np.random.choice(len(coords_grid), 5000, replace=False)
            coords_grid_sub = coords_grid[indices]
            prob_media_sub = prob_media[indices]
        else:
            coords_grid_sub = coords_grid
            prob_media_sub = prob_media
            
        # Calcula la distancia euclidiana mínima de cada punto marino al avistamiento real más cercano.
        distancias = np.min(cdist(coords_grid_sub, coords_obs, metric='euclidean'), axis=1)
        
        # Crea la figura del scatter plot de validación.
        plt.figure(figsize=(8, 6))
        plt.scatter(distancias, prob_media_sub, alpha=0.2, c='teal', s=10)
        plt.title(f'Validación Espacial - Probabilidad vs Distancia: {especie}')
        plt.xlabel('Distancia Mínima al Avistamiento Real más Cercano (grados)')
        plt.ylabel('Probabilidad de Presencia Predicha')
        plt.grid(True, linestyle='--', alpha=0.6)
        # Guarda la gráfica generada en disco.
        plt.savefig(ruta_output_scatter, dpi=300)
        plt.close()
        print(f"Gráfico de validación espacial guardado en {ruta_output_scatter}")
    else:
        # Notifica que se omite el gráfico si no se localiza el CSV.
        print(f"No se encontró el CSV en {ruta_csv_obs}. Se omite el scatter de validación espacial.")

def crear_grid_corregido(valores, mask_array):
    # Inicializa una matriz 2D llena de NaNs con la forma geográfica del cubo.
    g = np.full(mask_array.shape, np.nan)
    # Inserta los valores calculados solo en los píxeles marinos válidos indicados por la máscara.
    g[mask_array.values] = valores
    # Voltea verticalmente la matriz para corregir la orientación espacial de los mapas raster al renderizarlos.
    return np.flipud(g)

# 2. Carga y Predicción
print("Cargando modelo y calculando predicciones...")
# Carga el modelo Random Forest desde el archivo binario pkl.
modelo = joblib.load(ruta_modelo)

# Lista de las 8 variables ambientales del cubo netCDF que utiliza el modelo.
variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho']
# Si existe el archivo de datos de entrenamiento, procesa sus variables.
if os.path.exists(ruta_csv_train):
    df_train = pd.read_csv(ruta_csv_train)
    for v in variables:
        if v in df_train.columns:
            # Limpia posibles corchetes o formato de texto remanente en los números.
            df_train[v] = df_train[v].astype(str).str.replace(r'[\[\]]', '', regex=True)
            # Convierte las columnas a tipo numérico.
            df_train[v] = pd.to_numeric(df_train[v], errors='coerce')
    # Guarda el array con las variables limpias sin valores nulos para el cálculo del MESS.
    datos_train_ref = df_train[variables].dropna().values
else:
    datos_train_ref = None

# Abre la conexión con el dataset multidimensional del cubo ambiental netCDF.
with xr.open_dataset(ruta_cubo) as ds:
    # Selecciona el primer paso temporal si el archivo contiene dimensión de tiempo.
    ds_mapa = ds.isel(time=0) if 'time' in ds.dims else ds
    # Obtiene los límites geográficos máximos y mínimos de la región.
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    
    # Crea la malla 2D de coordenadas de longitud y latitud.
    lons2d, lats2d = np.meshgrid(ds.longitude.values, ds.latitude.values)
    
    # Genera una máscara booleana para filtrar celdas marinas válidas usando la batimetría (deptho).
    mask = ds_mapa['deptho'].notnull().squeeze()
    # Apila las 8 variables ambientales filtradas por la máscara en una matriz de datos 2D.
    datos = np.column_stack([np.nan_to_num(ds_mapa[v].squeeze().values[mask.values], nan=0.0) for v in variables])
    
    # Extrae las longitudes y latitudes que corresponden a celdas marinas válidas.
    lons_validos = lons2d[mask.values]
    lats_validos = lats2d[mask.values]
    
    # Procesamiento por lotes (Batching) para prevenir problemas de memoria RAM
    batch_size = 10000
    n_muestras = datos.shape[0]
    
    # Inicializa arreglos para almacenar la media (predicción) y desviación estándar (incertidumbre).
    prob_media = np.zeros(n_muestras)
    incertidumbre = np.zeros(n_muestras)
    
    # Procesa los datos marinos por bloques/lotes para optimizar el rendimiento y memoria.
    for start_idx in range(0, n_muestras, batch_size):
        end_idx = min(start_idx + batch_size, n_muestras)
        batch_data = datos[start_idx:end_idx]
        
        # Obtiene la predicción de cada uno de los árboles individuales del Random Forest para el lote actual.
        batch_probs = np.array([tree.predict_proba(batch_data)[:, 1] for tree in modelo.estimators_])
        # Calcula el promedio de probabilidad entre todos los árboles.
        prob_media[start_idx:end_idx] = batch_probs.mean(axis=0)
        # Calcula la desviación estándar entre las predicciones de los árboles como medida de incertidumbre.
        incertidumbre[start_idx:end_idx] = batch_probs.std(axis=0)
        
    # Cálculo de la superficie MESS
    if datos_train_ref is not None and len(datos_train_ref) > 0:
        print("Calculando superficie MESS robusta...")
        mess_valores = calcular_superficie_mess_robusta(datos, datos_train_ref)
    else:
        mess_valores = np.zeros(n_muestras)
        
    # Generar scatter plot de validación espacial real
    generar_scatter_validacion_espacial(prob_media, lons_validos, lats_validos, especie)
    
    # Reconstruye las matrices 2D espaciales a partir de los resultados vectorizados.
    grid_prob = crear_grid_corregido(prob_media, mask)
    grid_incert = crear_grid_corregido(incertidumbre, mask)
    grid_mess = crear_grid_corregido(mess_valores, mask)

# 3. Renderizado Final de Mapas (3 Paneles: Probabilidad, Incertidumbre y MESS)
print("Generando mapas...")
# Crea la figura de 3 paneles con proyección cartográfica equirectangular (PlateCarree).
fig, axes = plt.subplots(1, 3, figsize=(21, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Define el color hexadecimal para representar las masas de tierra.
color_tierra_costa = '#393731'

# Función auxiliar para configurar y dibujar los mapas con sus atributos visuales.
def configurar_mapa(ax, grid, cmap_name, titulo, vmin, vmax):
    # Ajusta los límites de visualización geográfica al área de Macaronesia.
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # Añade el polígono de continentes/islas con el color asignado.
    ax.add_feature(cfeature.LAND, facecolor=color_tierra_costa)
    # Dibuja la línea de costa en blanco.
    ax.coastlines(color='white', linewidth=0.8)
    # Carga el mapa de colores especificado.
    cmap = plt.get_cmap(cmap_name).copy()
    
    # Configura el color de fondo para valores nulos (tierra/NaNs) y valores fuera de escala.
    cmap.set_bad(color=color_tierra_costa)
    cmap.set_over(cmap(1.0))
    cmap.set_under(cmap(0.0))
    
    # Dibuja la imagen raster en la proyección geográfica correspondiente.
    im = ax.imshow(grid, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max], 
                   transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    # Asigna el título de la subfigura.
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    return im

# Panel 1: Probabilidad de Presencia
im1 = configurar_mapa(axes[0], grid_prob, 'magma', f'Probabilidad de Presencia: {especie}', 0.0, 1.0)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Probabilidad')

# Panel 2: Incertidumbre
im2 = configurar_mapa(axes[1], grid_incert, 'inferno', 'Incertidumbre (Desv. Estándar)', 0.0, 0.5)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Desv. Estándar')

# Panel 3: Superficie MESS (Cálculo dinámico de límites simétricos basados en el rango real de los datos)
# Encuentra los valores mínimos y máximos en el grid MESS ignorando NaNs.
mess_min = np.nanmin(grid_mess)
mess_max = np.nanmax(grid_mess)
# Determina el valor absoluto máximo para balancear la escala divergente simétricamente alrededor de 0.
max_abs_mess = max(abs(mess_min), abs(mess_max))
# Evita un rango de cero absoluto si todos los valores fuesen exactamente 0
if max_abs_mess == 0:
    max_abs_mess = 1.0

# Define los límites simétricos para el mapa de color divergente.
v_min_dinamico = -max_abs_mess
v_max_dinamico = max_abs_mess

# Dibuja el panel 3 (MESS) usando la paleta divergente 'coolwarm'.
im3 = configurar_mapa(axes[2], grid_mess, 'coolwarm', 'Superficie MESS (Similitud Ambiental)', v_min_dinamico, v_max_dinamico)
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Índice MESS (%)', extend='both')

# Ajusta el diseño de los tres paneles para evitar superposiciones de etiquetas.
plt.tight_layout()
# Guarda la figura completa en alta resolución (300 DPI).
plt.savefig(ruta_output_mapas, dpi=300, bbox_inches='tight')
# Cierra las figuras en memoria para liberar recursos informáticos.
plt.close('all')

# Mensaje final informando la ruta donde se guardaron los resultados.
print(f"Análisis completo guardado en {ruta_base}\\results\\mapas")