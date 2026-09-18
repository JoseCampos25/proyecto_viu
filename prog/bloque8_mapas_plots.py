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
    n_puntos, n_vars = datos_pred.shape
    similitudes_por_var = np.zeros((n_puntos, n_vars))
    
    for v in range(n_vars):
        ref_v = np.sort(datos_train_ref[:, v])
        n_ref = len(ref_v)
        if n_ref == 0:
            continue
            
        p = datos_pred[:, v]
        sim = np.zeros_like(p, dtype=float)
        
        min_ref = ref_v[0]
        max_ref = ref_v[-1]
        rango_ref = max_ref - min_ref
        if rango_ref == 0:
            rango_ref = 1e-6
            
        for i, val in enumerate(p):
            if np.isnan(val):
                sim[i] = np.nan
                continue
                
            f = np.sum(ref_v <= val) / n_ref
            
            if val < min_ref:
                sim[i] = 100.0 * f
            elif val > max_ref:
                sim[i] = 100.0 * (1.0 - f)
            else:
                if f <= 0.5:
                    sim[i] = 200.0 * f
                else:
                    sim[i] = 200.0 * (1.0 - f)
                    
        similitudes_por_var[:, v] = sim
        
    return np.min(similitudes_por_var, axis=1)

# Función de validación espacial corregida (Probabilidad vs Distancia a avistamientos reales)
def generar_scatter_validacion_espacial(prob_media, lons_grid, lats_grid, especie):
    print("Generando gráfico de validación espacial (Distancia a avistamientos)...")
    if os.path.exists(ruta_csv_obs):
        df_obs = pd.read_csv(ruta_csv_obs)
        df_obs = df_obs.dropna(subset=['latitud', 'longitud'])
        coords_obs = df_obs[['longitud', 'latitud']].values
        
        coords_grid = np.column_stack((lons_grid, lats_grid))
        if len(coords_grid) > 5000:
            indices = np.random.choice(len(coords_grid), 5000, replace=False)
            coords_grid_sub = coords_grid[indices]
            prob_media_sub = prob_media[indices]
        else:
            coords_grid_sub = coords_grid
            prob_media_sub = prob_media
            
        distancias = np.min(cdist(coords_grid_sub, coords_obs, metric='euclidean'), axis=1)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(distancias, prob_media_sub, alpha=0.2, c='teal', s=10)
        plt.title(f'Validación Espacial - Probabilidad vs Distancia: {especie}')
        plt.xlabel('Distancia Mínima al Avistamiento Real más Cercano (grados)')
        plt.ylabel('Probabilidad de Presencia Predicha')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(ruta_output_scatter, dpi=300)
        plt.close()
        print(f"Gráfico de validación espacial guardado en {ruta_output_scatter}")
    else:
        print(f"No se encontró el CSV en {ruta_csv_obs}. Se omite el scatter de validación espacial.")

def crear_grid_corregido(valores, mask_array):
    g = np.full(mask_array.shape, np.nan)
    g[mask_array.values] = valores
    return np.flipud(g)

# 2. Carga y Predicción
print("Cargando modelo y calculando predicciones...")
modelo = joblib.load(ruta_modelo)

variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho']
if os.path.exists(ruta_csv_train):
    df_train = pd.read_csv(ruta_csv_train)
    for v in variables:
        if v in df_train.columns:
            df_train[v] = df_train[v].astype(str).str.replace(r'[\[\]]', '', regex=True)
            df_train[v] = pd.to_numeric(df_train[v], errors='coerce')
    datos_train_ref = df_train[variables].dropna().values
else:
    datos_train_ref = None

with xr.open_dataset(ruta_cubo) as ds:
    ds_mapa = ds.isel(time=0) if 'time' in ds.dims else ds
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    
    lons2d, lats2d = np.meshgrid(ds.longitude.values, ds.latitude.values)
    
    mask = ds_mapa['deptho'].notnull().squeeze()
    datos = np.column_stack([np.nan_to_num(ds_mapa[v].squeeze().values[mask.values], nan=0.0) for v in variables])
    
    lons_validos = lons2d[mask.values]
    lats_validos = lats2d[mask.values]
    
    # Procesamiento por lotes (Batching) para prevenir problemas de memoria RAM
    batch_size = 10000
    n_muestras = datos.shape[0]
    
    prob_media = np.zeros(n_muestras)
    incertidumbre = np.zeros(n_muestras)
    
    for start_idx in range(0, n_muestras, batch_size):
        end_idx = min(start_idx + batch_size, n_muestras)
        batch_data = datos[start_idx:end_idx]
        
        batch_probs = np.array([tree.predict_proba(batch_data)[:, 1] for tree in modelo.estimators_])
        prob_media[start_idx:end_idx] = batch_probs.mean(axis=0)
        incertidumbre[start_idx:end_idx] = batch_probs.std(axis=0)
        
    # Cálculo de la superficie MESS
    if datos_train_ref is not None and len(datos_train_ref) > 0:
        print("Calculando superficie MESS robusta...")
        mess_valores = calcular_superficie_mess_robusta(datos, datos_train_ref)
    else:
        mess_valores = np.zeros(n_muestras)
        
    # Generar scatter plot de validación espacial real
    generar_scatter_validacion_espacial(prob_media, lons_validos, lats_validos, especie)
    
    grid_prob = crear_grid_corregido(prob_media, mask)
    grid_incert = crear_grid_corregido(incertidumbre, mask)
    grid_mess = crear_grid_corregido(mess_valores, mask)

# 3. Renderizado Final de Mapas (3 Paneles: Probabilidad, Incertidumbre y MESS)
print("Generando mapas...")
fig, axes = plt.subplots(1, 3, figsize=(21, 6), subplot_kw={'projection': ccrs.PlateCarree()})

color_tierra_costa = '#393731'

def configurar_mapa(ax, grid, cmap_name, titulo, vmin, vmax):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=color_tierra_costa)
    ax.coastlines(color='white', linewidth=0.8)
    cmap = plt.get_cmap(cmap_name).copy()
    
    cmap.set_bad(color=color_tierra_costa)
    cmap.set_over(cmap(1.0))
    cmap.set_under(cmap(0.0))
    
    im = ax.imshow(grid, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max], 
                   transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    return im

# Panel 1: Probabilidad de Presencia
im1 = configurar_mapa(axes[0], grid_prob, 'magma', f'Probabilidad de Presencia: {especie}', 0.0, 1.0)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Probabilidad')

# Panel 2: Incertidumbre
im2 = configurar_mapa(axes[1], grid_incert, 'inferno', 'Incertidumbre (Desv. Estándar)', 0.0, 0.5)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Desv. Estándar')

# Panel 3: Superficie MESS (Cálculo dinámico de límites simétricos basados en el rango real de los datos)
mess_min = np.nanmin(grid_mess)
mess_max = np.nanmax(grid_mess)
max_abs_mess = max(abs(mess_min), abs(mess_max))
# Evita un rango de cero absoluto si todos los valores fuesen exactamente 0
if max_abs_mess == 0:
    max_abs_mess = 1.0

v_min_dinamico = -max_abs_mess
v_max_dinamico = max_abs_mess

im3 = configurar_mapa(axes[2], grid_mess, 'coolwarm', 'Superficie MESS (Similitud Ambiental)', v_min_dinamico, v_max_dinamico)
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Índice MESS (%)', extend='both')

plt.tight_layout()
plt.savefig(ruta_output_mapas, dpi=300, bbox_inches='tight')
plt.close('all')

print(f"Análisis completo guardado en {ruta_base}\\results\\mapas")