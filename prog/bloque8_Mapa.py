import os
# 1. Configuración de entorno: Evitar errores de librerías
os.environ['PROJ_LIB'] = r'C:\Users\joslo\miniconda3\envs\clima_tfm\Library\share\proj'

# 2. Configuración de backend: Forzar ejecución en segundo plano (sin abrir ventanas)
import matplotlib
matplotlib.use('Agg') 

import xarray as xr
import numpy as np
import joblib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 3. Configuración de rutas
especie = input("Nombre de la especie: ").strip()
nombre_folder = especie.replace(" ", "_")
ruta_base = r"C:\TFM\proyecto_viu"
ruta_modelo = os.path.join(ruta_base, "data", "carpeta_especies", nombre_folder, f"modelo_RF_{nombre_folder}.pkl")
ruta_cubo = os.path.join(ruta_base, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")
ruta_output = os.path.join(ruta_base, "results", "mapas", f"Mapa_Nicho_{nombre_folder}.png")
os.makedirs(os.path.dirname(ruta_output), exist_ok=True)

# 4. Carga y Predicción
print("⏳ Cargando modelo y datos...")
modelo = joblib.load(ruta_modelo)

with xr.open_dataset(ruta_cubo) as ds:
    ds_mapa = ds.isel(time=0)
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    
    mask = ds_mapa['deptho'].notnull().squeeze()
    variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2']
    datos = np.column_stack([ds_mapa[v].squeeze().values[mask.values] for v in variables])
    
    prob = modelo.predict_proba(datos)[:, 1]
    grid = np.zeros(mask.shape)
    grid[mask.values] = prob
    grid = np.flipud(grid)
    # Convertir ceros a NaN para que no se pinten en el mapa
    grid[grid == 0] = np.nan

# 5. Renderizado final (con bordes blancos)
print("🚀 Generando mapa...")
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Configurar el zoom
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# --- AJUSTES DE DELINEADO ---
# Añadimos edgecolor='white' para el borde de la tierra
# y linewidth para el grosor del delineado
ax.add_feature(cfeature.LAND, facecolor='#393731', edgecolor='white', linewidth=1.5)

# Delineado de costa en blanco
ax.coastlines(resolution='50m', color='white', linewidth=1.0)
# ---------------------------

# Configurar color de los NaNs a negro
cmap = plt.get_cmap('magma').copy()
cmap.set_bad(color='black') 

# Pintar datos
im = ax.imshow(grid, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max], 
               transform=ccrs.PlateCarree(), cmap=cmap, 
               vmin=0.3, vmax=1.0)

plt.colorbar(im, ax=ax, label='Probabilidad de Presencia')
plt.title(f"Distribución: {especie}")

plt.savefig(ruta_output, dpi=300, bbox_inches='tight')
plt.close('all')

print(f"✅ ÉXITO: Mapa con bordes blancos generado en {ruta_output}")