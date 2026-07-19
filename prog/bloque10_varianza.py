import os
import matplotlib
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import joblib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. Configuración
especie = input("Nombre de la especie: ").strip()
nombre_folder = especie.replace(" ", "_")
ruta_base = r"C:\TFM\proyecto_viu"
ruta_modelo = os.path.join(ruta_base, "data", "carpeta_especies", nombre_folder, f"modelo_RF_{nombre_folder}.pkl")
ruta_cubo = os.path.join(ruta_base, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")
ruta_output = os.path.join(ruta_base, "results", "mapas", f"Mapa_Nicho_Incertidumbre_{nombre_folder}.png")

# 2. Carga y Predicción
print("⏳ Cargando modelo y calculando predicciones...")
modelo = joblib.load(ruta_modelo)

with xr.open_dataset(ruta_cubo) as ds:
    ds_mapa = ds.isel(time=0)
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    
    # Obtenemos la máscara correctamente
    mask = ds_mapa['deptho'].notnull().squeeze()
    variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho']
    datos = np.column_stack([ds_mapa[v].squeeze().values[mask.values] for v in variables])
    
    # Predicciones
    all_probs = np.array([tree.predict_proba(datos)[:, 1] for tree in modelo.estimators_])
    prob_media = all_probs.mean(axis=0)
    incertidumbre = all_probs.std(axis=0)
    
    # CREACIÓN DEL GRID: Inicializamos con NaN
    # Esto asegura que todo lo que no sea máscara sea transparente/negro
    def crear_grid_corregido(valores, mask_array):
        g = np.full(mask_array.shape, np.nan)
        g[mask_array.values] = valores
        return np.flipud(g) # Volteamos el grid completo para corregir la inversión de latitud

    grid_prob = crear_grid_corregido(prob_media, mask)
    grid_incert = crear_grid_corregido(incertidumbre, mask)

# 3. Renderizado Final
print("🚀 Generando mapas corregidos...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': ccrs.PlateCarree()})

def configurar_mapa(ax, grid, cmap_name, titulo, vmin, vmax):
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#393731')
    ax.coastlines(color='white')
    
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color='black') 
    cmap.set_over(cmap(1.0))
    cmap.set_under(cmap(0.0))
    
    # Usamos origin='upper' porque el flipud previo ya corrigió la latitud
    im = ax.imshow(grid, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max], 
                   transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(titulo)
    return im

# Mapas
im1 = configurar_mapa(axes[0], grid_prob, 'magma', f'Probabilidad de Presencia: {especie}', 0.3, 1.0)
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Probabilidad')

im2 = configurar_mapa(axes[1], grid_incert, 'inferno', 'Incertidumbre (Desv. Estándar)', 0, 0.35)
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Incertidumbre')

plt.savefig(ruta_output, dpi=300, bbox_inches='tight')
plt.close('all')

print(f"✅ ÉXITO: Mapa limpio generado en {ruta_output}")