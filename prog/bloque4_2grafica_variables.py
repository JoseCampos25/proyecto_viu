# Script de diagnóstico para generar mapas individuales de las 8 variables ambientales del cubo NetCDF

import os
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. Configuración de rutas
ruta_base = os.getcwd()
ruta_cubo = os.path.join(ruta_base, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")
carpeta_salida = os.path.join(ruta_base, "results", "diagnostico_variables")

os.makedirs(carpeta_salida, exist_ok=True)

# Lista completa de las 8 variables del modelo
variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho']

print("Iniciando generación de mapas de diagnóstico para las 8 variables...")

if not os.path.exists(ruta_cubo):
    print(f"Error: No se encontró el cubo NetCDF en {ruta_cubo}")
    exit()

# 2. Carga del cubo y bucle de renderizado por variable
with xr.open_dataset(ruta_cubo) as ds:
    # Se calcula la media temporal para todo el cubo de manera eficiente
    ds_mapa = ds.mean(dim='time', skipna=True)
    
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    
    for var in variables:
        if var in ds_mapa.variables:
            print(f"Generando mapa de la media para la variable: {var}...")
            
            fig, ax = plt.subplots(figsize=(8, 9), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            
            # Capas base idénticas al estilo de tus mapas de nicho
            ax.add_feature(cfeature.LAND, facecolor='#393731')
            ax.coastlines(color='white')
            
            # Extracción y ploteo de la variable (ya promediada en el tiempo)
            data_var = ds_mapa[var].squeeze()
            im = data_var.plot(
                ax=ax, 
                transform=ccrs.PlateCarree(), 
                cmap='viridis', 
                cbar_kwargs={'label': var, 'fraction': 0.046, 'pad': 0.04}
            )
            
            ax.set_title(f'Diagnóstico Media Climatológica ({var})', fontsize=14)
            
            # Guardar imagen individual
            ruta_out = os.path.join(carpeta_salida, f"Diagnostico_{var}.png")
            plt.savefig(ruta_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"-> Guardado en: {ruta_out}")
        else:
            print(f"Advertencia: La variable '{var}' no se encuentra en el cubo NetCDF.")

print(f"\n¡Diagnóstico completado! Todos los mapas de las variables se guardaron en: {carpeta_salida}")