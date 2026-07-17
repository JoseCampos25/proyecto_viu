import xarray as xr
import numpy as np
import joblib
import os
from PIL import Image

# 1. Configuración
especie = input("Nombre de la especie: ").strip()
nombre_folder = especie.replace(" ", "_")
ruta_base = r"C:\Users\joslo\OneDrive\Desktop\proyecto_viu"
ruta_modelo = os.path.join(ruta_base, "data", "carpeta_especies", nombre_folder, f"modelo_RF_{nombre_folder}.pkl")
ruta_cubo = os.path.join(ruta_base, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")
ruta_output = os.path.join(ruta_base, "results", "mapas", f"Mapa_Nicho_{nombre_folder}.png")
os.makedirs(os.path.dirname(ruta_output), exist_ok=True)

# 2. Carga y Predicción
modelo = joblib.load(ruta_modelo)
with xr.open_dataset(ruta_cubo) as ds:
    ds_mapa = ds.isel(time=0)
    # Identificar océano
    mask = ds_mapa['deptho'].notnull().squeeze()
    # Preparar datos
    variables = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho']
    datos = np.column_stack([ds_mapa[v].squeeze().values[mask.values] for v in variables])
    # Predecir
    prob = modelo.predict_proba(datos)[:, 1]
    # Crear grid
    grid = np.zeros(mask.shape)
    grid[mask.values] = prob
    # Invertir para visualización (geográfico)
    grid = np.flipud(grid)
    tierra = np.flipud(ds_mapa['deptho'].isnull().squeeze().values)

# 3. Renderizado Píxel a Píxel
print("🚀 Dibujando mapa...")
h, w = grid.shape
img_data = np.zeros((h, w, 3), dtype=np.uint8)

# Colores (Magma)
for r in range(h):
    for c in range(w):
        if tierra[r, c]:
            img_data[r, c] = [57, 55, 49]  # Gris Tierra
        elif grid[r, c] >= 0.3:
            v = (grid[r, c] - 0.3) / 0.7
            img_data[r, c] = [min(255, int(40 + v * 215)), min(255, int(10 + v * 200)), min(255, int(70 - v * 30))]
        else:
            img_data[r, c] = [1, 5, 10]    # Océano profundo

# Crear imagen, escalar y guardar
img = Image.fromarray(img_data)
img = img.resize((w * 4, h * 4), Image.Resampling.LANCZOS)
img.save(ruta_output)

print(f"✅ Mapa generado correctamente en: {ruta_output}")