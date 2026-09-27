import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import BaseCrossValidator
from sklearn.inspection import permutation_importance  # <--- IMPORTACIÓN AÑADIDA
from scipy.stats import spearmanr

# ------------------------------------------- RUTAS Y ENTRADA -------------------------------------------------------------
if len(sys.argv) > 1:
    especie_input = sys.argv[1].replace("_", " ")
    print(f"Especie recibida automáticamente: {especie_input}")
else:
    especie_input = input("Especie para análisis por bloques espaciales (ej: Megaptera novaeangliae): ").strip()

nombre_folder = especie_input.replace(" ", "_")

ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_de_este_script)
carpeta_especies = os.path.join(ruta_raiz, "data", "carpeta_especies")
ruta_especie_folder = os.path.join(carpeta_especies, nombre_folder)

ruta_csv = os.path.join(ruta_especie_folder, f"Dataset_presencias_ausencias_{nombre_folder}.csv")
ruta_output_mapas_dir = os.path.join(ruta_raiz, "results", "especies", nombre_folder)
os.makedirs(ruta_output_mapas_dir, exist_ok=True)
os.makedirs(ruta_especie_folder, exist_ok=True)

ruta_output_txt_bloques = os.path.join(ruta_output_mapas_dir, f"Informe_Validacion_Bloques_{nombre_folder}.txt")

if not os.path.exists(ruta_csv):
    print(f"Error: No existe el dataset en {ruta_csv}")
    exit()

# ------------------------------------------- CARGA Y LIMPIEZA -------------------------------------------------------------
df = pd.read_csv(ruta_csv)

def limpiar_valor(x):
    if isinstance(x, str):
        return float(x.replace('[', '').replace(']', ''))
    return x

variables_completas = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho']

for var in variables_completas:
    if var in df.columns:
        df[var] = df[var].apply(limpiar_valor)

if 'latitud' not in df.columns and 'decimalLatitude' in df.columns:
    df['latitud'] = df['decimalLatitude']
if 'longitud' not in df.columns and 'decimalLongitude' in df.columns:
    df['longitud'] = df['decimalLongitude']

# Reemplazamos el dropna estricto de las variables ambientales interpolando con el vecino más cercano
df[variables_completas] = df[variables_completas].interpolate(method='nearest', axis=0)

# Si queda algún valor nulo en los bordes extremos, rellenamos con la media de la variable
df[variables_completas] = df[variables_completas].fillna(df[variables_completas].mean())

# El dropna final solo se asegura de que existan las coordenadas y la etiqueta, sin perder filas
df = df.dropna(subset=['presencia', 'latitud', 'longitud'])

# ------------------------------------------- CONFIGURACIÓN DE LÍMITES Y BINS ----------------------------------------------
lon_min, lon_max = df['longitud'].min() - 0.5, df['longitud'].max() + 0.5
lat_min, lat_max = df['latitud'].min() - 0.5, df['latitud'].max() + 0.5

n_lat, n_lon = 4, 4
lat_bins = np.linspace(lat_min, lat_max, n_lat + 1)
lon_bins = np.linspace(lon_min, lon_max, n_lon + 1)

# ------------------------------------------- CLASE VALIDACIÓN ESPACIAL ---------------------------------------------------
class SpatialBlockCV(BaseCrossValidator):
    def __init__(self, lat_bins, lon_bins):
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
        self.n_blocks_lat = len(lat_bins) - 1
        self.n_blocks_lon = len(lon_bins) - 1

    def _iter_test_masks(self, X, y=None, groups=None):
        lats = X['latitud']
        lons = X['longitud']
        
        lat_idx = np.digitize(lats, self.lat_bins) - 1
        lon_idx = np.digitize(lons, self.lon_bins) - 1
        
        lat_idx = np.clip(lat_idx, 0, self.n_blocks_lat - 1)
        lon_idx = np.clip(lon_idx, 0, self.n_blocks_lon - 1)
        
        block_ids = lat_idx * self.n_blocks_lon + lon_idx
        unique_blocks = np.unique(block_ids)
        
        for b in unique_blocks:
            test_mask = (block_ids == b)
            yield test_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_blocks_lat * self.n_blocks_lon

# ------------------------------------------- MAPA GENERAL DE LA CUADRÍCULA ------------------------------------------------
print(f"Generando mapa explicativo de la cuadrícula de bloques para {especie_input.upper()}...")

fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='#393731', zorder=2)
ax.add_feature(cfeature.OCEAN, facecolor='#1e1e1e', zorder=1)
ax.coastlines(color='white', zorder=3)

for lat in lat_bins:
    ax.hlines(lat, lon_min, lon_max, color='yellow', linestyle='--', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)
for lon in lon_bins:
    ax.vlines(lon, lat_min, lat_max, color='yellow', linestyle='--', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)

for i in range(n_lat):
    for j in range(n_lon):
        block_id = i * n_lon + j
        block_num = block_id + 1
        c_lat = (lat_bins[i] + lat_bins[i+1]) / 2
        c_lon = (lon_bins[j] + lon_bins[j+1]) / 2
        ax.text(c_lon, c_lat, f"B{block_num}", color='cyan', fontsize=11, weight='bold',
                ha='center', va='center', transform=ccrs.PlateCarree(), zorder=5,
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='yellow', pad=3))

ax.set_title(f'Cuadrícula de Referencia - {especie_input}')
ruta_mapa_grid = os.path.join(ruta_output_mapas_dir, f"Mapa_Cuadricula_Bloques_{nombre_folder}.png")
plt.savefig(ruta_mapa_grid, dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------- ENTRENAMIENTO Y MÉTRICAS POR FOLD --------------------------------------------
print(f"Iniciando entrenamiento y validación espacial avanzada para {especie_input.upper()}...")

X = df[variables_completas]
X_espacial = df[['latitud', 'longitud']]
y = df['presencia']

cv_espacial = SpatialBlockCV(lat_bins, lon_bins)

aucs_completos = []
boyce_completos = []
tss_completos = []
sens_completos = []
spec_completos = []
perm_importances_folds = []  # <--- LISTA PARA ACUMULAR IMPORTANCIAS POR PERMUTACIÓN

mejor_modelo = None
mejor_auc = -1
lineas_bloques = []

lats_arr = X_espacial['latitud'].values
lons_arr = X_espacial['longitud'].values
lat_idx_arr = np.clip(np.digitize(lats_arr, lat_bins) - 1, 0, n_lat - 1)
lon_idx_arr = np.clip(np.digitize(lons_arr, lon_bins) - 1, 0, n_lon - 1)
block_ids_arr = lat_idx_arr * n_lon + lon_idx_arr

for fold_idx, (train_index, test_index) in enumerate(cv_espacial.split(X_espacial)):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        continue
        
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # --- CÁLCULO DE UMBRAL ÓPTIMO (Maximización del TSS / Youden's Index en el conjunto de entrenamiento) ---
    y_train_proba = modelo.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_train_proba)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    y_proba = modelo.predict_proba(X_test)[:, 1]
    
    # 1. AUC
    score_auc = roc_auc_score(y_test, y_proba)
    aucs_completos.append(score_auc)
    
    # Tamaños de muestra ($N$)
    n_avistamientos = int((y_test == 1).sum())
    n_ausencias = int((y_test == 0).sum())
    n_total = n_avistamientos + n_ausencias
    
    # 2. Sensibilidad, Especificidad y TSS (Usando el umbral optimizado)
    y_pred_bin = (y_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tss = sensibilidad + especificidad - 1.0
    
    sens_completos.append(sensibilidad)
    spec_completos.append(especificidad)
    tss_completos.append(tss)
    
    # 3. Índice de Boyce aproximado (Spearman rank correlation of presences)
    pred_presencias = y_proba[y_test == 1]
    if len(pred_presencias) > 2:
        boyce_val, _ = spearmanr(pred_presencias, np.sort(pred_presencias))
        boyce_val = boyce_val if not np.isnan(boyce_val) else 0.0
    else:
        boyce_val = 0.0
    boyce_completos.append(boyce_val)
    
    # --- CÁLCULO DE IMPORTANCIA POR PERMUTACIÓN EN EL FOLD ESPACIAL (TEST) ---
    result_perm = permutation_importance(
        modelo, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_importances_folds.append(result_perm.importances_mean)
    
    if score_auc > mejor_auc:
        mejor_auc = score_auc
        mejor_modelo = modelo

    block_id_test = np.unique(block_ids_arr[test_index])[0]
    block_num_real = int(block_id_test) + 1

    block_txt = f"\n==================================================\n"
    block_txt += f"  BLOQUE B{block_num_real} (Fold {fold_idx + 1}) | N Total Test: {n_total}\n"
    block_txt += f"==================================================\n"
    block_txt += f"{'Métrica Evaluada':<35} | {'Valor':<10}\n"
    block_txt += "-" * 50 + f"\n"
    block_txt += f"{'Avistamientos (Presencias) [N]':<35} | {n_avistamientos:<10}\n"
    block_txt += f"{'Ausencias / Fondo [N]':<35} | {n_ausencias:<10}\n"
    block_txt += f"{'Umbral Óptimo (Max TSS)':<35} | {best_threshold:.4f}\n"
    block_txt += f"{'AUC Espacial':<35} | {score_auc:.4f}\n"
    block_txt += f"{'Índice de Boyce':<35} | {boyce_val:.4f}\n"
    block_txt += f"{'TSS (True Skill Statistic)':<35} | {tss:.4f}\n"
    block_txt += f"{'Sensibilidad':<35} | {sensibilidad:.4f}\n"
    block_txt += f"{'Especificidad':<35} | {especificidad:.4f}\n"
    block_txt += f"==================================================\n"

    print(block_txt)
    lineas_bloques.append(block_txt)

    # --- MAPA POR FOLD ---
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#393731', zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor='#1e1e1e', zorder=1)
    ax.coastlines(color='white', zorder=3)

    for lat in lat_bins:
        ax.hlines(lat, lon_min, lon_max, color='yellow', linestyle='--', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)
    for lon in lon_bins:
        ax.vlines(lon, lat_min, lat_max, color='yellow', linestyle='--', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)

    ax.scatter(df.iloc[train_index]['longitud'], df.iloc[train_index]['latitud'], c='royalblue', s=15, alpha=0.6, label='Entrenamiento', transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(df.iloc[test_index]['longitud'], df.iloc[test_index]['latitud'], c='crimson', s=30, alpha=0.9, label=f'Prueba / Test (B{block_num_real})', transform=ccrs.PlateCarree(), zorder=6)

    ax.legend(loc='lower left')
    ax.set_title(f'Validación Espacial - B{block_num_real} (AUC: {score_auc:.3f})')

    ruta_mapa_fold = os.path.join(ruta_output_mapas_dir, f"Mapa_Bloque_B{block_num_real}_{nombre_folder}.png")
    plt.savefig(ruta_mapa_fold, dpi=300, bbox_inches='tight')
    plt.close()

# --- PROMEDIO DE IMPORTANCIA POR PERMUTACIÓN SOBRE TODOS LOS FOLDS ---
mean_perm_imp = np.mean(perm_importances_folds, axis=0)
std_perm_imp = np.std(perm_importances_folds, axis=0)

df_perm_imp = pd.DataFrame({
    'Variable': variables_completas,
    'Importancia_Media': mean_perm_imp,
    'Desviacion_Std': std_perm_imp
}).sort_values(by='Importancia_Media', ascending=False)

# Resumen Global Final con desglose de N real, pseudoausencias e Importancia por Permutación
n_presencias_total = int((df['presencia'] == 1).sum())
n_ausencias_total = int((df['presencia'] == 0).sum())

res_blocks_final = f"\n==================================================\n"
res_blocks_final += f" RESUMEN GLOBAL ESPACIAL PARA: {especie_input.upper()}\n"
res_blocks_final += f"==================================================\n"
res_blocks_final += f" N Presencias reales (Avistamientos): {n_presencias_total}\n"
res_blocks_final += f" N Pseudoausencias / Fondo:          {n_ausencias_total}\n"
res_blocks_final += f" AUC Medio Espacial:                 {np.mean(aucs_completos):.4f} (±{np.std(aucs_completos):.4f})\n"
res_blocks_final += f" Boyce Medio:                        {np.mean(boyce_completos):.4f} (±{np.std(boyce_completos):.4f})\n"
res_blocks_final += f" TSS Medio:                          {np.mean(tss_completos):.4f} (±{np.std(tss_completos):.4f})\n"
res_blocks_final += f" Sensibilidad Media:                 {np.mean(sens_completos):.4f}\n"
res_blocks_final += f" Especificidad Media:                {np.mean(spec_completos):.4f}\n"
res_blocks_final += f"--------------------------------------------------\n"
res_blocks_final += f" IMPORTANCIA DE VARIABLES POR PERMUTACIÓN (FOLDS ESPACIALES):\n"
res_blocks_final += f"--------------------------------------------------\n"
for _, row in df_perm_imp.iterrows():
    res_blocks_final += f" {row['Variable']:<15} | Importancia: {row['Importancia_Media']:.4f} (±{row['Desviacion_Std']:.4f})\n"
res_blocks_final += f"==================================================\n"

print(res_blocks_final)
lineas_bloques.append(res_blocks_final)

with open(ruta_output_txt_bloques, 'w', encoding='utf-8') as f_bloques:
    f_bloques.write("".join(lineas_bloques))

print(f"Informe completo con métricas de tutor guardado en: {ruta_output_txt_bloques}")

joblib.dump(mejor_modelo, os.path.join(ruta_output_mapas_dir, f"modelo_RF_bloques_{nombre_folder}.pkl"))