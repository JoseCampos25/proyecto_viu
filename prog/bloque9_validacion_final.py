import os  # Importa el módulo del sistema operativo para gestionar rutas y directorios
import sys  # Importa el módulo sys para acceder a parámetros del sistema y argumentos de la línea de comandos
import joblib  # Importa joblib para guardar y cargar objetos de Python (como modelos de ML)
import numpy as np  # Importa NumPy para operaciones numéricas y manejo de arrays
import pandas as pd  # Importa Pandas para la manipulación y análisis de datos mediante DataFrames
import matplotlib  # Importa la librería Matplotlib para la creación de gráficos y mapas
matplotlib.use('Agg')  # Configura el backend de Matplotlib a 'Agg' para generar imágenes sin requerir una interfaz gráfica de pantalla
import matplotlib.pyplot as plt  # Importa el módulo pyplot de Matplotlib para la generación de figuras y mapas
import cartopy.crs as ccrs  # Importa Cartopy para definir y manipular sistemas de proyección cartográfica
import cartopy.feature as cfeature  # Importa características geográficas prediseñadas de Cartopy (costas, océanos, tierra)
from sklearn.ensemble import RandomForestClassifier  # Importa el clasificador Random Forest de scikit-learn
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve  # Importa métricas de evaluación para modelos de clasificación
from sklearn.model_selection import BaseCrossValidator  # Importa la clase base para crear esquemas de validación cruzada personalizados
from sklearn.inspection import permutation_importance  # <--- IMPORTACIÓN AÑADIDA: Importa la función para calcular la importancia de variables por permutación
from scipy.stats import spearmanr  # Importa la correlación de rango de Spearman utilizada para calcular el Índice de Boyce

# ------------------------------------------- RUTAS Y ENTRADA -------------------------------------------------------------
if len(sys.argv) > 1:  # Comprueba si se ha pasado al menos un argumento desde la línea de comandos
    especie_input = sys.argv[1].replace("_", " ")  # Obtiene el nombre de la especie reemplazando los guiones bajos por espacios
    print(f"Especie recibida automáticamente: {especie_input}")  # Muestra en consola la especie detectada por argumento
else:  # Si no se pasaron argumentos por la línea de comandos
    especie_input = input("Especie para análisis por bloques espaciales (ej: Megaptera novaeangliae): ").strip()  # Pide al usuario que ingrese manualmente la especie

nombre_folder = especie_input.replace(" ", "_")  # Formatea el nombre de la especie reemplazando espacios por guiones bajos para carpetas y archivos

ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))  # Obtiene la ruta absoluta del directorio donde se ubica este script
ruta_raiz = os.path.dirname(ruta_de_este_script)  # Obtiene la ruta del directorio raíz del proyecto (un nivel arriba del script)
carpeta_especies = os.path.join(ruta_raiz, "data", "carpeta_especies")  # Construye la ruta hacia la carpeta contenedora de especies dentro de /data
ruta_especie_folder = os.path.join(carpeta_especies, nombre_folder)  # Construye la ruta de la carpeta específica para la especie actual

ruta_csv = os.path.join(ruta_especie_folder, f"Dataset_presencias_ausencias_{nombre_folder}.csv")  # Define la ruta al CSV del dataset de la especie
ruta_output_mapas_dir = os.path.join(ruta_raiz, "results", "especies", nombre_folder)  # Define la ruta del directorio de salida para mapas y resultados
os.makedirs(ruta_output_mapas_dir, exist_ok=True)  # Crea la carpeta de resultados si no existe previamente
os.makedirs(ruta_especie_folder, exist_ok=True)  # Crea la carpeta del dataset de la especie si no existe previamente

ruta_output_txt_bloques = os.path.join(ruta_output_mapas_dir, f"Informe_Validacion_Bloques_{nombre_folder}.txt")  # Define la ruta del archivo TXT para el informe final

if not os.path.exists(ruta_csv):  # Comprueba si el archivo CSV del dataset existe en la ruta esperada
    print(f"Error: No existe el dataset en {ruta_csv}")  # Muestra un mensaje de error si no se encuentra el dataset
    exit()  # Detiene la ejecución del script

# ------------------------------------------- CARGA Y LIMPIEZA -------------------------------------------------------------
df = pd.read_csv(ruta_csv)  # Carga el dataset en un DataFrame de Pandas desde el archivo CSV

def limpiar_valor(x):  # Define una función para limpiar cadenas de texto formateadas como listas numéricas
    if isinstance(x, str):  # Evalúa si el valor ingresado es de tipo cadena de texto (string)
        return float(x.replace('[', '').replace(']', ''))  # Elimina corchetes de la cadena y la convierte a valor flotante
    return x  # Retorna el valor original si ya es numérico

variables_completas = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho']  # Lista las variables ambientales bio-orográficas y oceanográficas a utilizar

for var in variables_completas:  # Recorre cada una de las variables de la lista
    if var in df.columns:  # Verifica si la variable existe como columna dentro del DataFrame
        df[var] = df[var].apply(limpiar_valor)  # Aplica la función de limpieza a toda la columna para corregir su formato

if 'latitud' not in df.columns and 'decimalLatitude' in df.columns:  # Comprueba si la columna 'latitud' no existe pero sí 'decimalLatitude'
    df['latitud'] = df['decimalLatitude']  # Renombra/asigna el contenido de 'decimalLatitude' a 'latitud'
if 'longitud' not in df.columns and 'decimalLongitude' in df.columns:  # Comprueba si la columna 'longitud' no existe pero sí 'decimalLongitude'
    df['longitud'] = df['decimalLongitude']  # Renombra/asigna el contenido de 'decimalLongitude' a 'longitud'

# Reemplazamos el dropna estricto de las variables ambientales interpolando con el vecino más cercano
df[variables_completas] = df[variables_completas].interpolate(method='nearest', axis=0)  # Interpola datos faltantes en variables ambientales usando el valor vecino más cercano

# Si queda algún valor nulo en los bordes extremos, rellenamos con la media de la variable
df[variables_completas] = df[variables_completas].fillna(df[variables_completas].mean())  # Imputa cualquier nulo remanente en los extremos utilizando la media de la variable

# El dropna final solo se asegura de que existan las coordenadas y la etiqueta, sin perder filas
df = df.dropna(subset=['presencia', 'latitud', 'longitud'])  # Elimina únicamente registros donde falten las coordenadas o la variable respuesta 'presencia'

# ------------------------------------------- CONFIGURACIÓN DE LÍMITES Y BINS ----------------------------------------------
lon_min, lon_max = df['longitud'].min() - 0.5, df['longitud'].max() + 0.5  # Calcula los límites mínimo y máximo de longitud añadiendo un margen de 0.5°
lat_min, lat_max = df['latitud'].min() - 0.5, df['latitud'].max() + 0.5  # Calcula los límites mínimo y máximo de latitud añadiendo un margen de 0.5°

n_lat, n_lon = 4, 4  # Establece la división del espacio geográfico en una cuadrícula de 4x4 bloques
lat_bins = np.linspace(lat_min, lat_max, n_lat + 1)  # Genera 5 bordes latitudinales equidistantes para definir 4 bloques verticales
lon_bins = np.linspace(lon_min, lon_max, n_lon + 1)  # Genera 5 bordes longitudinales equidistantes para definir 4 bloques horizontales

# ------------------------------------------- CLASE VALIDACIÓN ESPACIAL ---------------------------------------------------
class SpatialBlockCV(BaseCrossValidator):  # Define una clase de validación cruzada por bloques espaciales heredando de BaseCrossValidator
    def __init__(self, lat_bins, lon_bins):  # Constructor de la clase que recibe los límites de los bloques
        self.lat_bins = lat_bins  # Almacena los bordes latitudinales
        self.lon_bins = lon_bins  # Almacena los bordes longitudinales
        self.n_blocks_lat = len(lat_bins) - 1  # Calcula la cantidad de bloques en latitud (4)
        self.n_blocks_lon = len(lon_bins) - 1  # Calcula la cantidad de bloques en longitud (4)

    def _iter_test_masks(self, X, y=None, groups=None):  # Genera iterativamente las máscaras de prueba (test) para cada bloque espacial
        lats = X['latitud']  # Extrae la columna de latitudes del conjunto de datos
        lons = X['longitud']  # Extrae la columna de longitudes del conjunto de datos
        
        lat_idx = np.digitize(lats, self.lat_bins) - 1  # Determina a qué índice de bin latitudinal pertenece cada punto
        lon_idx = np.digitize(lons, self.lon_bins) - 1  # Determina a qué índice de bin longitudinal pertenece cada punto
        
        lat_idx = np.clip(lat_idx, 0, self.n_blocks_lat - 1)  # Asegura que los índices latitudinales se mantengan dentro de los límites válidos
        lon_idx = np.clip(lon_idx, 0, self.n_blocks_lon - 1)  # Asegura que los índices longitudinales se mantengan dentro de los límites válidos
        
        block_ids = lat_idx * self.n_blocks_lon + lon_idx  # Mapea los índices 2D (lat, lon) a un identificador único 1D para cada bloque espacial
        unique_blocks = np.unique(block_ids)  # Obtiene la lista de bloques que contienen al menos un punto de datos
        
        for b in unique_blocks:  # Recorre cada uno de los bloques únicos identificados
            test_mask = (block_ids == b)  # Crea una máscara booleana donde es True para los datos contenidos en el bloque actual 'b'
            yield test_mask  # Retorna la máscara de prueba para la iteración actual del generador

    def get_n_splits(self, X=None, y=None, groups=None):  # Devuelve el número total de particiones (splits) del validador
        return self.n_blocks_lat * self.n_blocks_lon  # Retorna el total teórico de bloques (4 * 4 = 16)

# ------------------------------------------- MAPA GENERAL DE LA CUADRÍCULA ------------------------------------------------
print(f"Generando mapa explicativo de la cuadrícula de bloques para {especie_input.upper()}...")  # Informa al usuario en consola sobre la generación del mapa de referencia

fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})  # Inicializa la figura y ejes proyectados en coordenadas geográficas (PlateCarree)
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # Define los límites geográficos visibles del mapa
ax.add_feature(cfeature.LAND, facecolor='#393731', zorder=2)  # Añade la capa de tierra emergida con color gris oscuro
ax.add_feature(cfeature.OCEAN, facecolor='#1e1e1e', zorder=1)  # Añade la capa del océano con fondo casi negro
ax.coastlines(color='white', zorder=3)  # Dibuja la línea de costa con trazo blanco

for lat in lat_bins:  # Recorre cada coordenada latitudinal que marca un límite de bloque
    ax.hlines(lat, lon_min, lon_max, color='yellow', linestyle='--', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)  # Dibuja líneas horizontales discontinuas amarillas
for lon in lon_bins:  # Recorre cada coordenada longitudinal que marca un límite de bloque
    ax.vlines(lon, lat_min, lat_max, color='yellow', linestyle='--', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)  # Dibuja líneas verticales discontinuas amarillas

for i in range(n_lat):  # Itera sobre la dimensión latitudinal de la cuadrícula
    for j in range(n_lon):  # Itera sobre la dimensión longitudinal de la cuadrícula
        block_id = i * n_lon + j  # Calcula el identificador numérico base cero del bloque
        block_num = block_id + 1  # Define el número visible del bloque (base 1)
        c_lat = (lat_bins[i] + lat_bins[i+1]) / 2  # Obtiene la latitud central del bloque
        c_lon = (lon_bins[j] + lon_bins[j+1]) / 2  # Obtiene la longitud central del bloque
        ax.text(c_lon, c_lat, f"B{block_num}", color='cyan', fontsize=11, weight='bold',  # Dibuja la etiqueta de texto (ej: B1) en el centro del bloque
                ha='center', va='center', transform=ccrs.PlateCarree(), zorder=5,  # Ajusta alineación y proyección espacial de la etiqueta
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='yellow', pad=3))  # Agrega una caja con fondo negro y borde amarillo para destacar el texto

ax.set_title(f'Cuadrícula de Referencia - {especie_input}')  # Agrega el título del mapa con el nombre de la especie
ruta_mapa_grid = os.path.join(ruta_output_mapas_dir, f"Mapa_Cuadricula_Bloques_{nombre_folder}.png")  # Define la ruta de salida de la imagen del mapa de cuadrícula
plt.savefig(ruta_mapa_grid, dpi=300, bbox_inches='tight')  # Guarda el mapa de la cuadrícula con alta resolución (300 DPI)
plt.close()  # Cierra la figura para liberar memoria RAM

# ------------------------------------------- ENTRENAMIENTO Y MÉTRICAS POR FOLD --------------------------------------------
print(f"Iniciando entrenamiento y validación espacial avanzada para {especie_input.upper()}...")  # Notifica el inicio del bucle de validación cruzada espacial

X = df[variables_completas]  # Asigna al conjunto de predictores X únicamente las columnas de variables ambientales
X_espacial = df[['latitud', 'longitud']]  # Asigna al DataFrame X_espacial las coordenadas geográficas para coordinar el particionado
y = df['presencia']  # Asigna la variable objetivo binaria y (1: presencia, 0: pseudoausencia)

cv_espacial = SpatialBlockCV(lat_bins, lon_bins)  # Instancia la clase de validación cruzada por bloques espaciales creada previamente

aucs_completos = []  # Inicializa la lista para guardar los valores de AUC de cada fold
boyce_completos = []  # Inicializa la lista para guardar los valores del Índice de Boyce de cada fold
tss_completos = []  # Inicializa la lista para guardar los valores de TSS de cada fold
sens_completos = []  # Inicializa la lista para guardar la sensibilidad de cada fold
spec_completos = []  # Inicializa la lista para guardar la especificidad de cada fold
perm_importances_folds = []  # <--- LISTA PARA ACUMULAR IMPORTANCIAS POR PERMUTACIÓN: Almacenará la importancia de variables calculada en cada iteración

mejor_modelo = None  # Variable para mantener guardada la instancia del modelo con mejor desempeño
mejor_auc = -1  # Variable inicializada en -1 para realizar el seguimiento del mayor AUC alcanzado
lineas_bloques = []  # Lista para recopilar el texto que conformará el informe final de salida

lats_arr = X_espacial['latitud'].values  # Convierte la columna latitud a un array de NumPy
lons_arr = X_espacial['longitud'].values  # Convierte la columna longitud a un array de NumPy
lat_idx_arr = np.clip(np.digitize(lats_arr, lat_bins) - 1, 0, n_lat - 1)  # Mapea cada registro a su índice de bloque latitudinal
lon_idx_arr = np.clip(np.digitize(lons_arr, lon_bins) - 1, 0, n_lon - 1)  # Mapea cada registro a su índice de bloque longitudinal
block_ids_arr = lat_idx_arr * n_lon + lon_idx_arr  # Array completo con el ID de bloque asignado a cada fila del dataset

for fold_idx, (train_index, test_index) in enumerate(cv_espacial.split(X_espacial)):  # Recorre iterativamente las particiones geográficas de entrenamiento y prueba
    X_train = X.iloc[train_index]  # Selecciona las características ambientales de entrenamiento para el fold actual
    X_test = X.iloc[test_index]  # Selecciona las características ambientales de prueba (bloque fuera de muestra)
    y_train = y.iloc[train_index]  # Selecciona las etiquetas de presencia/ausencia de entrenamiento
    y_test = y.iloc[test_index]  # Selecciona las etiquetas de presencia/ausencia de prueba
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:  # Omite la evaluación del fold si en train o test solo existe una única clase (falta de variabilidad)
        continue  # Salta a la siguiente iteración del bucle
        
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)  # Instancia un modelo Random Forest con 100 árboles de decisión y semilla fija
    modelo.fit(X_train, y_train)  # Entrena el clasificador Random Forest utilizando exclusivamente los datos del bloque de entrenamiento
    
    # --- CÁLCULO DE UMBRAL ÓPTIMO (Maximización del TSS / Youden's Index en el conjunto de entrenamiento) ---
    y_train_proba = modelo.predict_proba(X_train)[:, 1]  # Predice las probabilidades de idoneidad/presencia para el conjunto de entrenamiento
    fpr, tpr, thresholds = roc_curve(y_train, y_train_proba)  # Calcula las tasas de falsos positivos, verdaderos positivos y sus umbrales asociados
    best_threshold = thresholds[np.argmax(tpr - fpr)]  # Encuentra el umbral que maximiza el Índice de Youden (TPR - FPR) en el conjunto de entrenamiento
    
    y_proba = modelo.predict_proba(X_test)[:, 1]  # Predice las probabilidades de idoneidad para el bloque de prueba fuera de muestra
    
    # 1. AUC
    score_auc = roc_auc_score(y_test, y_proba)  # Calcula el área bajo la curva ROC (AUC Espacial) para el conjunto de prueba
    aucs_completos.append(score_auc)  # Añade el AUC de este fold a la lista global
    
    # Tamaños de muestra ($N$)
    n_avistamientos = int((y_test == 1).sum())  # Cuenta la cantidad real de registros de presencia en el bloque de prueba
    n_ausencias = int((y_test == 0).sum())  # Cuenta la cantidad de registros de pseudoausencia/fondo en el bloque de prueba
    n_total = n_avistamientos + n_ausencias  # Calcula el tamaño total de la muestra dentro del bloque de prueba
    
    # 2. Sensibilidad, Especificidad y TSS (Usando el umbral optimizado)
    y_pred_bin = (y_proba >= best_threshold).astype(int)  # Binariza las probabilidades predichas según el umbral óptimo identificado
    cm = confusion_matrix(y_test, y_pred_bin, labels=[0, 1])  # Genera la matriz de confusión comparando valores reales y binarios predichos
    tn, fp, fn, tp = cm.ravel()  # Descompone la matriz de confusión en verdaderos negativos, falsos positivos, falsos negativos y verdaderos positivos
    
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Calcula la Sensibilidad (tasa de verdaderos positivos) evitando divisiones por cero
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Calcula la Especificidad (tasa de verdaderos negativos) evitando divisiones por cero
    tss = sensibilidad + especificidad - 1.0  # Calcula la estadística TSS (True Skill Statistic)
    
    sens_completos.append(sensibilidad)  # Añade la sensibilidad de este fold a la lista global
    spec_completos.append(especificidad)  # Añade la especificidad de este fold a la lista global
    tss_completos.append(tss)  # Añade el valor de TSS de este fold a la lista global
    
    # 3. Índice de Boyce aproximado (Spearman rank correlation of presences)
    pred_presencias = y_proba[y_test == 1]  # Filtra únicamente las probabilidades predichas para las presencias reales en el test
    if len(pred_presencias) > 2:  # Evalúa si se tienen más de 2 presencias para poder calcular la correlación
        boyce_val, _ = spearmanr(pred_presencias, np.sort(pred_presencias))  # Calcula la correlación de Spearman ordenada como una aproximación del Índice de Boyce
        boyce_val = boyce_val if not np.isnan(boyce_val) else 0.0  # Asigna 0.0 si el valor resultante es indeterminado (NaN)
    else:  # Si no hay suficientes presencias en el bloque
        boyce_val = 0.0  # Asigna 0.0 al valor de Boyce
    boyce_completos.append(boyce_val)  # Guarda el valor de Boyce en la lista global
    
    # --- CÁLCULO DE IMPORTANCIA POR PERMUTACIÓN EN EL FOLD ESPACIAL (TEST) ---
    result_perm = permutation_importance(  # Evalúa la degradación del AUC permutando cada variable en los datos no vistos del bloque de prueba
        modelo, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42, n_jobs=-1  # Realiza 10 repeticiones por variable utilizando procesamiento en paralelo
    )  # Almacena el objeto con los resultados del análisis de permutación
    perm_importances_folds.append(result_perm.importances_mean)  # Añade el promedio de importancia por permutación del fold a la lista
    
    if score_auc > mejor_auc:  # Verifica si el modelo entrenado en este fold supera el récord del mejor AUC registrado
        mejor_auc = score_auc  # Actualiza la marca del mejor AUC
        mejor_modelo = modelo  # Guarda la referencia a este modelo entrenado como el mejor del ensamble

    block_id_test = np.unique(block_ids_arr[test_index])[0]  # Recupera el identificador único del bloque evaluado en la iteración actual
    block_num_real = int(block_id_test) + 1  # Convierte el identificador a base 1 para una lectura humana amigable

    block_txt = f"\n==================================================\n"  # Formatea encabezado visual para el bloque
    block_txt += f"  BLOQUE B{block_num_real} (Fold {fold_idx + 1}) | N Total Test: {n_total}\n"  # Agrega nombre del bloque y tamaño de muestra test
    block_txt += f"==================================================\n"  # Formatea separador
    block_txt += f"{'Métrica Evaluada':<35} | {'Valor':<10}\n"  # Define encabezados de tabla para métricas
    block_txt += "-" * 50 + f"\n"  # Agrega línea divisoria
    block_txt += f"{'Avistamientos (Presencias) [N]':<35} | {n_avistamientos:<10}\n"  # Registra el número de presencias en la prueba
    block_txt += f"{'Ausencias / Fondo [N]':<35} | {n_ausencias:<10}\n"  # Registra el número de pseudoausencias en la prueba
    block_txt += f"{'Umbral Óptimo (Max TSS)':<35} | {best_threshold:.4f}\n"  # Registra el umbral de corte óptimo derivado de entrenamiento
    block_txt += f"{'AUC Espacial':<35} | {score_auc:.4f}\n"  # Registra la métrica AUC fuera de muestra
    block_txt += f"{'Índice de Boyce':<35} | {boyce_val:.4f}\n"  # Registra el Índice de Boyce en la prueba
    block_txt += f"{'TSS (True Skill Statistic)':<35} | {tss:.4f}\n"  # Registra la métrica TSS
    block_txt += f"{'Sensibilidad':<35} | {sensibilidad:.4f}\n"  # Registra la tasa de verdaderos positivos (Sensibilidad)
    block_txt += f"{'Especificidad':<35} | {especificidad:.4f}\n"  # Registra la tasa de verdaderos negativos (Especificidad)
    block_txt += f"==================================================\n"  # Cierra la sección de texto del bloque

    print(block_txt)  # Imprime la tabla de métricas del bloque actual en la consola
    lineas_bloques.append(block_txt)  # Almacena el bloque de texto para la escritura del informe final en disco

    # --- MAPA POR FOLD ---
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})  # Crea una nueva figura cartográfica para el fold actual
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # Establece la extensión geográfica del mapa
    ax.add_feature(cfeature.LAND, facecolor='#393731', zorder=2)  # Añade la masa continental en gris oscuro
    ax.add_feature(cfeature.OCEAN, facecolor='#1e1e1e', zorder=1)  # Añade la capa marina en fondo oscuro
    ax.coastlines(color='white', zorder=3)  # Agrega la línea de costa en color blanco

    for lat in lat_bins:  # Recorre la lista de latitudes para la cuadrícula
        ax.hlines(lat, lon_min, lon_max, color='yellow', linestyle='--', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)  # Dibuja divisiones horizontales de bloques
    for lon in lon_bins:  # Recorre la lista de longitudes para la cuadrícula
        ax.vlines(lon, lat_min, lat_max, color='yellow', linestyle='--', linewidth=1, transform=ccrs.PlateCarree(), zorder=4)  # Dibuja divisiones verticales de bloques

    ax.scatter(df.iloc[train_index]['longitud'], df.iloc[train_index]['latitud'], c='royalblue', s=15, alpha=0.6, label='Entrenamiento', transform=ccrs.PlateCarree(), zorder=5)  # Representa espacialmente las presencias/ausencias usadas para entrenar en azul
    ax.scatter(df.iloc[test_index]['longitud'], df.iloc[test_index]['latitud'], c='crimson', s=30, alpha=0.9, label=f'Prueba / Test (B{block_num_real})', transform=ccrs.PlateCarree(), zorder=6)  # Representa espacialmente el bloque de prueba dejado fuera en carmesí

    ax.legend(loc='lower left')  # Posiciona la leyenda en la esquina inferior izquierda
    ax.set_title(f'Validación Espacial - B{block_num_real} (AUC: {score_auc:.3f})')  # Asigna título al mapa mostrando el bloque y su AUC alcanzado

    ruta_mapa_fold = os.path.join(ruta_output_mapas_dir, f"Mapa_Bloque_B{block_num_real}_{nombre_folder}.png")  # Define el nombre de archivo para el mapa del fold actual
    plt.savefig(ruta_mapa_fold, dpi=300, bbox_inches='tight')  # Exporta la figura a un archivo PNG de alta resolución
    plt.close()  # Cierra la figura para liberar la memoria del sistema

# --- PROMEDIO DE IMPORTANCIA POR PERMUTACIÓN SOBRE TODOS LOS FOLDS ---
mean_perm_imp = np.mean(perm_importances_folds, axis=0)  # Calcula la media de la importancia por permutación de cada variable a través de todos los folds
std_perm_imp = np.std(perm_importances_folds, axis=0)  # Calcula la desviación estándar de la importancia por permutación a través de todos los folds

df_perm_imp = pd.DataFrame({  # Estructura los resultados de la importancia por permutación en un DataFrame de Pandas
    'Variable': variables_completas,  # Asigna los nombres de las 8 variables ambientales
    'Importancia_Media': mean_perm_imp,  # Asigna los valores promedios de importancia calculados
    'Desviacion_Std': std_perm_imp  # Asigna las desviaciones estándar correspondientes
}).sort_values(by='Importancia_Media', ascending=False)  # Ordena las variables de mayor a menor según su importancia predictiva relativa

# Resumen Global Final con desglose de N real, pseudoausencias e Importancia por Permutación
n_presencias_total = int((df['presencia'] == 1).sum())  # Calcula el número total de avistamientos/presencias reales en todo el dataset de la especie
n_ausencias_total = int((df['presencia'] == 0).sum())  # Calcula el número total de puntos de pseudoausencia/fondo en el dataset

res_blocks_final = f"\n==================================================\n"  # Encabezado del bloque de resumen global
res_blocks_final += f" RESUMEN GLOBAL ESPACIAL PARA: {especie_input.upper()}\n"  # Muestra el nombre de la especie analizada en mayúsculas
res_blocks_final += f"==================================================\n"  # Separador visual
res_blocks_final += f" N Presencias reales (Avistamientos): {n_presencias_total}\n"  # Añade al texto el recuento global de presencias
res_blocks_final += f" N Pseudoausencias / Fondo:          {n_ausencias_total}\n"  # Añade al texto el recuento global de pseudoausencias
res_blocks_final += f" AUC Medio Espacial:                 {np.mean(aucs_completos):.4f} (±{np.std(aucs_completos):.4f})\n"  # Calcula y formatea la media y desviación estándar global del AUC
res_blocks_final += f" Boyce Medio:                        {np.mean(boyce_completos):.4f} (±{np.std(boyce_completos):.4f})\n"  # Calcula y formatea la media y desviación estándar global de Boyce
res_blocks_final += f" TSS Medio:                          {np.mean(tss_completos):.4f} (±{np.std(tss_completos):.4f})\n"  # Calcula y formatea la media y desviación estándar global de TSS
res_blocks_final += f" Sensibilidad Media:                 {np.mean(sens_completos):.4f}\n"  # Calcula y formatea la Sensibilidad media a través de los folds
res_blocks_final += f" Especificidad Media:                {np.mean(spec_completos):.4f}\n"  # Calcula y formatea la Especificidad media a través de los folds
res_blocks_final += f"--------------------------------------------------\n"  # Encabezado de la sección de importancia de variables
res_blocks_final += f" IMPORTANCIA DE VARIABLES POR PERMUTACIÓN (FOLDS ESPACIALES):\n"  # Título de la tabla de variables en el informe
res_blocks_final += f"--------------------------------------------------\n"  # Separador visual de tabla
for _, row in df_perm_imp.iterrows():  # Recorre cada variable ordenada de mayor a menor contribución
    res_blocks_final += f" {row['Variable']:<15} | Importancia: {row['Importancia_Media']:.4f} (±{row['Desviacion_Std']:.4f})\n"  # Imprime cada variable con su importancia relativa y variabilidad
res_blocks_final += f"==================================================\n"  # Cierre visual del informe final

print(res_blocks_final)  # Muestra el resumen global estructurado en la consola
lineas_bloques.append(res_blocks_final)  # Agrega la sección de resumen global al conjunto de líneas a exportar

with open(ruta_output_txt_bloques, 'w', encoding='utf-8') as f_bloques:  # Abre el archivo de salida del informe TXT en modo escritura y codificación UTF-8
    f_bloques.write("".join(lineas_bloques))  # Escribe la totalidad de las líneas concatenadas en el archivo TXT

print(f"Informe completo con métricas de tutor guardado en: {ruta_output_txt_bloques}")  # Confirma en la consola la generación exitosa del informe escrito

joblib.dump(mejor_modelo, os.path.join(ruta_output_mapas_dir, f"modelo_RF_bloques_{nombre_folder}.pkl"))  # Exporta e independiza en formato .pkl el mejor modelo Random Forest obtenido