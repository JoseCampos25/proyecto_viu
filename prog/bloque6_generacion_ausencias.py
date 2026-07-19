######
# En este script se genera un dataset balanceado de presencias y ausencias para una especie marina. Este dataset va acombinar:
#Presencias reales (que vienen de los avistamientos)
#Ausencias simuladas (generadas artificialmente pero con condiciones ecologicos realistas)

#------------------------------------------LIBRERIAS-------------------------------------------------------------

import pandas as pd  #Para trabajar con tablas (leer csv, crear dataframe...)
import numpy as np   #lo suaremos para generar numeros aleatorios de coordenadas entre otros.
import xarray as xr #Para trabajar con el cubo ambiental que esta en formato NetCDF
import os           #Sirve para generar rutas, crear carpetas...
import rasterio     #Para abrir archivos gebco entre otros
from tqdm import tqdm #Añade una barra de progreso al bucle que genera ausencias visualiza cuantas ausencias se llevan generadas

#-------------------------------------------ENTRADA Y RUTAS-------------------------------------------------------------

especie = input("Introduce el nombre científico de la especie (ej: Balaenoptera musculus): ").strip()
nombre_carpeta = especie.replace(" ", "_")   # Creamos variable con el nombre de la carpeta

# 1. Enraizamiento dinámico del script
ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_de_este_script) # ✓ Paréntesis corregido

# 2. Configuración de carpetas usando tu estructura real de /data
carpeta_especies = os.path.join(ruta_raiz, "data", "carpeta_especies")

# 3. Rutas de Entrada y Salida de la especie
ruta_presencias = os.path.join(carpeta_especies, nombre_carpeta, f"datos_cubo_{nombre_carpeta}.csv") 
ruta_salida = os.path.join(carpeta_especies, nombre_carpeta, f"Dataset_presencias_ausencias_{nombre_carpeta}.csv")

# 4. CORREGIDO: Rutas del Cubo y de GEBCO apuntando a donde están de verdad
ruta_cubo = os.path.join(ruta_raiz, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")
ruta_gebco = os.path.join(ruta_raiz, "data", "gebco_macaronesia.tif") # ✓ Modificado según tu 'ls data/'


#-------------------------------------------APERTURA CUBO Y PRESENCIAS-------------------------------------------------------------
#Vamos a abrir el cubo ambiental y el documento de avistamientos con datos del cubo (presencias) para preparar el dataset

print("Abriendo cubo ambiental y archivo de presencias...")
ds = xr.open_dataset(ruta_cubo, chunks={'time': 1, 'latitude': 50, 'longitude': 50}) #Abre cubo ambiental sin cargarlo en memoria
df_p = pd.read_csv(ruta_presencias) #Abre el archivo donde se encuentran los avistamientos con datos del cubo
df_p['fechas'] = pd.to_datetime(df_p['fechas']) #conmvierte la columana fecha en datetime del archivo de presencias
n_objetivo = len(df_p) #calcula cuantas ausencias hay que generar (en este caso el mismo que presencuas)


#Abrimos GEBCO una vez para que sea rápido
with rasterio.open(ruta_gebco) as src_gebco: #abrimos gebco que contiene la barimetria para ver la profundidad real en cada punto simulado
    mapa_profundidad = src_gebco.read(1) #se carga la matriz de profundidades reales
    transform_gebco = src_gebco.transform #permite convertir coordenadas lat/lon → fila/columna del raster

    datos_lista = [] #lista doinde se guardaran las ausencias
    pbar = tqdm(total=n_objetivo, desc=f"Generando ausencias {nombre_carpeta}") #Se crea una barra de progreso para ver cuántas ausencias faltan por generar

#-------------------------------------------GENERACION DE AUSENCIAS-------------------------------------------------------------
    #Generamos un punto aleatorio con lat y lon aleatorios que luego pincharemos en el cubo con dicho punto para extraer los datos
    while len(datos_lista) < n_objetivo: #Repite hasta generar todas las ausencias necesaria
        ref = df_p.sample(1).iloc[0] #Toma una presencia real como referencia
        lat_sim = ref['latitud'] + np.random.uniform(-6, 6) #Crea una latitud y longitud simulada
        lon_sim = ref['longitud'] + np.random.uniform(-6, 6)
        
        f_min, f_max = df_p['fechas'].min().value, df_p['fechas'].max().value #Genera una fecha aleatoria dentro del rango real asi las ausencias respetan la estacionalidad de las especies
        ts_sim = np.random.randint(f_min, f_max, dtype=np.int64)
        fecha_sim = pd.to_datetime(ts_sim)

        #-------------------------------------------PINCHAMOS CUBO-------------------------------------------------------------
        #Vamos a pinchar el cubo con los datos aleatorios generados de las ausencias para extraer la informacion para las ausencias
        
        try: #empieza el bloque de prueba de pinchat en el cubo
            punto = ds.sel(latitude=lat_sim, longitude=lon_sim, time=fecha_sim, method='nearest').load() #pinchamos el cubo con la lon, lat y fecha aleatorias
            if 'depth' in punto.dims: punto = punto.isel(depth=0) #seleccion de la capa superficial de profundidad
            
            temp = float(punto.thetao.values.flatten()[0]) #extraccion de la variable temperatura como control


            #-------------------------------------------COMPROBACION DEL DATO DE TEMP-------------------------------------------------------------
            #Comprobacion del dato de temperatura extraido, si hay dato se sigue si no, se descarta           
            if not np.isnan(temp): #Se comprueba si la temperatura es validad, no es NAN
                registro = {'fechas': fecha_sim.strftime('%Y-%m-%d'), 'latitud': lat_sim, 'longitud': lon_sim, 'presencia': 0} #Se crea un diccionario que será el registro final que acabará en el CSV
                
                ##-------------------------------------------EEXTRACCION DE VARIABLES DEL CUBO-------------------------------------------------------------
                # Metemos todas las variables del cubo (temperatura, clorofila, etc.) en el diccionario
                for var in ds.data_vars: #Recorre todas las variables del cubo ambiental 
                    registro[var] = float(punto[var].values.flatten()[0]) #Lo anade al diccionario
                
                #CORRECCIÓN GEBCO: Sobrescribimos 'deptho' con el dato real
                row, col = src_gebco.index(lon_sim, lat_sim) #Convierte coordenadas geográficas a píxel de raster
                prof_real = abs(float(mapa_profundidad[row, col])) #- Lee la profundidad real en ese píxel.
                registro['deptho'] = prof_real #- Guarda esa profundidad real en el registro.

                
                datos_lista.append(registro) #Añade el registro completo (todas las variables + profundidad real + coordenadas + fecha) a la lista de ausencias.
                pbar.update(1) #Actualiza la barra de progreso para mostrar que se ha generado una ausencia válida más.
                
        except Exception:  #Si algo falla (punto fuera del cubo, GEBCO sin datos, etc.), simplemente se ignora y se genera otro punto
            continue

    pbar.close()  #Cierra la barra de progreso al terminar.

 #-------------------------------------------GUARDADO-------------------------------------------------------------
df_a = pd.DataFrame(datos_lista) #- Convierte la lista de diccionarios datos_lista (todas las ausencias simuladas) en un DataFrame de pandas.
df_p['presencia'] = 1 #- Asegura que el DataFrame de presencias reales tenga la columna presencia = 1
df_final = pd.concat([df_p, df_a], ignore_index=True) #Une presencias reales () y ausencias simuladas () en un único DataFrame.

  #-------------------------------------------LIMPIEZA-------------------------------------------------------------
if 'depth' in df_final.columns: df_final = df_final.drop(columns=['depth']) # Limpieza final: si el cubo traía una columna 'depth' vieja, la quitamos para no confundir

#-------------------------------------------LIMPIEZA-------------------------------------------------------------
df_final.to_csv(ruta_salida, index=False) #- Guarda el dataset final (presencias + ausencias + variables ambientales + profundidad real) en un archivo CSV.
print(f"¡Conseguido! El dataset para {especie} ya tiene ausencias y presencias.")