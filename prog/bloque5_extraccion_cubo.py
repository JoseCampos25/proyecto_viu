# Este script "Pincha" el cubo ambiental en cada punto de avistamiento y extrae las condiciones del océano en ese lugar y fecha.

# -----------------------------------------------LIBRERIAS-------------------------------------------
import xarray as xr  # Es la herramienta que permite abrir y manipular el cubo ambiental NetCDF
import pandas as pd   # Es la librería que maneja tablas y CSVs.
import os       # Es una librería del sistema operativo, construye rutas, comprueba si el csv de entrada existe...
import rasterio # IMPORTANTE: Para abrir el archivo ráster de GEBCO
import sys
# -------------------------------------------------RUTAS----------------------------------------------
if len(sys.argv) > 1:
    especie = sys.argv[1].replace("_", " ")
    print(f"Especie recibida automáticamente: {especie}")
else:
    especie = input("Introduce el nombre científico de la especie: ").strip()
    
nombre_carpeta_especie = especie.replace(" ", "_")      # Crea un nombre de carpeta para después construir las rutas.

ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_de_este_script)
carpeta_especies = os.path.join(ruta_raiz, "data", "carpeta_especies")
ruta_cubo = os.path.join(ruta_raiz, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")  # Ruta cubo ambiental
ruta_gebco = os.path.join(ruta_raiz, "data", "gebco_macaronesia.tif") # RUTA DE GEBCO

ruta_csv_entrada = os.path.join(carpeta_especies, nombre_carpeta_especie, f"avistamientos_{nombre_carpeta_especie}.csv") 
ruta_csv_salida = os.path.join(carpeta_especies, nombre_carpeta_especie, f"datos_cubo_{nombre_carpeta_especie}.csv") 

# -------------------------------------------------EXTRACCION DE VARIABLES----------------------------------------------
def pinchar_cubo():
    if not os.path.exists(ruta_csv_entrada): 
        print(f"No se encontró el CSV de avistamientos en: {ruta_csv_entrada}")
        return 

    print(f"Cargando cubo ambiental, GEBCO y avistamientos de {especie}...") 

    # Abre el cubo, avistamientos y GEBCO
    ds = xr.open_dataset(ruta_cubo) 
    df_obis = pd.read_csv(ruta_csv_entrada) 
    df_obis['fechas'] = pd.to_datetime(df_obis['fechas'])

    lista_ambiental = []

    print("Extrayendo datos ambientales y batimetría de alta precisión...") 

    # Abrimos GEBCO una sola vez para que el bucle sea rápido
    with rasterio.open(ruta_gebco) as src_gebco:
        mapa_profundidad = src_gebco.read(1) 

        # Aqui pinchamos el cubo y recorremos los avistamientos
        for i, fila in df_obis.iterrows():  
            try:
                lat = fila['latitud']
                lon = fila['longitud']
                
                # 1. Pinchamos el cubo ambiental para las variables dinámicas (temperatura, salinidad, etc.)
                punto = ds.sel(  
                    latitude=lat, 
                    longitude=lon, 
                    time=fila['fechas'], 
                    method='nearest'
                )

                datos_punto = punto.to_dict()['data_vars']
                valores = {k: v['data'] for k, v in datos_punto.items()}
                
                # Unimos los datos originales de OBIS con los ambientales del cubo
                registro_completo = {**fila.to_dict(), **valores}
                
                # 2. CORRECCIÓN GEBCO: Sobrescribimos el 'deptho' de Copernicus por el real de GEBCO
                row, col = src_gebco.index(lon, lat) 
                prof_real = abs(float(mapa_profundidad[row, col])) 
                registro_completo['deptho'] = prof_real  # Actualizamos con alta resolución

                lista_ambiental.append(registro_completo) 

            except Exception as e:  
                print(f"Error en registro {i}: {e}") 

    # Creamos el DataFrame final
    df_final = pd.DataFrame(lista_ambiental)  

    # --- LIMPIEZA FINAL ---
    # Eliminamos registros que hayan caído en tierra o fuera del cubo (NaN en temperatura)
    df_final = df_final.dropna(subset=['thetao'])

    # Guardamos el resultado
    df_final.to_csv(ruta_csv_salida, index=False) 
    
    print(f"¡Proceso completado!")
    print(f"Dataset listo para Machine Learning en: {ruta_csv_salida}")
    print(f"Total registros válidos: {len(df_final)}")
    print("\nColumnas del nuevo dataset:")
    print(df_final.columns.tolist())

if __name__ == "__main__":
    pinchar_cubo()