#Este script 5Pincha” el cubo ambiental en cada punto de avistamiento y extrae las condiciones del océano en ese lugar y fecha.

#-----------------------------------------------LIBRERIAS-------------------------------------------
import xarray as xr  #Es la herramienta que permite abrir y manipular el cubo ambiental NetCDF
import pandas as pd   #Es la librería que maneja tablas y CSVs.
import os       #Es una librería del sistema operativo, construye rutas, comprueba si el csv de ntrada exsite...
import sys
#-------------------------------------------------RUTAS----------------------------------------------
if len(sys.argv) > 1:
    especie = sys.argv[1].replace("_", " ")
    print(f"Especie recibida automáticamente: {especie}")
else:
    especie = input("Introduce el nombre científico de la especie: ").strip()
    
nombre_carpeta_especie = especie.replace(" ", "_")      #Crea un nombre de carpeta para despues construir las rutas.

ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_de_este_script)
carpeta_especies = os.path.join(ruta_raiz, "data", "carpeta_especies")
ruta_cubo = os.path.join(ruta_raiz, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")  #ruta donde se ecnuentra nuestro cubo ambiental

ruta_csv_entrada = os.path.join(carpeta_especies, nombre_carpeta_especie, f"avistamientos_{nombre_carpeta_especie}.csv") #aqui es donde encontara el archivo de avistamientos
ruta_csv_salida = os.path.join(carpeta_especies, nombre_carpeta_especie, f"datos_cubo_{nombre_carpeta_especie}.csv") #Aqui generara el nuevo archivo

#-------------------------------------------------EXTRACCION DE VARIABLES----------------------------------------------
#Vamos a generar una funcion que  va a  tomar cada avistamiento real y le va a añadir todas las variables ambientales del cubo NetCDF

def pinchar_cubo():
    if not os.path.exists(ruta_csv_entrada): #comprueba que el archivo de avistamientos existe
        print(f"No se encontró el CSV de avistamientos en: {ruta_csv_entrada}")
        return #con esta funcion return sale de kla funncion si no encunetra el archivo, para que no falle mas adelante.

    print(f"Cargando cubo ambiental y avistamientos de {especie}...") #Si lo encuentra genera este mensaje

    #Abre el cubo y Avistamientos
    ds = xr.open_dataset(ruta_cubo) #abre el cubo ambiental
    df_obis = pd.read_csv(ruta_csv_entrada) #carga el csv de avistamientos

    # Convertimos la columna de fechas a formato datetime para que Xarray la entienda
    df_obis['fechas'] = pd.to_datetime(df_obis['fechas'])


    # Convertimos la columna de fechas a formato datetime para que Xarray la entienda
    df_obis['fechas'] = pd.to_datetime(df_obis['fechas']) #Convierte las fechas del CSV a un formato que el cubo pueda entender

    lista_ambiental = []

    print("Extrayendo datos ambientales para cada avistamiento...") #mensaje para informar de lo que viene

    #Aqui pinchamos el cubo
    for i, fila in df_obis.iterrows():  #Recorre cada avistamiento del csv
        try:
            # "Pinchamos" el cubo en la lat, lon y tiempo más cercanos
            # El método 'nearest' busca el píxel y el mes que mejor coincidan
            punto = ds.sel(     #punto va a devolver un xarray con todas las variables del cubo en ese lugar y fecha como profundidad, salinidad...
                latitude=fila['latitud'], 
                longitude=fila['longitud'], 
                time=fila['fechas'], 
                method='nearest'
            )

            # Convertimos ese punto del cubo a un diccionario de Python
            datos_punto = punto.to_dict()['data_vars']
            
            # Extraemos los valores reales (están dentro de la clave 'data')
            valores = {k: v['data'] for k, v in datos_punto.items()}
            
            # Unimos los datos originales de OBIS con los nuevos ambientales
            registro_completo = {**fila.to_dict(), **valores}
            lista_ambiental.append(registro_completo) #Cada avistamiento enriquecido CON LOS DATOS DEL CUBO se guarda en la lista


        except Exception as e:  #Este except evita que si al pinchar no hay datos no se rompa la funcion saltando al siguiente avistamiento
            print(f"Error en registro {i}: {e}") #dice que fila dio problemas y que tipo de error ocurrio

    # Creamos el DataFrame final
    df_final = pd.DataFrame(lista_ambiental)  #convertimos la lista en un Dataframe

    # --- LIMPIEZA FINAL ---
    # Eliminamos registros que hayan caído en tierra o fuera del cubo (NaN)
    df_final = df_final.dropna(subset=['thetao', 'deptho'])

     # Guardamos el resultado
    df_final.to_csv(ruta_csv_salida, index=False) #guarda el dataframe en un csv
    
    print(f"¡Proceso completado!")
    print(f"Dataset listo para Machine Learning en: {ruta_csv_salida}")
    print(f"Total registros válidos: {len(df_final)}")
    print("\nColumnas del nuevo dataset:")
    print(df_final.columns.tolist())

if __name__ == "__main__":
    pinchar_cubo()




