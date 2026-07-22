#Avistamientos.py
#Aqui va el codigo para descargar los avistamientos.

#-----------PAQUETES QUE SE USAN----------------------
import requests #hace peticioones a servidores, en nuestro caso Envia una petición GET a la API de OBIS y Recibe respuesta en formato JSON
from datetime import datetime #Convierte cadenas de texto en fechas y te permite trabajar con fechas.
import folium #Para generar mapas interactivos
import csv #Guardar datos en archivos csv
import os #Permite gestionar archivos y carpetas
import sys  # Permite leer argumentos desde la terminal


#----------------- PEDIR NOMBRE CIENTIFICO ----------------------
if len(sys.argv) > 1:
    especie = sys.argv[1].replace("_", " ")
    print(f"Especie recibida automáticamente: {especie}")
else:
    especie = input("Introduce el nombre científico de la especie (ej: Orcinus orca): ").strip()


#-------------------- CREAMOS DIRECTORIO PARA LOS DATOS-------Or--------------------
# Creamos carpeta de especie

ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_de_este_script)
carpeta_especies = os.path.join(ruta_raiz, "data", "carpeta_especies")
if not os.path.exists(carpeta_especies):
    os.makedirs(carpeta_especies)
    print(f"Carpeta creada: {carpeta_especies}")
# Creamos un nombre de carpeta seguro (ej: "Delphinus delphis" -> "Delphinus_delphis")

nombre_carpeta=os.path.join(carpeta_especies, especie.replace(" ", "_"))
if not os.path.exists(nombre_carpeta): #comprueba si existe o no en el directorio un archivo/carpeta con el nombre (devuelve TRUE o FALSE)
    os.makedirs(nombre_carpeta) # si la carpeta no existe la crea
    print(f"carpeta creada: {nombre_carpeta}")


#----------------- BOUNDING BOX DE LA MACARONESIA ----------------------
#Delimitaremos el area a trabajar en nuestro caso la macaronesia

params= {           #Diccionario para la API del OBIS para buscar la especie en la macaronesia y como maximo que nos de 5000 registros
    "scientificname": especie,
    "geometry": "POLYGON((-32 14, -32 40, -13 40, -13 14, -32 14))", #Define un poligono en formato WKT con longitudes de -32 a -13 y latitudes de 14 a 40 donde se perimetriza la region de la MAcaronesia
    "startdate": "2010-01-01",
    "size": 10000
}
url="https://api.obis.org/v3/occurrence" #el servidos de OBIS a quien pedimos los datos
response= requests.get(url, params=params) #envia la peticion y  requests convierte nuestro diccionario params en una URL con parámetros

if response.status_code !=200:  # response.status_code es el código HTTP de la respuesta y 200 es que todo esta bien
    print("Problemas para conectar con OBIS")  #Si la peticion de OBIS no ha ido bien entra en este bucle if
    exit()

data= response.json()  #.json() convierte el contenido de la respuesta (que viene en formato JSON) en un diccionario de Python.
                        # data es otro diccionario
total= data.get("total", 0) #Para evitar que haya falla, siempre habra o cero o un el numero de avistamientos que coincida con al clave total

print(f"\nAvistamientos en la Macaronesia para {especie}: {total}\n") 

if total == 0:
    print("No hay registros en esta región.")
    exit()

def limpiar_fecha(fecha_raw):  #Normaliza cualquier formato de fecha de OBIS devuelve (fecha, hora, fecha_sort)

    if not isinstance(fecha_raw, str):   #comprueba si fecha_raw es una cadena de texto y si no devuelve NA para fecha y hora y None para fecha_sort
        return "NA", "NA", None          #fecha_sort es la versión “ordenable” de la fecha. Es la pieza clave que permite filtrar por año y ordenar cronológicament

    fecha_raw= fecha_raw.split(",")[0] #Limpieza sobre la cadena que devuelve OBiS, divide la cadena donde estan las comas y se queda con el primer valor antes de la coma (el que nos interesa)
    fecha_raw= fecha_raw.split("/")[0] #mismo proceso que el anterior si la fecha es un rango, separa las fechas con / y se queda con la primeraque es la más útil para ordenar y filtrar

    if "T" in fecha_raw:   #Comprueba si la cadena contiene una "T", que es típica del formato ISO 8601 (tipico de los datos del OBIS). Si hay una , significa que la fecha incluye hora, así que se procesa con este bloque.
        partes= fecha_raw.split("T") #divide la cadena en dos partes antes de la T sera la fecha y despues de la T sera la hora.
        fecha= partes[0] #guarda la parte de fechas
        hora= partes[1].replace("Z", "")  #guarda la parte de la hora y quita la "Z" final, que indica que la hora está en UTC
        try:  #Indica que va a ejecutar un código que podría fallar
            fecha_sort = datetime.fromisoformat(fecha) #Intenta convertir la cadena fecha (por ejemplo "2015-06-01") en un objeto datetime.
        except: #Si la conversión falla (por ejemplo, si la fecha es "2015-13-40" o "unknown"), el programa no se rompe. En lugar de eso, pasa al bloque except.
            fecha_sort= None #Si la fecha no se puede convertir, se asigna None
        return fecha, hora, fecha_sort #devuelve tupla con los 3 elementos que nos interesa
    
    if len(fecha_raw) >= 10: #Comprueba que la cadena tiene al menos 10 caracteres ya que  fecha estándar  tiene exactamente 10 caracteres
        fecha= fecha_raw[:10] #extrae solo los primeros 10 caracteres.
        try:
            fecha_sort= datetime.fromisoformat(fecha) #Esta línea intenta transformar la cadena fecha (por ejemplo "2015-06-01") en un objeto datetime
        except:
            fecha_sort= None #Si la fecha no se puede convertir, se asigna None
        return fecha, "NA", fecha_sort
    
    return fecha_raw, "NA", None #Si no cumple las tres condiciones anteriores que devuelva la tupla que hay escrita.

#-----------------PROCESAMIENTO DE REGISTROS---------------------------------
registros= [] #creamos una lista vacia
for occ in data["results"]: #data es el diccionario que devuelve response.json() y "results" es una lista de ocurrencias (cada una es un diccionario con info del avistamiento)
    date_raw= occ.get("eventDate", "NA") #obtener la clave eventDate si no existe dara NA. EventDate es el nombre de un campo estándar del formato Darwin Core, que es el esquema que usa OBIS para describir metadatos de biodiversidad. Representa la fecha del evento de observación, es decir, cuándo ocurrió el avistamiento
    lat= occ.get("decimalLatitude", None) #Para extrer lat y lon del registro si no hay dara None
    lon= occ.get("decimalLongitude", None)
    fecha, hora, fecha_sort= limpiar_fecha(date_raw) #Llamamos a la función  limpiar_fecha para normalizar la fecha. Nos devuelve la fecha, hora y una fecha que se puede ordenar (si hay) 
    registros.append({"fecha": fecha, "hora": hora, "lat": lat, "lon": lon, "fecha_sort": fecha_sort}) #creamos un diccionario con los campos ya limpios y lo anadimos a la lista registro.

#Al final de este bucle registro es una lista con diccionarios cada uno represetara un avistamiento con su informacion. Ya podriamos generar un .csv, filtrar por fecha o incluso generar un mapa

#-------------------------FILTRA Y ORDENAR FECHAS A PARTIR DEL 2010---------------------------

registros_filtrados= [      #Recorre todos los registros y se queda solo con los mayores del 2010
     r for r in registros
     if r["fecha_sort"] is not None and r["fecha_sort"].year >=2010]
    
registros_ordenados= sorted(registros_filtrados, key= lambda x: x["fecha_sort"]) #coge los registros filtrados y los ordena a traves de fecha_sort y los guarda en una nueva lista registros_ordenados
    

#----------------------DEFINIMOS RUTA Y NOMBRE DE ARCHIVOS-----------------------------------------------------
# Definimos las rutas dentro de la carpeta de la especie
csv_filename= os.path.join(nombre_carpeta, f"avistamientos_{especie.replace(' ', '_')}.csv") #Generamos el nombre del archivo donde se guardaran los avistamientos en formato .csv
mapa_filename= os.path.join(nombre_carpeta, f"mapaOBIS_{especie.replace(' ', '_')}.html")  #Generamos el nombre del archivo html que contendra el mapa de avistamientos.

#--------------------------GUARDADO DE ARCHIVOS-----------------------------------------------------------

with open(csv_filename,"w", newline="", encoding="utf-8") as f: #abre el archivo y con newline evitamos generar lineas ne blanco y con encoding=utf-8 asegura que no haya problemas con los caracteres especiales y se guarden correctamente
    writer= csv.writer(f) #Para comenzar a escribir en el archivo
    writer.writerow(["fechas", "hora", "latitud", "longitud"]) #escribe la primera fila del archivo que seran los nombres de las columna
    for r in registros_ordenados: #recorre la variables registros_ordenados 
        writer.writerow([r["fecha"], r["hora"], r["lat"], r["lon"]]) #escribe en el archivo los parametros

print(f"csv guardado en {csv_filename}")


#-------------------------- CREA Y GUARDA EL MAPA--------------------------------------------------------

mapa= folium.Map(location=[28, -20], zoom_start=5) #define el punto central del mapa y el zoom inicial
for r in registros_ordenados:
    if r["lat"] and r['lon']: #coimprueba que haya en los registros los parametros lon y lat
        folium.Marker(   #Este bloque anande un punto en el mapa que es el del avistamiento
            location=[r['lat'], r["lon"]], #marca el avistamiento en el mapa
            popup= f"Fecha :{r['fecha']} <br> Especie: {especie}", #define el texto que aparece al hacer click
            icon= folium.Icon(color= "green") #pone un icono verde para todos los avistamientos
            ).add_to(mapa) #lo anade al mapa
        
mapa.save(mapa_filename)
print(f"Mapa guardado en: {mapa_filename}")