#Aqui va el codigo para crear un cubo de datos de los parametros del agua con el que trabajaremos y sacaremos la informacion de este cubo con los datos de los avistamientos


#---------------------------------------------------PAQUETES QUE SE USAN----------------------

import xarray as xr #Importo la liberia array para trabajar con los archivos .nc (guardar el cubo que vamos a generar, interpolar datos, abrir archivos .nc....)
import os


#----------------------------------------------------- RUTAS ---------------------------------------
ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_de_este_script)
ruta_cubo = os.path.join(ruta_raiz, "data", "carpeta_cubo", "par_ambientales")
path_fis= os.path.join(ruta_cubo, "para_fisicos.nc") #Ruta del archivo con parametros fisicos
path_biog= os.path.join(ruta_cubo, "para_biog.nc")   #Ruta del archivo con parametros biogeoquimicos
path_bati= os.path.join(ruta_cubo, "para_bat.nc")   #Ruta del archivo con parametros batimetria
path_cubo= os.path.join(ruta_cubo, "cubo_ambiental_macaronesia.nc") #Ruta donde vamos a generar el cubo y su nombre

print("Iniciando la fusion del cubo...")


#----------------------------------------------------- CARGA  ---------------------------------------
#"El siguiente bloque sirve para abrir los 3 archivos sin saturar la memoria RAM gracias a la carga perezosa (lazy loading) de xarray

dataset_fis= xr.open_dataset(path_fis) #abre el archivo de los parametros fisicos y lo convierte en un dataset
dataset_biog= xr.open_dataset(path_biog) #abre el archivo de los parametros biogeoquimicos y lo convierte en un dataset
dataset_bati= xr.open_dataset(path_bati)  #abre el archivo de los parametros profundidad y lo convierte en un dataset

#----------------------------------------------------- FILTRADO  ---------------------------------------

#Nuestro archivo parametros biogeoquimicos (para_biog.nc) tiene datos asociados a diferentes capas de profundidad, vamos a sacar solo los datos de la primera capa superficial.
#Esto lo hacemos para que al fusionar los demas archivos esten en la misma capa de superficie.

print("Extrayendo datos biogeoquimicos de la capa superficial...")
dataset_biog_superf= dataset_biog.isel(depth=0).drop_vars("depth") #dataset_biog.isel(depth=0) selecciona la primera capa de profundidad del dataset de parametros biogeoquimicos
                                                                   #drop_vars('depth') elimina las otras capas con diferentes profundidades del dataset que estamso creando
                                                                  

#----------------------------------------------------- INTERPOLACION ---------------------------------------
#Los dataset de la fisica y la biog tiene diferentes resoluciones (0,083 grados y 0,025 grados respectivamente)
#Para poder fusionarlos en un cubo ambos deben de tener la misma rejillas de latitudes y longitudes

print("interpolando la rejilla Biogeoquimica a la rejilla de parametros fisicos...")

dataset_biog_interpol= dataset_biog_superf.interp(  #coge el dataset superficial de parametros biogeoquimicos y lo ajusta a nuevas coordenadas
    latitude= dataset_fis.latitude, #coge las latitudes del dataset físico como referencia
    longitude= dataset_fis.longitude, #coge las longitudes del dataset fisico como referencia
    method= "nearest" #con esto interpola por el valor vecino mas cercano
)

#----------------------------------------------------- UNION ---------------------------------------
#Aqui vamos a unir los dataset que hemos creado y unirlos para crear nuestro cubo que sera un superdataset ambiental
#Este cubo tendra todos los valores de los parametros de los diferentes dataset alineados en latitud y longitud y fecha.


print("Fusionando datos fisicos con biogeoquimcos con datos batimetricos...")
cubo_tfm= xr.merge([dataset_fis, dataset_biog_interpol, dataset_bati])


#----------------------------------------------------- GUARDADO ---------------------------------------
#Guarda el cubo en un archivo NetCDF

print(f"Guardando el Cubo Ambiental en {path_cubo}...")
cubo_tfm.to_netcdf(path_cubo) #escribe el dataset del cubo en un archivo NetCDF y lo nombra y lo guarda en el directorio que definimos al principio.


print("TENEMOS EL CUBO PREPARADO!!!")
print(f"Ahora nuestro cubo tiene las siguientes 8 variables alineadas en la superficie: {list(cubo_tfm.data_vars)}")