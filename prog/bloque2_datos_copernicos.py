
#------------------------------------PAQUETES QUE SE USAN----------------------
#Antes de nada hemos tenido que crear una cuenta en copernicus y despues logearnos en nuestra terminal

import copernicusmarine #acceder, explorar y descargar datos oceánicos del servicio Copernicus Marine
import os   ##Permite gestionar archivos y carpetas
import shutil



# ------------------------------------------ CONFIGURACIÓN GEOGRÁFICA MACARONESIA ------------------------------------

#definimos el espacio y el tiempo de los datos que vamos a querer descargarnos
rango_longitud= [-32.0, -13.0]
rango_latitud= [14.0, 40.0]
fecha_inicio= "2010-01-01T00:00:00"
fecha_fin= "2025-11-30T00:00:00"

def descarga_datos(): #definimos una funcion que despues llamaremos, es una recomendacion para que el script se vea mas ordenado, se podria hacer sin esta funcion.
    print("Comenzando la descarga de datos")

    #comenzamos con la descarga de los paarmetros
    #PARAMETROS FISICOS
    ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_de_este_script)
    ruta_cubo = os.path.join(ruta_raiz, "data", "carpeta_cubo", "par_ambientales")
# Si la carpeta ya existe con descargas anteriores (o corruptas), la borramos entera
    if os.path.exists(ruta_cubo):
        print(f"Detectada carpeta previa en: {ruta_cubo}")
        print("Limpiando archivos antiguos de forma segura...")
        for archivo in os.listdir(ruta_cubo):
            ruta_archivo = os.path.join(ruta_cubo, archivo)
            try:
                if os.path.isfile(ruta_archivo):
                    os.remove(ruta_archivo)
                    print(f"Eliminado: {archivo}")
            except Exception as e:
                print(f"No se pudo eliminar {archivo}: {e}")
    else:
        os.makedirs(ruta_cubo, exist_ok=True)
        print(f"Carpeta de destino creada en: {ruta_cubo}")
    os.makedirs(ruta_cubo, exist_ok=True)

    par_fisicos = os.path.join(ruta_cubo, "para_fisicos.nc")
    print("Descargando parametros fisicos...")

    copernicusmarine.subset( #Copernicusmarine.subset() le da la instrucccion a copernicus para descargar nuestros datos espaciales y temporales en la macaronesia.
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m", #paquete de datos que queremos descargar
        variables=["thetao", "so", "uo", "vo", "zos"], #Variables que queremos descargar (thetao:temperatura del mar, so:salinidad, uo, vo: componentes zonal y meridional de la corriente, zos:altura de la superficie del mar)
        minimum_longitude= rango_longitud[0], #Usa los rango de longitud, latitud y temporales para limitar la descarga de datos
        maximum_longitude= rango_longitud[1],
        minimum_latitude= rango_latitud[0],
        maximum_latitude= rango_latitud[1],
        start_datetime=fecha_inicio,
        end_datetime=fecha_fin,
        minimum_depth=0.49, #esto seleccionamos solo la superficie.
        maximum_depth=0.5,
        output_filename= par_fisicos #el archivo de salida con el nombre

    )

    #PARAMETROS BIOGEOQUIMICOS
    par_biog=os.path.join(ruta_cubo, "para_biog.nc")
    print("Descargando parametros biogeoquimicos...")

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1M-m",
        variables=["chl", "o2"],       #Variables que queremos descargar (chl:clorodfila, o2: oxigeno)
        minimum_longitude= rango_longitud[0], #Usa los rango de longitud, latitud y temporales para delimitar la descarga de datos
        maximum_longitude= rango_longitud[1],
        minimum_latitude= rango_latitud[0],
        maximum_latitude= rango_latitud[1],
        start_datetime= fecha_inicio,
        end_datetime= fecha_fin,
        output_filename= par_biog
    )

    #PARAMETRO PROFUNDIDAD (BATIMETRIA)

    par_bat= os.path.join(ruta_cubo, "para_bat.nc")
    print("Descargando datos de batimetria")

    copernicusmarine.subset(
         dataset_id="cmems_mod_glo_phy_my_0.083deg_static",
        variables=["deptho"], #deptho:profundidad del fondo marino en metros
        minimum_longitude= rango_longitud[0],
        maximum_longitude= rango_longitud[1],
        minimum_latitude= rango_latitud[0],
        maximum_latitude= rango_latitud[1],
        output_filename= par_bat
    )

    print(f"Descarga completada! Los archivos estan en {ruta_cubo}")

if __name__ == "__main__":  #Se lanza la el programa descarga_datos si ejecute/abro este script. Puede considerarse como una seguridad para que no se lance automaticamente
    descarga_datos()