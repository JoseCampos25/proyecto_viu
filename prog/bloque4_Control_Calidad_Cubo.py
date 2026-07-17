#---------------------------------------------------PAQUETES QUE SE USAN----------------------
import xarray as xr
import numpy as np
import os

#----------------------------------------------------- RUTAS ---------------------------------------
ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_de_este_script)
path_cubo = os.path.join(ruta_raiz, "data", "carpeta_cubo", "par_ambientales", "cubo_ambiental_macaronesia.nc")

print("🔍 Iniciando la inspección del Cubo Ambiental...")

#----------------------------------------------------- COMPROBACIÓN ---------------------------------------
if not os.path.exists(path_cubo):
    print(f"❌ Error: No se encuentra el archivo del cubo en {path_cubo}")
    print("Asegúrate de haber ejecutado primero el script de fusión.")
else:
    # Abrimos el cubo generado
    cubo = xr.open_dataset(path_cubo)
    
    print("\n📊 --- RESUMEN DE VARIABLES ENCONTRADAS ---")
    print(f"El cubo contiene las siguientes variables: {list(cubo.data_vars)}\n")
    
    # Recorremos cada variable ambiental para analizar su contenido
    for var in cubo.data_vars:
        print(f"Anclando análisis en la variable: [{var}]")
        
        # Extraemos los datos de la variable como un array de numpy
        datos = cubo[var].values
        
        # Contamos cuántos elementos totales hay
        total_puntos = datos.size
        
        # Contamos cuántos NaNs hay (común en zonas de tierra/islas como Canarias/Azores)
        total_nans = np.isnan(datos).sum()
        
        # Contamos cuántos ceros REALES hay (excluyendo los NaNs)
        total_ceros = np.count_nonzero(datos == 0)
        
        # Puntos con datos numéricos reales y válidos (ni cero, ni NaN)
        datos_validos = total_puntos - total_nans - total_ceros
        
        # Calculamos porcentajes para que sea más fácil de interpretar
        pct_validos = (datos_validos / total_puntos) * 100
        pct_nans = (total_nans / total_puntos) * 100
        pct_ceros = (total_ceros / total_puntos) * 100
        
        print(f"   🔹 Total de píxeles/puntos: {total_puntos}")
        print(f"   🔹 Datos válidos (con valores): {datos_validos} ({pct_validos:.2f}%)")
        print(f"   🔹 Datos vacíos (NaN / Tierra): {total_nans} ({pct_nans:.2f}%)")
        print(f"   🔹 Ceros absolutos (0.0): {total_ceros} ({pct_ceros:.2f}%)")
        
        # Alerta si la variable es un desierto de ceros o NaNs
        if datos_validos == 0:
            print(f"   🚨 ALERTA: ¡La variable '{var}' NO tiene datos útiles! Son todo ceros o NaNs.")
        elif pct_ceros > 95:
            print(f"   ⚠️ ADVERTENCIA: Más del 95% de '{var}' son ceros. Revisa si es normal.")
        else:
            print(f"   ✅ Variable '{var}' verificada correctamente.")
        print("-" * 50)

    print("\n🏁 ¡Inspección terminada!")