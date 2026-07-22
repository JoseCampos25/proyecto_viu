ESPECIE = config.get("especie", "Tursiops_truncatus")

rule all:
    input:
        f"data/carpeta_especies/{ESPECIE}/mapaOBIS_{ESPECIE}.html",
        f"data/carpeta_especies/{ESPECIE}/avistamientos_{ESPECIE}.csv",
        f"data/carpeta_especies/{ESPECIE}/datos_cubo_{ESPECIE}.csv",
        f"data/carpeta_especies/{ESPECIE}/Dataset_presencias_ausencias_{ESPECIE}.csv",
        f"data/carpeta_especies/{ESPECIE}/modelo_RF_{ESPECIE}.pkl",
        f"results/mapas/Mapa_Nicho_Incertidumbre_{ESPECIE}.png",
        f"results/mapas/Scatter_Validacion_{ESPECIE}.png"

rule descargar_datos_obis:
    output:
        csv="data/carpeta_especies/{especie}/avistamientos_{especie}.csv",
        mapa="data/carpeta_especies/{especie}/mapaOBIS_{especie}.html"
    shell:
        "python prog/bloque1_avistamientos.py {wildcards.especie}"

rule extraccion_cubo_ambiental:
    input:
        csv="data/carpeta_especies/{especie}/avistamientos_{especie}.csv",
        cubo="data/carpeta_cubo/par_ambientales/cubo_ambiental_macaronesia.nc"
    output:
        csv_salida="data/carpeta_especies/{especie}/datos_cubo_{especie}.csv"
    shell:
        "python prog/bloque5_extraccion_cubo.py {wildcards.especie}"

rule generar_presencias_ausencias:
    input:
        csv_presencias="data/carpeta_especies/{especie}/datos_cubo_{especie}.csv",
        cubo="data/carpeta_cubo/par_ambientales/cubo_ambiental_macaronesia.nc",
        gebco="data/gebco_macaronesia.tif"
    output:
        csv_final="data/carpeta_especies/{especie}/Dataset_presencias_ausencias_{especie}.csv"
    shell:
        "python prog/bloque6_generacion_ausencias.py {wildcards.especie}"

rule entrenamiento_random_forest:
    input:
        csv_final="data/carpeta_especies/{especie}/Dataset_presencias_ausencias_{especie}.csv"
    output:
        informe="data/carpeta_especies/{especie}/Informe_Final_ML_{especie}.txt",
        modelo="data/carpeta_especies/{especie}/modelo_RF_{especie}.pkl"
    shell:
        "python prog/bloque7_Entrenamiento.py {wildcards.especie}"


rule generar_mapa_desvest:
    input:
        modelo="data/carpeta_especies/{especie}/modelo_RF_{especie}.pkl",
        cubo="data/carpeta_cubo/par_ambientales/cubo_ambiental_macaronesia.nc"
    output:
        mapa_nicho="results/mapas/Mapa_Nicho_Incertidumbre_{especie}.png",
        scatter="results/mapas/Scatter_Validacion_{especie}.png"
    shell:
        "python prog/bloque11_plots.py {wildcards.especie}"