#En este bloque se presentan dos modelos de randomforest (uno con todas las variables y otro sin la profundidad)
#Asi podremos  comparar su rendimiento y ver cómo cambia la importancia de cada variable.
#terminara generando un archivo de texto con la informacion

#------------------------------------------LIBRERIAS-------------------------------------------------------------

import pandas as pd #Manipulamos los datos, cargar csv, crear dataframe
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score #Son esenciales para evaluar el modelo correctamente: divide los datos en entrenamiento y test y  calcula el AUC con validación cruzada
from sklearn.ensemble import RandomForestClassifier #con esto creamos el modelo random forest
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, f1_score #Lo usaremos para calcular metricas del modelo
import joblib

#-------------------------------------------ENTRADA Y RUTAS-------------------------------------------------------------
especie_input = input("Especie para análisis completo (ej: Physeter macrocephalus): ").strip()
nombre_folder = especie_input.replace(" ", "_") #los espacios en blancos se sustituyen por barras bajas

ruta_de_este_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_de_este_script)
carpeta_especies = os.path.join(ruta_raiz, "data", "carpeta_especies")

ruta_csv = os.path.join(carpeta_especies, nombre_folder, f"Dataset_presencias_ausencias_{nombre_folder}.csv") #ruta del archivo generado en el paso anterior y que necesita para realizar este paso
ruta_output_txt = os.path.join(carpeta_especies, nombre_folder, f"Informe_Final_ML_{nombre_folder}.txt") #ruta y nombre del archivo que generara este script

if not os.path.exists(ruta_csv): #comprueba si existe el archivo
    print(f" Error: No existe el dataset en {ruta_csv}")
    exit()

#-------------------------------------------CARGA Y LIMPIEZA-------------------------------------------------------------
#Carga el archivo de presencias y ausencias y lo limpia quitando las filas que tienen algun parametro en blanco
#Esto lo hace para dejarlo listo para el entrenamiento

df = pd.read_csv(ruta_csv) #carga el archivo y lo convierte en dataframe

def limpiar_valor(x): #- Define una función llamada limpiar_valor que servirá para corregir valores mal formateados
    if isinstance(x, str): #Comprueba si x es una cadena de texto
        return float(x.replace('[', '').replace(']', '')) #-Si es un string, elimina los corchetes [ ] y convierte el resultado a número flotante
    return x        #Si no es un string, devuelve el valor tal cual

variables_completas = ['thetao', 'so', 'uo', 'vo', 'zos', 'chl', 'o2', 'deptho'] #Creamos una lista con los nombres de todas las variables ambientales que quieres limpiar y usar en el modelo

for var in variables_completas: #Inicia un bucle que recorre cada variable de la lista
    df[var] = df[var].apply(limpiar_valor) #- Aplica la función limpiar_valor a cada valor de esa columna del DataFram


df = df.dropna(subset=variables_completas + ['presencia']) #- Elimina cualquier fila que tenga valores faltantes (NaN) en las variables ambientales o en la columna presencia


#-------------------------------------------ENTRENAMIENTO DEL MODELO-------------------------------------------------------------
#Crea una funcion que entrena un modelo con las variables y calcula las metricas (AUC, accuracy, F1) y genera un ranking de importancia de variables.
#Después devuelve todo empaquetado para que puedas compararlo con otros modelos

def ejecutar_modelo_completo(lista_vars, nombre_analisis): #Define una función que entrenará y evaluará un modelo usando las variables de lista_vars
    X = df[lista_vars] #Crea la matriz de predictores X con solo las columnas indicadas en lista_vars
    y = df['presencia'] #- Define el vector objetivo y usando la columna presencia (0/1)
    
    X_train, X_test, y_train, y_test = train_test_split(  #Separa los datos en entrenamiento (80%) y test (20%), manteniendo el balance de clases (stratify=y) y fijando la semilla para reproducibilidad
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42) #Crea un modelo de Random Forest con 100 árboles y semilla fija
    modelo.fit(X_train, y_train) #Entrena el modelo con los datos de entrenamiento
    
    # Predicciones para métricas
    y_pred = modelo.predict(X_test) #Genera predicciones de clase (0/1) sobre el conjunto de test
    y_proba = modelo.predict_proba(X_test)[:, 1] #- Obtiene las probabilidades de pertenecer a la clase 1 (presencia) para cada registro del tes
    
    # Cálculos de Calidad
    auc_cv = cross_val_score(modelo, X, y, cv=5, scoring='roc_auc').mean() #Calcula el AUC medio usando validación cruzada de 5 particiones sobre todos los datos
    acc = accuracy_score(y_test, y_pred) #Calcula la precisión global (porcentaje de aciertos) en el conjunto de test.
    f1 = f1_score(y_test, y_pred) #Calcula el F1-score (media armónica entre precisión y recall) en el test
    
    # Ranking de Importancia
    imp = modelo.feature_importances_ #Extrae la importancia de cada variable según el Random Forest
    ranking = pd.DataFrame({  #Crea un DataFrame con las variables y su influencia en porcentaje, y lo ordena de mayor a menor importancia
        'Variable': lista_vars,
        'Influencia': imp * 100
    }).sort_values(by='Influencia', ascending=False)
    
    return {   #- Devuelve un diccionario con el modelo entrenado, el ranking de variables, las métricas principales y el informe de clasificación completo
        'modelo': modelo,
        'ranking': ranking,
        'auc': auc_cv,
        'accuracy': acc,
        'f1': f1,
        'report': classification_report(y_test, y_pred)
    }

#-------------------------------------------EJECUCION DE LOS DOS MODELOS/ESCENARIOS-------------------------------------------------------------
print(f"Iniciando procesamiento para {especie_input.upper()}...")

# Escenario A: Con todo (incluyendo corrección GEBCO)
res_con = ejecutar_modelo_completo(variables_completas, "Completo") #Ejecuta la función ejecutar_modelo_completo usando todas las variables ambientales, incluyendo deptho


# Escenario B: Sin profundidad (Sensibilidad)
vars_sin_depth = [v for v in variables_completas if v != 'deptho'] #Crea una nueva lista de variables eliminando deptho
res_sin = ejecutar_modelo_completo(vars_sin_depth, "Sin Profundidad")#Ejecuta el modelo otra vez, pero sin la variable de profundidad.

#-------------------------------------------CONSTRUCCION DE INFORME Y SALIDA-------------------------------------------------------------

lineas = [] # lista será el contenedor donde irás añadiendo todas las líneas del informe final
lineas.append(f"INFORME TFM: MODELADO DE NICHOS PARA {especie_input.upper()}") #Anade el titulo del informe
lineas.append("="*70) #Anade una barra de separacion creando una barra de 70 signos “=”, que sirve como separador visual en el informe
lineas.append(f"Muestra total: {len(df)} registros (Balanceados 1:1)") #Anade el numero de nuestras totales (cuenta cuántas filas tiene el DataFrame después de limpiar los datos)
lineas.append("="*70 + "\n") #Anade linea de separacion (70 signos iguales) y salto de linea

#-------------------------------------------CONVERSION DE RESULTADOS A BLOQUE DE TEXTO-------------------------------------------------------------
#Se genera una funcion que Toma los resultados de un modelo (AUC, accuracy, ranking, clasificación) y convertirlos en un bloque de texto bonito y legible para el informe

def formatear_resultados(titulo, res): #creamos una funcionque necesita un titulo y un diccionario
    txt = f" {titulo.upper()}\n" #Pone el futuro titulo en mayusculas y da un salto de linea
    txt += f"AUC (Validación Cruzada): {res['auc']:.4f}\n" #Concatena mas texto a la cabecera, el valor de AUC dentro del diccionario res
    txt += f" Precisión Global: {res['accuracy']:.2%} | F1-Score: {res['f1']:.4f}\n" #Anade texto y el valor de Accuracy como porcentaje
    txt += "-"*50 + "\n" #Anade una linea de 50 guiones
    for i, (_, row) in enumerate(res['ranking'].iterrows(), 1):#- Recorre fila por fila el DataFrame res['ranking'], que contiene las variables en una columna y el valor de su influencia en otro
        barra = "█" * int(row['Influencia'] / 2) #Construccion de la barrera visual
        txt += f"{i:2d}º | {row['Variable']:8s} | {row['Influencia']:6.2f}% {barra}\n"#construye la linea del rankig:con posicion, texto (variable), su porcentaje y la barra visual
    txt += "\nDETALLE ESTADÍSTICO:\n" + res['report'] + "\n" #Añade el classification report completo generado por sklearn
    return txt

#-------------------------------------------LLAMADA DE LA FUNCION-------------------------------------------------------------
#llama a la función formatear_resultados y le pasa los dos argumentos: titulo y res

lineas.append(formatear_resultados("Modelo Completo (Oceanografía + GEBCO)", res_con)) #Llama a la funcion con un titulo y un diccionario (con la profundidad) y los resultados generados los guarda en la lista linea
lineas.append(formatear_resultados("Análisis de Sensibilidad (Sin Profundidad)", res_sin)) #Llama a la funcion con un titulo y un diccionario (sin la profundidad) y los resultados generados los guarda en la lista linea

# Conclusión de Sensibilidad
diff_auc = res_con['auc'] - res_sin['auc'] # Resta el AUC del modelo sin profundidad al AUC del modelo completo
lineas.append("="*70) #Añade una línea de separación
lineas.append(f" IMPACTO DE LA PROFUNDIDAD (Pérdida de AUC): {diff_auc:.4f}") #Añade la frase del impacto de la profundidad e inserta el valor calculado
lineas.append(f"NUEVA VARIABLE DOMINANTE: {res_sin['ranking'].iloc[0]['Variable']}") #Añade la nueva variable dominante tomando la primera fila 
lineas.append("="*70) #Añade otra línea de separación

#-------------------------------------------GUARDADO Y MOSTRAR-------------------------------------------------------------
texto_final = "\n".join(lineas) #- Toma la lista lineas, que contiene cada sección del informe como un elemento independiente y las une en una sola cadena de texto separadas por saltos de linea
with open(ruta_output_txt, 'w', encoding='utf-8') as f: #abre el archivo que se genera y escribe en el todo lo guardado en la lista linea
    f.write(texto_final)

print(texto_final)
print(f"\n Análisis finalizado. Informe guardado en: {ruta_output_txt}") #muestra la ruta exacta donde se guardo el txt

#-------------------------------------------EXPORTAR MODELO ENTRENADO-------------------------------------------------------------
# Definimos la ruta exacta dentro de la carpeta de la especie con el nombre que espera el script del mapa
ruta_modelo = os.path.join(carpeta_especies, nombre_folder, f"modelo_RF_{nombre_folder}.pkl")

# Extraemos el modelo entrenado con profundidad (res_con) y lo guardamos en el disco duro
joblib.dump(res_con['modelo'], ruta_modelo)

print(f"¡Cerebro del modelo guardado! Archivo listo para el mapa en: {ruta_modelo}")