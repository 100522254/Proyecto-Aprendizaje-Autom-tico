"""
PRIMERA PRACTICA - ENTRENAMIENTO Y ANALISIS DEL MODELO
Grupo 82 - Equipo 15
*   Ariana Cornejo Infante,     100522121, 100522121@alumnos.uc3m.es
*   Francisco Pérez Sokolowski, 100522254, 100522254@alumnos.uc3m.es

Aplicación Streamlit para despliegue del modelo de predicción bancaria.
"""

# Imports necesarios
import pandas as pd
import numpy as np
import requests
import io
from joblib import load

"""
6.3. Carga del modelo final y datos de competición
Se realizará la carga del modelo final entrenado en el Notebook 1 (`modelo_final.joblib`).
"""

# --- CARGAR EL MODELO FINAL ---
# Ruta del modelo exportado desde el Notebook 1
import requests

MODEL_PATH = "modelo_final.joblib"
url_model = "https://github.com/100522254/Proyecto-Aprendizaje-Autom-tico/raw/main/modelo_final.joblib"
url_comp = "https://github.com/100522254/Proyecto-Aprendizaje-Autom-tico/raw/main/bank_competition.pkl"

# Descargar y cargar modelo
response = requests.get(url_model)
with open(MODEL_PATH, "wb") as f:
    f.write(response.content)

print("---- CARGA DEL MODELO FINAL ----\n")
print("Modelo descargado correctamente.")

try:
    pack = load(MODEL_PATH)
    final_pipeline  = pack["pipeline"]
    feature_metadata = pack["feature_metadata"]
    classes_         = pack["classes_"]
    print(f"Modelo cargado correctamente desde '{MODEL_PATH}'.\n")
    print(f"Clases del modelo: {classes_}\n")
except FileNotFoundError:
    raise FileNotFoundError(
        f"No se encontró '{MODEL_PATH}'. "
        "Asegúrate de ejecutar primero el Notebook 1 para generar el modelo."
    )

# --- CARGAR LOS DATOS DE COMPETICIÓN ---
def load_pkl_from_url(url):
    """Carga un archivo .pkl desde una URL y lo devuelve como DataFrame."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_pickle(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error cargando {url}: {e}")
        return None

df_comp = load_pkl_from_url(url_comp)
if df_comp is None:
    raise RuntimeError("No se pudo cargar el dataset de competición. Verifica el enlace.")

print("---- CARGA DE DATOS DE COMPETICIÓN ----\n")
print(f"Dataset de competición cargado: {df_comp.shape[0]} instancias, {df_comp.shape[1]} variables.")
print(df_comp.head(3).to_string())

# --- Preproceso previo de la variable 'pdays_contacted' ---
# Replicar el mismo preproceso aplicado en el Notebook 1
df_comp['pdays_contacted'] = np.where(df_comp['pdays'] == -1, 0, 1)
print("Columna 'pdays_contacted' añadida al dataset de competición.")

"""
6.4. Predicciones para la competición
Se utilizará el modelo final para obtener predicciones para el conjunto de datos de la competición, y se guardarán en 'predicciones.csv'.
"""
predicciones_num = final_pipeline.predict(df_comp)

# Convertir 0/1 → 'no'/'yes'
pred_labels = np.where(predicciones_num == 1, "yes", "no")
pred_df = pd.DataFrame({"deposit": pred_labels})

print("---- PREDICCIONES PARA LA COMPETICIÓN ----\n")
print(f"Total de predicciones generadas: {len(pred_df)}")
print("\nDistribución de predicciones:")
print(pred_df["deposit"].value_counts().to_string())
print(f"\nPorcentaje YES: {(pred_df['deposit'] == 'yes').mean() * 100:.2f}%")
print(f"Porcentaje NO : {(pred_df['deposit'] == 'no').mean() * 100:.2f}%")