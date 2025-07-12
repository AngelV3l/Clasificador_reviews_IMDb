from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import sqlite3
from datetime import datetime

# Inicializamos la app
app = FastAPI()

# Cargamos modelo y vectorizador
with open('model/modelo_svm.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Clase para recibir la reseña vía POST
class ReseñaEntrada(BaseModel):
    review_text: str
    true_label: str = None  # Opcional

# Función para guardar predicciones en la base de datos
def guardar_prediccion(review_text, predicted_label, true_label=None, db_path='predicciones.db'):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predicciones (review_text, true_label, predicted_label, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (review_text, true_label, predicted_label, timestamp))
    conn.commit()
    conn.close()

# Endpoint raíz (opcional)
@app.get("/")
def read_root():
    return {"mensaje": "API para clasificar reseñas de películas"}

# Endpoint de predicción
@app.post("/predict")
def predecir_sentimiento(datos: ReseñaEntrada):
    texto = datos.review_text
    etiqueta_real = datos.true_label

    # Vectorizamos y predecimos
    texto_vectorizado = vectorizer.transform([texto])
    prediccion = modelo.predict(texto_vectorizado)[0]

    # Guardamos en la base de datos
    guardar_prediccion(texto, prediccion, etiqueta_real)

    return {
        "texto": texto,
        "sentimiento_predicho": prediccion
    }
