import gradio as gr
import pickle
from datetime import datetime
import sqlite3

# Cargar modelo y vectorizador
with open('../model/modelo_svm.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('../model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Función para guardar predicción en la base de datos
def guardar_prediccion(review_text, predicted_label, db_path='predicciones.db'):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predicciones (review_text, true_label, predicted_label, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (review_text, None, predicted_label, timestamp))
    conn.commit()
    conn.close()

# Función de predicción para Gradio
def predecir_sentimiento(texto):
    try:
        # Vectorizar texto
        vector = vectorizer.transform([texto])
        
        # Predecir
        pred = modelo.predict(vector)[0]
        
        # Guardar en la base de datos
        guardar_prediccion(texto, pred)
        
        return f"🧠 Sentimiento predicho: {pred.upper()}"
    
    except Exception as e:
        return f"Error: {str(e)}"
# Interfaz de Gradio
interface = gr.Interface(
    fn=predecir_sentimiento,
    inputs=gr.Textbox(lines=5, placeholder="Escribe tu reseña aquí..."),
    outputs="text",
    title="Clasificador de Sentimiento de Reseñas de Películas 🎬",
    description="Introduce una reseña de película para obtener su sentimiento (positivo o negativo)."
)

# Lanzamos interfaz
interface.launch()
