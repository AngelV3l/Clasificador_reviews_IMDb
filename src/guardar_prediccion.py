import sqlite3
from datetime import datetime

def guardar_prediccion(review_text, predicted_label, true_label=None, db_path='predicciones.db'):
    """
    Registra una predicción en la base de datos predicciones.db.

    Parámetros:
    - review_text (str): Texto original de la reseña.
    - predicted_label (str): Etiqueta predicha por el modelo.
    - true_label (str): Etiqueta real si está disponible.
    - db_path (str): Ruta a la base de datos SQLite.

    """

    # Obtener timestamp actual
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Conectar a la base de datos
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insertar la fila
    cursor.execute('''
        INSERT INTO predicciones (review_text, true_label, predicted_label, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (review_text, true_label, predicted_label, timestamp))

    # Guardar y cerrar
    conn.commit()
    conn.close()

    print("Predicción registrada correctamente.")
