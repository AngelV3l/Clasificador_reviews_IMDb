import sqlite3

# Creamos la base de datos
conn = sqlite3.connect('predicciones.db')
cursor = conn.cursor()

# Creamos la tabla si no existe
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predicciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_text TEXT,
        true_label TEXT,
        predicted_label TEXT,
        timestamp TEXT
    )
''')

# Guardamos cambios y cerramos conexión
conn.commit()
conn.close()

print("Base de datos creada y tabla 'predicciones' lista.")
