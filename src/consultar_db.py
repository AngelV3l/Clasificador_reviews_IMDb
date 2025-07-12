import sqlite3
import pandas as pd

# Conectamos a la base de datos
conn = sqlite3.connect('predicciones.db')

# Leemos toda la tabla en un DataFrame
df_predicciones = pd.read_sql_query("SELECT * FROM predicciones", conn)

# Cerramos la conexión
conn.close()

# Mostramos las predicciones
print(df_predicciones.head())