# 🎬 Clasificador de Reseñas de Películas (IMDb)

Este proyecto consiste en un pipeline completo de ciencia de datos para clasificar reseñas de películas en **positivas** o **negativas**, utilizando técnicas de **Procesamiento de Lenguaje Natural (NLP)** y **Machine Learning**. Incluye análisis exploratorio, preprocesamiento, entrenamiento de modelos, almacenamiento en base de datos y dos formas de despliegue: **API REST con FastAPI** e **interfaz web interactiva con Gradio**.

---

## 🧠 Objetivo

- Clasificar reseñas de películas reales del dataset IMDb.
- Construir un modelo interpretable y preciso usando TF-IDF + SVM.
- Desarrollar herramientas accesibles tanto para humanos como para máquinas:
  - `FastAPI` para integración vía API.
  - `Gradio` para uso interactivo.

---

## 📁 Estructura del proyecto

Clasificador de reseñas/

│

├── app/ ← Contiene API REST (FastAPI) y frontend (Gradio)

│ ├── api.py

│ └── app_gradio.py

│

├── data/ ← Datos crudos y preprocesados

│ ├── IMDB_reviews.csv

│ └── IMDB_reviews_clean.csv

│

├── model/ ← Modelo entrenado y vectorizador

│ ├── modelo_svm.pkl

│ └── tfidf_vectorizer.pkl

│

├── notebooks/ ← Exploración y modelado en Jupyter

│ ├── EDA_reviews.ipynb

│ └── preprocesamiento_modelado.ipynb

│

├── src/ ← Scripts reutilizables (utils)

│ ├── guardar_prediccion.py

│ ├── consultar_db.py

│ └── setup_db.py

│

├── predicciones.db ← Base de datos SQLite con las predicciones

├── requirements.txt ← Dependencias del proyecto

└── README.md ← Este archivo

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clona el repositorio

git clone https://github.com/AngelV3l/Clasificador-de-rese-as.git


### 2. Crea un entorno virtual

python -m venv venv

source venv/bin/activate   # Linux/Mac

venv\Scripts\activate      # Windows

### 3. Instala las dependencias

pip install -r requirements.txt

### 4. Ejecutar la interfaz visual (Gradio)

cd app

python app_gradio.py

- Se abrirá en tu navegador en http://127.0.0.1:7860/

### 5. Ejecutar la API REST (FastAPI)

cd app

uvicorn api:app --reload

- Accede a la documentación interactiva en http://127.0.0.1:8000/docs

---

## 🧪 Dataset

Fuente: IMDb Movie Reviews Dataset.

Clasificación binaria: positive o negative.

Dataset balanceado con 50.000 reseñas.

---

## 🔍 Tecnologías utilizadas

| Tecnología           | Uso principal                         |
| -------------------- | ------------------------------------- |
| **Python**           | Lenguaje principal                    |
| **Jupyter Notebook** | EDA y entrenamiento                   |
| **Scikit-learn**     | Modelado con SVM, TF-IDF, etc.        |
| **Gradio**           | Interfaz interactiva para usuarios    |
| **FastAPI**          | API REST para despliegue programático |
| **SQLite3**          | Registro de predicciones              |


---

## 🛠️ Funcionalidades destacadas

-Clasificación automática de reseñas con SVM.

-Limpieza avanzada de texto.

-Comparación de varios modelos (LogReg, Naive Bayes, SVM).

-Almacenamiento de cada predicción en base de datos.

-Interfaz interactiva vía navegador.

-API REST para integración con otros sistemas.

---

## 📦 Futuras mejoras

-Incluir dashboard con visualizaciones en Streamlit o PowerBI.

-Añadir un modelo basado en deep learning (LSTM o BERT).

-Desplegar en Hugging Face Spaces o Render.

## 👨‍💻 Autor

Ángel Velázquez Bolívar  
📫 [LinkedIn](https://www.linkedin.com/in/angelvelazquezbolivar)  
💼 [GitHub](https://github.com/AngelV3l)

---
