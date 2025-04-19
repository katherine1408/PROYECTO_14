# 🚖 Sprint 14 – Predicción de Pedidos por Hora (Series Temporales – Sweet Lift Taxi)

## 📌 Descripción del Proyecto

En este proyecto se aplican técnicas de **análisis de series temporales** para predecir la **cantidad de pedidos de taxis** por hora. Esta predicción es clave para que la empresa **Sweet Lift Taxi** pueda anticipar las **horas pico** y disponer de suficientes conductores disponibles en el momento adecuado.

El objetivo es construir un modelo de predicción cuya métrica **RMSE** en el conjunto de prueba no supere el umbral de **48 pedidos por hora**.

## 🎯 Objetivos del Proyecto

- Convertir una serie temporal en una secuencia regular horaria.
- Aplicar técnicas de descomposición, suavizado y análisis estacional.
- Entrenar modelos de predicción como:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `GradientBoostingRegressor`
- Comparar su rendimiento utilizando **RMSE** como métrica principal.

## 📁 Dataset utilizado

- `taxi.csv`

Columnas:

- `datetime`: marca de tiempo (índice de la serie)
- `num_orders`: número de pedidos de taxis

## 🧰 Funcionalidades del Proyecto

### 🧹 Preparación y remuestreo

- Conversión de `datetime` a índice
- Remuestreo de los datos en intervalos de **1 hora**
- División en conjunto de entrenamiento (90%) y prueba (10%)

### 📈 Análisis de series temporales

- Análisis de estacionalidad y tendencias
- Visualización con medias móviles y ventanas de suavizado

### 🤖 Modelado y predicción

- Generación de variables de retraso (lags)
- Ingeniería de características temporales (hora, día de la semana)
- Comparación de modelos con ajuste de hiperparámetros
- Evaluación de RMSE sobre el conjunto de prueba

## 📊 Herramientas utilizadas

- Python  
- pandas / numpy  
- scikit-learn (`LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`)  
- matplotlib / seaborn  

---

📌 Proyecto desarrollado como parte del Sprint 14 del programa de Ciencia de Datos en **TripleTen**.
