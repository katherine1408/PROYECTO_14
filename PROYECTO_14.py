
# # Descripción del proyecto
# 
# La compañía Sweet Lift Taxi ha recopilado datos históricos sobre pedidos de taxis en los aeropuertos. Para atraer a más conductores durante las horas pico, necesitamos predecir la cantidad de pedidos de taxis para la próxima hora. Construye un modelo para dicha predicción.
# 
# La métrica RECM en el conjunto de prueba no debe ser superior a 48.
# 
# 
# 



# ## Instrucciones del proyecto.
# 
# 1. Descarga los datos y haz el remuestreo por una hora.
# 2. Analiza los datos
# 3. Entrena diferentes modelos con diferentes hiperparámetros. La muestra de prueba debe ser el 10% del conjunto de datos inicial.4. Prueba los datos usando la muestra de prueba y proporciona una conclusión.
# 

# ### Inicialización:

# In[1]:


# Importamos librerías necesarias :

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# 
# Los datos se almacenan en el archivo `taxi.csv`. 	
# El número de pedidos está en la columna `num_orders`.

# In[2]:


data_taxi = pd.read_csv('taxi.csv')


# In[3]:


display(data_taxi.sample(10))


# In[4]:


data_taxi.info()


# In[5]:


data_taxi.describe()


# ## Preparación

# In[6]:


data_taxi.columns = data_taxi.columns.str.strip()


# In[7]:


data_taxi = data_taxi.rename(columns={'datetime': 'date_time'})


# In[8]:


data_taxi['date_time'] = pd.to_datetime(data_taxi['date_time'])
data_taxi.info()


# In[9]:


data_taxi.isna().sum()


# In[10]:


data_taxi.duplicated().sum()


# In[11]:


# Establecer la columna de fecha y hora como índice:

data_taxi.set_index('date_time', inplace=True)


# In[12]:


# Remuestrear datos por horas sumando los pedidos en intervalos de 1 hora:

data_horario = data_taxi.resample('1H').sum()


# In[13]:


# Mostrar las primeras filas de los datos horarios:

data_horario.head()


# ## Análisis

# In[14]:


plt.figure(figsize=(12, 6))
plt.plot(data_horario.index, data_horario['num_orders'], label='Numero de ordenes')
plt.title('Pedidos de taxi por horas a lo largo del tiempo')
plt.xlabel('Tiempo')
plt.ylabel('Numero de ordenes')
plt.legend()
plt.show()


# El gráfico muestra la evolución de los pedidos por hora, permitiendo identificar posibles patrones de tendencia o estacionalidad.
# 

# In[15]:


decomposed = seasonal_decompose(data_horario)

plt.figure(figsize=(6, 8))
plt.subplot(311)

decomposed.trend.plot(ax=plt.gca())
plt.title('TENDENCIA')
plt.subplot(312)

decomposed.seasonal.plot(ax=plt.gca())
plt.title('ESTACIONALIDAD')
plt.subplot(313)

decomposed.resid.plot(ax=plt.gca())
plt.title('RESIDUOS')
plt.tight_layout()




# ### Agregar columnas de análisis para identificar tendencias:

# In[16]:


# Agregar características de tiempo:

data_horario['hour'] = data_horario.index.hour
data_horario['day_of_week'] = data_horario.index.dayofweek # Día de la semana (0=Lunes, 6=Domingo)
data_horario['day_of_month'] = data_horario.index.day
data_horario['month'] = data_horario.index.month
data_horario['is_weekend'] = data_horario['day_of_week'].isin([5, 6]).astype(int)


# #### Análisis promedio por hora del día:

# In[17]:


promedio_por_hora = data_horario.groupby('hour')['num_orders'].mean()

plt.figure(figsize=(12, 6))
plt.plot(promedio_por_hora.index, promedio_por_hora.values, marker='o', label='Pedidos promedio por hora del día')
plt.title('Promedio de pedidos por hora del día', fontsize=16)
plt.xlabel('Hora del día', fontsize=12)
plt.ylabel('Número promedio de pedidos', fontsize=12)
plt.grid()
plt.xticks(range(0, 24))
plt.legend()
plt.show()


# In[18]:


print(promedio_por_hora.sort_values(ascending=False))


# **Análisis por hora del día:**  
# 
# Mayor demanda: La hora con más pedidos es a las 0:00 (144.4 pedidos en promedio). 
# 
# Menor demanda: Los pedidos disminuyen notablemente entre las 6:00 y las 7:00 (25.2 pedidos en promedio).

# #### Análisis promedio por día de la semana:

# In[19]:


promedio_por_dia = data_horario.groupby('day_of_week')['num_orders'].mean()

# Visualizar los promedios por día de la semana:

plt.figure(figsize=(12, 6))
plt.bar(promedio_por_dia.index, promedio_por_dia.values, tick_label=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'], color='skyblue')
plt.title('Promedio de pedidos por día de la semana', fontsize=16)
plt.xlabel('Día de la semana', fontsize=12)
plt.ylabel('Número promedio de pedidos', fontsize=12)
plt.grid(axis='y')
plt.show()

print(promedio_por_dia.sort_values(ascending=False))


# **Análisis por día de la semana:** 
# 
# Mayor demanda: El lunes (90.25 pedidos en promedio) tiene más actividad, posiblemente por el inicio de la semana.  
# 
# Menor demanda: El domingo (78.99 pedidos en promedio) muestra una caída, probablemente debido a menor actividad general.

# In[20]:


data_horario




# ## Formación

# In[21]:


# Crear características:

"""
    Agrega características de retraso (lags) y promedios móviles a un conjunto de datos.
    """

def add_lag_and_rolling_features (data, max_lag,target_column='num_orders', rolling_windows=[3, 6, 12]):
        
    for lag in range(1, max_lag + 1):
        data[f'lag_{lag}'] = data['num_orders'].shift(lag)
        
    #data[f'rolling_mean_{rolling_mean_size}'] = data['num_orders'].rolling(window=rolling_mean_size).mean()#
    
    for window in rolling_windows:
        data[f'rolling_mean_{window}'] = data[target_column].rolling(window=window).mean()
   
    return data.dropna()

# Aplicar la función con max_lag=5 y rolling_mean_size=10

data_horario = add_lag_and_rolling_features(data_horario, target_column='num_orders', max_lag=5, rolling_windows=[3, 6, 12])


# In[22]:


data_horario.head()


# In[23]:


# Visualización de las características agregadas
plt.figure(figsize=(12, 6))
plt.plot(data_horario.index, data_horario['num_orders'], label='Número de órdenes')
plt.plot(data_horario.index, data_horario['rolling_mean_12'], label='Tendencia (Promedio móvil 12h)', color='red')
plt.title('Pedidos de taxi con características de tendencia')
plt.xlabel('Fecha')
plt.ylabel('Número de órdenes')
plt.legend()
plt.grid(True)
plt.show()




# In[24]:


# Dividir en características (X) y objetivo (y):

X = data_horario.drop('num_orders', axis=1)

y = data_horario['num_orders']


# In[25]:


# Dividir en conjuntos de prueba (10%) y entrenamiento-validación (90%)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

# Dividir entrenamiento-validación en entrenamiento (80%) y validación (20%)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=False)

# Mostrar dimensiones de los conjuntos
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")




# In[26]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ## Prueba

# In[27]:


# Modelos básicos:

#model_lr = LinearRegression()
model_dt = DecisionTreeRegressor(random_state=42)
model_rf = RandomForestRegressor(random_state=42, n_estimators=100)
model_lgb = lgb.LGBMRegressor(random_state=42, n_estimators=100)

# Entrenar y evaluar los modelos en validación

models = {
      #'Linear Regression': model_lr,
    'Decision Tree': model_dt,
    'Random Forest': model_rf,
    'LightGBM': model_lgb
}

rmse_scores = {}

for name, model in models.items():
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Predicción en validación
    y_pred = model.predict(X_test)
    # Calcular RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores[name] = rmse
    print(f"{name}: RMSE en validación = {rmse:.2f}")

# Identificar el mejor modelo
best_model_name = min(rmse_scores, key=rmse_scores.get)

print(f"El mejor modelo es {best_model_name} con RMSE = {rmse_scores[best_model_name]:.2f}")


# In[28]:


# Probar el modelo optimizado (ejemplo con LightGBM):

y_pred_test = model_lgb.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"RMSE en el conjunto de prueba: {rmse_test:.2f}")
if rmse_test <= 48:
    print("El modelo cumple con el objetivo de RMSE ≤ 48.")
else:
    print("El modelo NO cumple con el objetivo de RMSE ≤ 48.")


# 
# 
# 



# #  Conclusion

# El objetivo del proyecto era predecir la cantidad de pedidos de taxis para la próxima hora utilizando datos históricos de pedidos de la compañía Sweet Lift Taxi. El modelo debía cumplir con la métrica de desempeño definida: una raíz del error cuadrático medio (RMSE) no mayor a 48 en el conjunto de prueba.



# # Lista de revisión

# - [x]  	
# Jupyter Notebook está abierto.
# - [ ]  El código no tiene errores
# - [ ]  Las celdas con el código han sido colocadas en el orden de ejecución.
# - [ ]  	
# Los datos han sido descargados y preparados.
# - [ ]  Se ha realizado el paso 2: los datos han sido analizados
# - [ ]  Se entrenó el modelo y se seleccionaron los hiperparámetros
# - [ ]  Se han evaluado los modelos. Se expuso una conclusión
# - [ ] La *RECM* para el conjunto de prueba no es más de 48

# In[ ]:




