import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de la temperatura por ciudad ''')
st.image("cl.jpeg", caption="Predicción de la temperatura (Vamos a ver que tal el clima).")

st.header('Descripción de la actividad')

def user_input_features():
    # Entradas del usuario
    City = st.number_input('Ciudad (Acuña=2, Aguascalientes=0, Acapulco=1):', min_value=1, max_value=3, value=1, step=1)
    Mes = st.number_input('Mes (1-12):', min_value=1, max_value=12, value=1, step=1)
    Año = st.number_input('Año (1843-2100):', min_value=1843, max_value=2100, value=2025, step=1)

    # Nombres igual que en el dataset
    user_input_data = {
        'City': City,
        'Mes': Mes,
        'Año': Año
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

# Cargar base de datos real
datos = pd.read_csv('temp.csv', encoding='latin-1')
X = datos.drop(columns='Temperatura')
y = datos['Temperatura']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613090)
LR = LinearRegression()
LR.fit(X_train, y_train)

b1 = LR.coef_
b0 = LR.intercept_

prediccion = b0 + b1[0]*df['City'] + b1[1]*df['Mes'] + b1[2]*df['Año']

st.subheader('Predicción de temperatura')
st.write('La temperatura será...', prediccion, '°C')
