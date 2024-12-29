import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import kaggle
import kagglehub
from fpdf import FPDF
import webbrowser 

st.logo("logo_ucg.png")

# Configuración de la API de Kaggle
kaggle.api.authenticate()

# Título de la aplicación
st.title('Herramienta de Análisis de Datos Interactiva')

# Carga de Datasets desde Kaggle
dataset_url = st.text_input('Introduce la URL del dataset de Kaggle')
if dataset_url:
    path = kaggle.api.dataset_download_files(dataset_url, path='datasets/', unzip=True)
    # Download latest version
    #path = kagglehub.dataset_download("vedaantsingh/comprehensive-cryptocurrency-market-data")
    #print("Path to dataset files:", path)
    st.write('Path to dataset files:', path)
    st.success('Dataset descargado y descomprimido en datasets')

# Cargar el dataset en un DataFrame
uploaded_file = st.file_uploader('Cargar archivo CSV', type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write('Datos del Dataset', df.head())

    # Módulo de EDA
    st.header('Análisis Exploratorio de Datos (EDA)')
    st.write('Descripción del Dataset', df.describe())
    st.write('Información del Dataset', df.info())
    
    # Visualizaciones
    st.subheader('Visualizaciones')
    st.write('Distribución de Variables Numéricas')
    st.bar_chart(df.select_dtypes(include=['float64', 'int64']).describe().transpose())
    
    st.write('Matriz de Correlación')
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    # Módulo de Regresiones
    st.header('Análisis de Regresión')
    target = st.selectbox('Selecciona la variable objetivo', df.columns)
    features = st.multiselect('Selecciona las variables predictoras', df.columns)
    
    if st.button('Ejecutar Regresión Lineal'):
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write('Error Cuadrático Medio (MSE):', mse)
        st.write('Coeficientes del Modelo:', model.coef_)

    # Generación de Informes
    st.header('Generación de Informes')
    if st.button('Generar Informe Ejecutivo'):
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        with open('informe_ejecutivo.txt', 'w') as f:
            f.write('Informe Ejecutivo\n')
            f.write('=================\n\n')
            f.write('Descripción del Dataset\n')
            f.write(str(df.describe()))
            f.write('\n\nMatriz de Correlación\n')
            f.write(str(corr))
            f.write('\n\nResultados de la Regresión\n')
            f.write('Error Cuadrático Medio (MSE): ' + str(mse) + '\n')
            f.write('Coeficientes del Modelo: ' + str(model.coef_) + '\n')
        # Crear una instancia de FPDF 
        pdf = FPDF() 
        # Agregar una página 
        pdf.add_page() 
        # Establecer fuente 
        pdf.set_font("Arial", size=12) 
        # Abrir el archivo .txt
        with open('informe_ejecutivo.txt', 'r') as file: 
             for line in file: 
                 pdf.cell(200, 10, txt=line, ln=True) 
        # Guardar el PDF 
        pdf.output("informe_ejecutivo.pdf")
        # Mostrar el PDF
        webbrowser.open("informe_ejecutivo.pdf")
        st.success('Informe generado y guardado como informe_ejecutivo.pdf')

