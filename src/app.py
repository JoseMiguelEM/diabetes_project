# src/app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from components.model import DiabetesModel
from components.eda.data_loader import load_dataset, check_data_quality, get_feature_types
from components.eda.univariate import analyze_numeric_variables, analyze_categorical_variables, analyze_binary_variables
from components.eda.correlation import get_correlation_matrix, get_feature_correlations_with_target, get_high_correlation_pairs
from components.eda.balance import analyze_class_balance, apply_balancing_techniques
from components.eda.visualization import (
    plot_numeric_distribution, 
    plot_correlation_matrix,
    plot_class_distribution,
    plot_balancing_results
)

# Configuración de la página
st.set_page_config(page_title="Diabetes Analysis", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    try:
        # Intenta cargar usando la ruta por defecto
        df = load_dataset()
    except FileNotFoundError:
        # Si falla, busca en la ubicación relativa a app.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        filepath = os.path.join(project_root, 'data', 'dataset.csv')
        
        if not os.path.exists(filepath):
            st.error(f"No se encuentra el archivo dataset.csv. Por favor, verifica que existe en la carpeta 'data'")
            st.stop()
            
        df = load_dataset(filepath)
    
    return df

df = load_data()

# Inicializar el modelo
model = DiabetesModel()

# Sidebar
st.sidebar.title('Navegación')
page = st.sidebar.radio('Selecciona una página:', ['Home', 'EDA Básico', 'EDA Avanzado', 'Modelado', 'Predicción'])

if page == 'Home':
    st.title('🏥 Sistema de Análisis de Diabetes')
    st.write("""
    ## Bienvenido a nuestro Dashboard de Análisis de Diabetes
    
    Este sistema te permite:
    - Explorar datos relacionados con la diabetes
    - Visualizar patrones y correlaciones
    - Entrenar modelos predictivos
    - Realizar predicciones de riesgo de diabetes
    """)
    
    # Mostrar algunas estadísticas generales
    info, stats = check_data_quality(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", info['total_rows'])
    with col2:
        st.metric("Casos de Diabetes", len(df[df['Diabetes_012'] == 2]))
    with col3:
        st.metric("Variables Analizadas", info['total_columns'])

elif page == 'EDA Básico':
    st.title('📊 Análisis Exploratorio de Datos Básico')
    
    # Mostrar estadísticas básicas
    if st.checkbox('Mostrar estadísticas básicas'):
        st.write(df.describe())
    
    # Selector de variables para visualización
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Distribución de Variables')
        variable = st.selectbox('Selecciona una variable:', df.columns)
        
        fig = plot_numeric_distribution(df, variable)
        st.pyplot(fig)
    
    with col2:
        st.subheader('Matriz de Correlación')
        if st.checkbox('Mostrar matriz de correlación'):
            corr_matrix = get_correlation_matrix(df)
            fig = plot_correlation_matrix(corr_matrix)
            st.pyplot(fig)

elif page == 'EDA Avanzado':
    st.title('🔬 Análisis Exploratorio Avanzado')
    
    # Análisis de tipos de variables
    feature_types = get_feature_types(df)
    
    # Análisis de Balance de Clases
    st.header('Balance de Clases')
    balance_stats = analyze_class_balance(df, 'Diabetes_012')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribución de clases:")
        st.write(balance_stats['counts'])
    with col2:
        fig = plot_class_distribution(df['Diabetes_012'])
        st.pyplot(fig)
    
    # Análisis de Variables Numéricas
    st.header('Análisis de Variables Numéricas')
    selected_numeric = st.selectbox(
        'Selecciona una variable numérica:',
        feature_types['numeric']
    )
    
    if selected_numeric:
        num_stats = analyze_numeric_variables(df, [selected_numeric])[selected_numeric]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Estadísticas:")
            st.write(f"Media: {num_stats['mean']:.2f}")
            st.write(f"Mediana: {num_stats['median']:.2f}")
            st.write(f"Desv. Est.: {num_stats['std']:.2f}")
            st.write(f"Outliers: {num_stats['outliers']['count']} ({num_stats['outliers']['percentage']:.2f}%)")
        
        with col2:
            fig = plot_numeric_distribution(df, selected_numeric)
            st.pyplot(fig)
    
    # Análisis de Correlaciones
    st.header('Análisis de Correlaciones')
    st.subheader('Correlaciones con Variable Objetivo')
    target_corr = get_feature_correlations_with_target(df, 'Diabetes_012')
    st.write(target_corr)
    
    if st.checkbox('Mostrar pares de alta correlación'):
        high_corr = get_high_correlation_pairs(df)
        st.write(high_corr)

elif page == 'Modelado':
    st.title('🤖 Modelado Predictivo')
    
    st.write("""
    ### Entrenamiento del Modelo
    El modelo utiliza Random Forest Classifier con técnicas de balanceo de clases.
    """)
    
    # Opciones de balanceo
    balance_technique = st.selectbox(
        "Selecciona técnica de balanceo:",
        ["Sin balanceo", "ClusterCentroids", "NearMiss", "TomekLinks"]
    )
    
    if st.button('Entrenar Nuevo Modelo'):
        # Preparar datos
        X = df.drop('Diabetes_012', axis=1)
        y = df['Diabetes_012']
        
        if balance_technique != "Sin balanceo":
            # Aplicar técnica de balanceo seleccionada
            results = apply_balancing_techniques(X, y)
            st.write(f"Resultados del balanceo con {balance_technique}:")
            st.write(results[balance_technique])
            
            # Visualizar resultados del balanceo
            fig = plot_balancing_results({balance_technique: results[balance_technique]})
            st.pyplot(fig)
        
        # Entrenar modelo
        with st.spinner('Entrenando modelo...'):
            score = model.train(X, y)
            st.success(f'¡Modelo entrenado exitosamente! Precisión: {score:.2f}')

elif page == 'Predicción':
    st.title('🔮 Predicción de Diabetes')
    
    try:
        model.load_model()
        st.info('Ingresa los datos del paciente para realizar una predicción')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input('Edad', min_value=1, max_value=13)
            bmi = st.number_input('BMI', min_value=0.0, max_value=100.0)
            high_bp = st.selectbox('Presión Alta', [0, 1])
            
        with col2:
            high_chol = st.selectbox('Colesterol Alto', [0, 1])
            smoker = st.selectbox('Fumador', [0, 1])
            stroke = st.selectbox('Stroke', [0, 1])
            
        with col3:
            phys_activity = st.selectbox('Actividad Física', [0, 1])
            fruits = st.selectbox('Consume Frutas', [0, 1])
            veggies = st.selectbox('Consume Vegetales', [0, 1])
        
        if st.button('Realizar Predicción'):
            # Crear dataframe con los inputs
            input_data = pd.DataFrame({
                'Age': [age],
                'BMI': [bmi],
                'HighBP': [high_bp],
                'HighChol': [high_chol],
                'Smoker': [smoker],
                'Stroke': [stroke],
                'PhysActivity': [phys_activity],
                'Fruits': [fruits],
                'Veggies': [veggies]
            })
            
            # Realizar predicción
            prediction = model.predict(input_data)
            
            # Mostrar resultado
            st.write('### Resultado:')
            if prediction[0] == 0:
                st.success('No Diabetes')
            elif prediction[0] == 1:
                st.warning('Prediabetes')
            else:
                st.error('Diabetes')
                
    except FileNotFoundError:
        st.error('Por favor entrena el modelo primero en la página de Modelado')

# Footer
st.markdown("---")
st.markdown("Developed for Data Analysis Course 2024-2")