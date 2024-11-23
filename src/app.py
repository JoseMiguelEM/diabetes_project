# src/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from components.model import DiabetesModel

# Configuración de la página
st.set_page_config(page_title="Diabetes Analysis", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('data/dataset.csv')
    return df

df = load_data()

# Inicializar el modelo
model = DiabetesModel()

# Sidebar
st.sidebar.title('Navegación')
page = st.sidebar.radio('Selecciona una página:', ['Home', 'EDA', 'Modelado', 'Predicción'])

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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", len(df))
    with col2:
        st.metric("Casos de Diabetes", len(df[df['Diabetes_012'] == 2]))
    with col3:
        st.metric("Variables Analizadas", len(df.columns))

elif page == 'EDA':
    st.title('📊 Análisis Exploratorio de Datos')
    
    # Mostrar estadísticas básicas
    if st.checkbox('Mostrar estadísticas básicas'):
        st.write(df.describe())
    
    # Selector de variables para visualización
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Distribución de Variables')
        variable = st.selectbox('Selecciona una variable:', df.columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if df[variable].dtype in ['int64', 'float64']:
            sns.histplot(df[variable], kde=True)
        else:
            sns.countplot(data=df, x=variable)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader('Matriz de Correlación')
        if st.checkbox('Mostrar matriz de correlación'):
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(fig)

elif page == 'Modelado':
    st.title('🤖 Modelado Predictivo')
    
    st.write("""
    ### Entrenamiento del Modelo
    El modelo utiliza Random Forest Classifier para predecir el riesgo de diabetes.
    """)
    
    if st.button('Entrenar Nuevo Modelo'):
        # Preparar datos
        X = df.drop('Diabetes_012', axis=1)
        y = df['Diabetes_012']
        
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