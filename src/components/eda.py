# src/components/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Carga y prepara el dataset"""
    return pd.read_csv('data/dataset.csv')

def plot_distribution(data, column):
    """Genera gráfico de distribución para una variable"""
    fig, ax = plt.subplots(figsize=(10, 6))
    if data[column].dtype in ['int64', 'float64']:
        sns.histplot(data[column], kde=True)
    else:
        sns.countplot(data=data, x=column)
    plt.title(f'Distribución de {column}')
    plt.xticks(rotation=45)
    return fig

def plot_correlation_matrix(data):
    """Genera matriz de correlación"""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    return fig

# src/components/model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_model(X, y):
    """Entrena el modelo de Random Forest"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, model.score(X_test, y_test)

def save_model(model, filename='diabetes_model.pkl'):
    """Guarda el modelo entrenado"""
    joblib.dump(model, f'models/{filename}')

def load_model(filename='diabetes_model.pkl'):
    """Carga el modelo guardado"""
    return joblib.load(f'models/{filename}')

# src/app.py
import streamlit as st
import pandas as pd
from components.eda import load_data, plot_distribution, plot_correlation_matrix
from components.model import train_model, save_model, load_model

def main():
    st.set_page_config(
        page_title="Diabetes Analysis Dashboard",
        page_icon="🏥",
        layout="wide"
    )
    
    # Sidebar navigation
    st.sidebar.title("Navegación")
    page = st.sidebar.radio("Selecciona una página:", ["Home", "EDA", "Modelado", "Predicción"])
    
    if page == "Home":
        st.title("🏥 Análisis de Diabetes")
        st.write("""
        ## Bienvenido al Dashboard de Análisis de Diabetes
        
        Este dashboard permite:
        - Explorar los datos y sus distribuciones
        - Visualizar relaciones entre variables
        - Entrenar y usar modelos predictivos
        """)
        
    elif page == "EDA":
        st.title("📊 Análisis Exploratorio")
        
        # Cargar datos
        df = load_data()
        
        # Mostrar distribuciones
        st.subheader("Distribución de Variables")
        column = st.selectbox("Selecciona una variable:", df.columns)
        st.pyplot(plot_distribution(df, column))
        
        # Mostrar correlaciones
        if st.checkbox("Mostrar Matriz de Correlación"):
            st.pyplot(plot_correlation_matrix(df))
            
    elif page == "Modelado":
        st.title("🤖 Entrenamiento del Modelo")
        
        df = load_data()
        if st.button("Entrenar Nuevo Modelo"):
            with st.spinner("Entrenando..."):
                X = df.drop('Diabetes_012', axis=1)
                y = df['Diabetes_012']
                model, score = train_model(X, y)
                save_model(model)
                st.success(f"¡Modelo entrenado! Precisión: {score:.2f}")
                
    elif page == "Predicción":
        st.title("🔮 Predicción de Diabetes")
        
        try:
            model = load_model()
            st.success("Modelo cargado correctamente")
            
            # Formulario de predicción
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    age = st.number_input("Edad", 1, 13)
                    bmi = st.number_input("BMI", 0.0, 100.0)
                    
                with col2:
                    high_bp = st.selectbox("Presión Alta", [0, 1])
                    high_chol = st.selectbox("Colesterol Alto", [0, 1])
                
                submitted = st.form_submit_button("Predecir")
                
                if submitted:
                    input_data = pd.DataFrame({
                        'Age': [age],
                        'BMI': [bmi],
                        'HighBP': [high_bp],
                        'HighChol': [high_chol]
                    })
                    
                    prediction = model.predict(input_data)
                    
                    if prediction[0] == 0:
                        st.success("No Diabetes")
                    elif prediction[0] == 1:
                        st.warning("Prediabetes")
                    else:
                        st.error("Diabetes")
                        
        except FileNotFoundError:
            st.error("No se encontró el modelo. Por favor, entrena uno nuevo en la página de Modelado.")

if __name__ == "__main__":
    main()