import streamlit as st
import pandas as pd
from utils.session_state import initialize_session_state
from components.model import DiabetesModel

def main():
    st.set_page_config(
        page_title="Prediction - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title('🔮 Diabetes Risk Prediction')
    
    try:
        model = DiabetesModel()
        model.load_model()
        
        # Interface según tipo de usuario
        if st.session_state.user_type == 'Data Analyst':
            show_technical_prediction(model)
        else:
            show_simple_prediction(model)
                
    except FileNotFoundError:
        st.error('Model not found. Please train the model first in the Modeling section.')

def show_simple_prediction(model):
    """Interfaz de predicción simplificada para usuarios generales"""
    st.write("""
    ### Quick Health Assessment
    Fill in your health information below to get a diabetes risk assessment.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.selectbox('Age Category', 
            options=list(range(1, 14)),
            help='1: 18-24, 2: 25-29, 3: 30-34, ..., 13: 80 or older')
        bmi = st.number_input('BMI', 
            min_value=15.0, 
            max_value=50.0, 
            value=25.0,
            help='Body Mass Index')
        high_bp = st.radio('Do you have high blood pressure?', 
            options=['No', 'Yes'])
    
    with col2:
        high_chol = st.radio('Do you have high cholesterol?', 
            options=['No', 'Yes'])
        smoker = st.radio('Are you a smoker?', 
            options=['No', 'Yes'])
        stroke = st.radio('Have you ever had a stroke?', 
            options=['No', 'Yes'])
    
    with col3:
        phys_activity = st.radio('Do you exercise regularly?', 
            options=['No', 'Yes'])
        fruits = st.radio('Do you eat fruit daily?', 
            options=['No', 'Yes'])
        veggies = st.radio('Do you eat vegetables daily?', 
            options=['No', 'Yes'])
    
    if st.button('Get Prediction'):
        show_prediction_results(model, {
            'Age': age,
            'BMI': bmi,
            'HighBP': 1 if high_bp == 'Yes' else 0,
            'HighChol': 1 if high_chol == 'Yes' else 0,
            'Smoker': 1 if smoker == 'Yes' else 0,
            'Stroke': 1 if stroke == 'Yes' else 0,
            'PhysActivity': 1 if phys_activity == 'Yes' else 0,
            'Fruits': 1 if fruits == 'Yes' else 0,
            'Veggies': 1 if veggies == 'Yes' else 0
        }, is_technical=False)

def show_technical_prediction(model):
    """Interfaz de predicción técnica para analistas"""
    st.write("""
    ### Technical Prediction Interface
    Input values for detailed model prediction and analysis.
    """)
    
    # Crear formulario para entrada de datos
    with st.form("prediction_form"):
        cols = st.columns(3)
        
        input_data = {}
        features = ['Age', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 
                   'PhysActivity', 'Fruits', 'Veggies']
        
        for i, feature in enumerate(features):
            col_idx = i % 3
            with cols[col_idx]:
                if feature in ['Age']:
                    input_data[feature] = st.number_input(
                        f"{feature}", 
                        min_value=1, 
                        max_value=13,
                        value=1
                    )
                elif feature == 'BMI':
                    input_data[feature] = st.number_input(
                        f"{feature}", 
                        min_value=15.0,
                        max_value=50.0,
                        value=25.0
                    )
                else:
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        options=[0, 1],
                        format_func=lambda x: 'Yes' if x == 1 else 'No'
                    )
        
        submitted = st.form_submit_button("Get Prediction")
        
        if submitted:
            show_prediction_results(model, input_data, is_technical=True)

def show_prediction_results(model, input_data, is_technical=False):
    """Muestra los resultados de la predicción"""
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    
    st.write('### Prediction Result:')
    
    # Mostrar resultado principal
    if prediction[0] == 0:
        st.success('No Diabetes Indicated')
    elif prediction[0] == 1:
        st.warning('Prediabetes Indicated')
    else:
        st.error('Diabetes Indicated')
    
    # Mostrar probabilidades si es vista técnica
    if is_technical:
        st.write('### Prediction Probabilities:')
        prob_df = pd.DataFrame({
            'Class': ['No Diabetes', 'Prediabetes', 'Diabetes'],
            'Probability': probabilities[0] * 100
        })
        st.write(prob_df)
        
        # Mostrar importancia de características
        st.write('### Feature Importance:')
        importance = model.get_feature_importance()
        fig = model.plot_feature_importance()
        st.pyplot(fig)
    
    # Mostrar recomendaciones
    show_recommendations(input_data, prediction[0])

def show_recommendations(data, prediction):
    """Muestra recomendaciones basadas en los resultados"""
    st.write('### Recommendations:')
    
    general_recs = []
    if prediction > 0:
        general_recs.extend([
            "Schedule a consultation with a healthcare provider",
            "Monitor your blood sugar levels regularly",
            "Consider getting a comprehensive health check-up"
        ])
    
    specific_recs = []
    if data['BMI'] > 25:
        specific_recs.append("Focus on maintaining a healthy weight through diet and exercise")
    if data['PhysActivity'] == 0:
        specific_recs.append("Incorporate regular physical activity into your routine")
    if data['Smoker'] == 1:
        specific_recs.append("Consider a smoking cessation program")
    if data['Fruits'] == 0 or data['Veggies'] == 0:
        specific_recs.append("Increase your intake of fruits and vegetables")
    if data['HighBP'] == 1:
        specific_recs.append("Monitor and manage your blood pressure")
    if data['HighChol'] == 1:
        specific_recs.append("Work on managing your cholesterol levels")
    
    # Mostrar recomendaciones
    if general_recs:
        st.write("General Recommendations:")
        for rec in general_recs:
            st.write(f"- {rec}")
    
    if specific_recs:
        st.write("Specific Recommendations:")
        for rec in specific_recs:
            st.write(f"- {rec}")

if __name__ == "__main__":
    main()