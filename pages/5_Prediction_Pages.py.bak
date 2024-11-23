# src/pages/prediction_pages.py
import streamlit as st
import pandas as pd
from components.model import DiabetesModel
from utils.session_state import get_data

def show_smart_prediction():
    st.title('🔮 Advanced Diabetes Risk Prediction')
    _show_prediction_interface(is_technical=False)

def show_technical_prediction():
    st.title('🔮 Technical Prediction Interface')
    _show_prediction_interface(is_technical=True)

def _show_prediction_interface(is_technical=False):
    try:
        model = DiabetesModel()
        model.load_model()
        
        st.write("""
        ### Health Assessment
        Please fill in all the information carefully.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input('Age Category', min_value=1, max_value=13)
            bmi = st.number_input('BMI', min_value=0.0, max_value=100.0)
            high_bp = st.selectbox('High Blood Pressure', ['No', 'Yes'])
            
        with col2:
            high_chol = st.selectbox('High Cholesterol', ['No', 'Yes'])
            smoker = st.selectbox('Smoker', ['No', 'Yes'])
            stroke = st.selectbox('Had Stroke', ['No', 'Yes'])
            
        with col3:
            phys_activity = st.selectbox('Regular Physical Activity', ['No', 'Yes'])
            fruits = st.selectbox('Regular Fruit Consumption', ['No', 'Yes'])
            veggies = st.selectbox('Regular Vegetable Consumption', ['No', 'Yes'])
        
        if st.button('Get Prediction'):
            _show_prediction_results(
                model, age, bmi, high_bp, high_chol, smoker, 
                stroke, phys_activity, fruits, veggies,
                is_technical
            )
                
    except FileNotFoundError:
        st.error('Please train the model first in the Modeling section')

def _show_prediction_results(model, age, bmi, high_bp, high_chol, smoker, 
                           stroke, phys_activity, fruits, veggies, is_technical):
    """Muestra los resultados de la predicción"""
    input_data = pd.DataFrame({
        'Age': [age],
        'BMI': [bmi],
        'HighBP': [1 if high_bp == 'Yes' else 0],
        'HighChol': [1 if high_chol == 'Yes' else 0],
        'Smoker': [1 if smoker == 'Yes' else 0],
        'Stroke': [1 if stroke == 'Yes' else 0],
        'PhysActivity': [1 if phys_activity == 'Yes' else 0],
        'Fruits': [1 if fruits == 'Yes' else 0],
        'Veggies': [1 if veggies == 'Yes' else 0]
    })
    
    prediction = model.predict(input_data)
    
    st.write('### Prediction Result:')
    if prediction[0] == 0:
        st.success('Low Risk - No Diabetes Indicated')
    elif prediction[0] == 1:
        st.warning('Moderate Risk - Prediabetes Indicated')
    else:
        st.error('High Risk - Diabetes Indicated')
    
    if is_technical:
        feature_importance = model.get_feature_importance(input_data)
        st.write('### Feature Importance:')
        for feature, importance in feature_importance.items():
            st.write(f"- {feature}: {importance:.2f}% influence")
    
    _show_recommendations(prediction[0], bmi, phys_activity, fruits, veggies)

def _show_recommendations(prediction, bmi, phys_activity, fruits, veggies):
    """Muestra recomendaciones basadas en los resultados"""
    st.write('### Recommendations:')
    if prediction > 0:
        st.write("""
        1. Consult with a healthcare provider
        2. Consider regular health check-ups
        3. Monitor your blood sugar levels
        """)
        if bmi > 25:
            st.write("4. Focus on weight management")
        if phys_activity == 'No':
            st.write("5. Increase physical activity")
        if not (fruits == 'Yes' and veggies == 'Yes'):
            st.write("6. Improve dietary habits")
    else:
        st.write("""
        1. Maintain your healthy lifestyle
        2. Continue regular check-ups
        3. Stay physically active
        """)