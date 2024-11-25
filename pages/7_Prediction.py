import streamlit as st
import pandas as pd
from utils.session_state import initialize_session_state, get_data
from components.model import DiabetesModel

def convert_age_to_category(age):
    """Convierte edad real a categoría del dataset"""
    if age < 18:
        return 1
    elif age < 25:
        return 1
    elif age < 30:
        return 2
    elif age < 35:
        return 3
    elif age < 40:
        return 4
    elif age < 45:
        return 5
    elif age < 50:
        return 6
    elif age < 55:
        return 7
    elif age < 60:
        return 8
    elif age < 65:
        return 9
    elif age < 70:
        return 10
    elif age < 75:
        return 11
    elif age < 80:
        return 12
    else:
        return 13

def main():
    st.set_page_config(page_title="Prediction - Diabetes Analysis", layout="wide")
    initialize_session_state()
    st.title('🔮 Diabetes Risk Prediction')
    
    try:
        model = DiabetesModel()
        model.load_model()
        
        if st.session_state.user_type == 'Data Analyst':
            show_technical_prediction(model)
        else:
            show_simple_prediction(model)
    except Exception as e:
        st.error(f'Error: {str(e)}')
        st.info('Please ensure the model is properly trained.')

def prepare_input_dataframe(input_data):
    """Asegura que las características están en el orden correcto"""
    expected_columns = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    
    ordered_data = {col: input_data.get(col, 0) for col in expected_columns}
    return pd.DataFrame([ordered_data])

def show_simple_prediction(model):
    st.write("### Quick Health Assessment")
    
    input_data = {}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        real_age = st.number_input('Age', min_value=18, max_value=120, value=30)
        input_data['Age'] = convert_age_to_category(real_age)
        input_data['BMI'] = st.number_input('BMI', min_value=15.0, max_value=50.0, value=25.0)
        input_data['HighBP'] = 1 if st.radio('High Blood Pressure?', ['No', 'Yes']) == 'Yes' else 0
        input_data['HighChol'] = 1 if st.radio('High Cholesterol?', ['No', 'Yes']) == 'Yes' else 0
    
    with col2:
        input_data['Smoker'] = 1 if st.radio('Smoker?', ['No', 'Yes']) == 'Yes' else 0
        input_data['Stroke'] = 1 if st.radio('Previous Stroke?', ['No', 'Yes']) == 'Yes' else 0
        input_data['HeartDiseaseorAttack'] = 1 if st.radio('Heart Disease/Attack?', ['No', 'Yes']) == 'Yes' else 0
        input_data['PhysActivity'] = 1 if st.radio('Regular Physical Activity?', ['No', 'Yes']) == 'Yes' else 0
    
    with col3:
        input_data['Fruits'] = 1 if st.radio('Daily Fruit Consumption?', ['No', 'Yes']) == 'Yes' else 0
        input_data['Veggies'] = 1 if st.radio('Daily Vegetable Consumption?', ['No', 'Yes']) == 'Yes' else 0
        input_data['GenHlth'] = st.selectbox('General Health', options=[1, 2, 3, 4, 5],
                                            help='1: Excellent, 5: Poor')
    
    if st.button('Get Assessment'):
        input_df = prepare_input_dataframe(input_data)
        show_prediction_results(model, input_df, is_technical=False)

def show_technical_prediction(model):
    st.write("### Technical Prediction Interface")
    
    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(3)
        
        features_col1 = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack']
        features_col2 = ['PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth']
        features_col3 = ['MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Education', 'Income']
        
        with cols[0]:
            for feature in features_col1:
                if feature == 'BMI':
                    input_data[feature] = st.number_input(feature, min_value=15.0, max_value=50.0, value=25.0)
                else:
                    input_data[feature] = st.selectbox(feature, options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        
        with cols[1]:
            for feature in features_col2:
                if feature == 'GenHlth':
                    input_data[feature] = st.selectbox(feature, options=[1, 2, 3, 4, 5])
                else:
                    input_data[feature] = st.selectbox(feature, options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        
        with cols[2]:
            real_age = st.number_input('Age', min_value=18, max_value=120, value=30)
            input_data['Age'] = convert_age_to_category(real_age)
            
            for feature in features_col3:
                if feature == 'Age':
                    continue
                elif feature in ['MentHlth', 'PhysHlth']:
                    input_data[feature] = st.number_input(feature, min_value=0, max_value=30, value=0)
                elif feature in ['Education', 'Income']:
                    input_data[feature] = st.number_input(feature, min_value=1, max_value=13, value=1)
                else:
                    input_data[feature] = st.selectbox(feature, options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        
        submitted = st.form_submit_button("Get Prediction")
        if submitted:
            input_df = prepare_input_dataframe(input_data)
            show_prediction_results(model, input_df, is_technical=True)

def show_prediction_results(model, input_df, is_technical=False):
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    
    st.write('### Prediction Result:')
    
    if prediction[0] == 0:
        st.success('No Diabetes Indicated')
    elif prediction[0] == 1:
        st.warning('Prediabetes Indicated')
    else:
        st.error('Diabetes Indicated')
    
    if is_technical:
        st.write('### Prediction Probabilities:')
        prob_df = pd.DataFrame({
            'Class': ['No Diabetes', 'Prediabetes', 'Diabetes'],
            'Probability': [f"{p:.2f}%" for p in probabilities[0] * 100]
        })
        st.write(prob_df)
        
        st.write('### Feature Importance:')
        fig = model.plot_feature_importance()
        st.pyplot(fig)
    
    show_recommendations(input_df.iloc[0].to_dict(), prediction[0])

def show_recommendations(data, prediction):
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