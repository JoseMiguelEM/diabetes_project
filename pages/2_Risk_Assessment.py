import streamlit as st
from utils.session_state import initialize_session_state

def main():
    st.set_page_config(
        page_title="Risk Assessment - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title('🎯 Diabetes Risk Assessment')
    
    st.write("""
    ### Quick Health Check
    
    Answer these simple questions to get a basic risk assessment.
    All information is processed locally and not stored.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_group = st.selectbox('Age Group', 
            ['20-30', '31-40', '41-50', '51-60', '60+'])
        bmi = st.number_input('BMI (Body Mass Index)', 15.0, 50.0, 25.0)
        high_bp = st.radio('Do you have high blood pressure?', ['No', 'Yes'])
    
    with col2:
        exercise = st.radio('Do you exercise regularly?', ['No', 'Yes'])
        healthy_diet = st.radio('Do you maintain a healthy diet?', ['No', 'Yes'])
        smoking = st.radio('Do you smoke?', ['No', 'Yes'])
    
    if st.button('Check My Risk'):
        # Calcular factores de riesgo
        risk_factors = 0
        risk_factors += 1 if int(age_group.split('-')[0]) > 40 else 0
        risk_factors += 1 if bmi > 25 else 0
        risk_factors += 1 if high_bp == 'Yes' else 0
        risk_factors += 1 if exercise == 'No' else 0
        risk_factors += 1 if healthy_diet == 'No' else 0
        risk_factors += 1 if smoking == 'Yes' else 0
        
        st.write("### Your Risk Assessment")
        if risk_factors <= 1:
            st.success("Low Risk - Keep up the good work!")
        elif risk_factors <= 3:
            st.warning("Moderate Risk - Consider lifestyle changes")
        else:
            st.error("High Risk - Consult with a healthcare provider")
        
        # Mostrar recomendaciones
        st.write("### Recommendations:")
        recommendations = []
        if bmi > 25:
            recommendations.append("Consider maintaining a healthy weight")
        if exercise == 'No':
            recommendations.append("Include regular physical activity")
        if healthy_diet == 'No':
            recommendations.append("Improve your diet with more fruits and vegetables")
        if smoking == 'Yes':
            recommendations.append("Consider quitting smoking")
            
        for rec in recommendations:
            st.write(f"- {rec}")

if __name__ == "__main__":
    main()