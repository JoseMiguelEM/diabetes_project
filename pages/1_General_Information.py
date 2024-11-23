import streamlit as st
from utils.session_state import initialize_session_state

def main():
    st.set_page_config(
        page_title="General Information - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title('📚 Understanding Diabetes')
    
    st.write("""
    ### Key Risk Factors
    
    Understanding these factors can help you make better health choices:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        #### Controllable Factors:
        - Blood Pressure
        - Body Mass Index (BMI)
        - Physical Activity
        - Diet (Fruits & Vegetables)
        - Smoking Habits
        """)
    
    with col2:
        st.write("""
        #### Non-Controllable Factors:
        - Age
        - Family History
        - Previous Health Conditions
        """)
    
    # Additional Information Section
    st.write("""
    ### Understanding Diabetes Types
    
    Diabetes comes in different forms:
    1. Type 1 Diabetes
    2. Type 2 Diabetes
    3. Gestational Diabetes
    4. Prediabetes
    """)
    
    # Prevention Tips
    st.write("""
    ### Prevention Tips
    
    Simple steps to reduce your risk:
    - Maintain a healthy weight
    - Exercise regularly
    - Eat a balanced diet
    - Monitor your blood pressure
    - Get regular check-ups
    """)

if __name__ == "__main__":
    main()