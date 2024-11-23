import streamlit as st
from utils.session_state import initialize_session_state

def setup_page():
    st.set_page_config(
        page_title="Diabetes Analysis System",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    setup_page()
    initialize_session_state()
    
    # Configuración del tipo de usuario
    if 'user_type' not in st.session_state:
        st.session_state.user_type = 'General User'
    
    # Selector de tipo de usuario
    st.sidebar.title('User Type')
    user_type = st.sidebar.radio(
        "Select user type:",
        options=['General User', 'Data Analyst'],
        key='user_type_radio'
    )
    st.session_state.user_type = user_type
    
    # Contenido principal
    st.title('🏥 Welcome to Diabetes Analysis System')
    
    if user_type == 'General User':
        st.write("""
        ### What can you do here?
        
        This system helps you:
        - Understand diabetes risk factors
        - Assess your potential diabetes risk
        - Get personalized recommendations
        - Make informed health decisions
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("Early detection and prevention are key to managing diabetes effectively.")
        with col2:
            st.info("Lifestyle changes can significantly reduce diabetes risk.")
            
    else:
        st.write("""
        ### Technical Dashboard Features
        
        This system provides:
        - Advanced data analysis tools
        - Dataset processing and optimization
        - Statistical analysis and visualization
        - Machine learning model development
        - Comprehensive prediction system
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("Access complete dataset analysis and processing tools.")
        with col2:
            st.info("Utilize machine learning models for prediction.")

    st.markdown("---")
    st.markdown("Developed for Data Analysis Course 2024-2")

if __name__ == "__main__":
    main()