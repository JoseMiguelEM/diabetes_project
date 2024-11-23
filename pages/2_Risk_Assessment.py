# src/pages/home.py
import streamlit as st
from utils.session_state import get_data
from components.eda.data_loader import check_data_quality

def show_general_home():
    st.title('🏥 Welcome to Diabetes Analysis System')
    st.write("""
    ### What can you do here?
    
    This system helps you:
    - Understand diabetes risk factors
    - Assess your potential diabetes risk
    - Get personalized recommendations
    - Make informed health decisions
    """)
    
    # Estadísticas generales simplificadas
    df = get_data()
    info, stats = check_data_quality(df)
    st.write("### Quick Facts About Diabetes")
    col1, col2 = st.columns(2)
    with col1:
        st.info("Early detection and prevention are key to managing diabetes effectively.")
    with col2:
        st.info("Lifestyle changes can significantly reduce diabetes risk.")

def show_technical_home():
    st.title('🏥 Diabetes Analysis System - Technical Overview')
    st.write("""
    ## Welcome to the Technical Dashboard
    
    This system provides:
    - Data exploration and analysis
    - Pattern visualization
    - Predictive modeling
    - Risk assessment
    """)
    
    # Mostrar estadísticas técnicas
    df = get_data(use_processed=False)
    info, stats = check_data_quality(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", info['total_rows'])
    with col2:
        st.metric("Diabetes Cases", len(df[df['Diabetes_012'] == 2]))
    with col3:
        st.metric("Variables Analyzed", info['total_columns'])