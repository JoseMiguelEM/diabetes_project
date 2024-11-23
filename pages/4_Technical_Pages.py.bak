# src/pages/technical_pages.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.session_state import get_data
from data_processing.dataset_processor import DatasetProcessor
from components.eda.correlation import get_correlation_matrix, get_feature_correlations_with_target, get_high_correlation_pairs
from components.eda.visualization import plot_numeric_distribution, plot_correlation_matrix, plot_class_distribution
from components.eda.data_loader import get_feature_types
from components.eda.balance import analyze_class_balance

def show_data_preparation():
    st.title('🔄 Data Preparation')
    
    processor = DatasetProcessor()
    
    st.write("""
    ### Dataset Processing Steps
    Follow these steps to prepare the optimal dataset:
    """)
    
    # Análisis inicial
    st.subheader("1. Initial Analysis")
    if st.button("Analyze Original Dataset"):
        initial_dist = processor.load_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Class Distribution:")
            st.write(pd.DataFrame({
                'Class': initial_dist['distribution'].keys(),
                'Count': initial_dist['distribution'].values(),
                'Percentage': [f"{v:.2f}%" for v in initial_dist['percentages'].values()]
            }))
        
        with col2:
            fig, ax = plt.subplots()
            plt.pie(
                initial_dist['distribution'].values(),
                labels=initial_dist['distribution'].keys(),
                autopct='%1.1f%%'
            )
            st.pyplot(fig)
    
    # Mostrar resto de las funciones de preparación de datos...

def show_basic_eda():
    st.title('📊 Basic Exploratory Data Analysis')
    df = get_data(use_processed=False)
    
    if st.checkbox('Show Basic Statistics'):
        st.write(df.describe())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Variable Distribution')
        variable = st.selectbox('Select a variable:', df.columns)
        fig = plot_numeric_distribution(df, variable)
        st.pyplot(fig)
    
    with col2:
        st.subheader('Correlation Matrix')
        if st.checkbox('Show correlation matrix'):
            corr_matrix = get_correlation_matrix(df)
            fig = plot_correlation_matrix(corr_matrix)
            st.pyplot(fig)

def show_advanced_eda():
    st.title('🔬 Advanced Exploratory Analysis')
    df = get_data(use_processed=False)
    
    feature_types = get_feature_types(df)
    
    # Análisis de Balance de Clases
    st.header('Class Balance Analysis')
    balance_stats = analyze_class_balance(df, 'Diabetes_012')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Class distribution:")
        st.write(balance_stats['counts'])
    with col2:
        fig = plot_class_distribution(df['Diabetes_012'])
        st.pyplot(fig)
    
    # Mostrar resto del análisis avanzado...

def show_modeling():
    st.title('🤖 Model Training')
    df = get_data(use_processed=True)
    
    st.write("""
    ### Model Training Configuration
    Configure and train the Random Forest model with class balancing.
    """)
    
    # Opciones de entrenamiento y visualización de resultados...