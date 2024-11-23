import streamlit as st
from utils.session_state import initialize_session_state, get_data
from components.eda.visualization import plot_numeric_distribution, plot_correlation_matrix
from components.eda.correlation import get_correlation_matrix, get_high_correlation_pairs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.set_page_config(
        page_title="Basic EDA - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Verificar tipo de usuario
    if st.session_state.user_type != 'Data Analyst':
        st.warning("This page is only accessible to Data Analysts")
        return
    
    st.title('📊 Basic Exploratory Data Analysis')
    
    try:
        df = get_data(use_processed=True)
        
        # Univariate Analysis
        st.header("Univariate Analysis")
        
        # Variable Selection
        col1, col2 = st.columns([1, 3])
        with col1:
            variable = st.selectbox(
                'Select Variable',
                options=df.columns.tolist(),
                key='univariate_var'
            )
        
        with col2:
            fig = plot_numeric_distribution(df, variable)
            st.pyplot(fig)
        
        # Statistical Summary
        st.subheader("Statistical Summary")
        st.write(df[variable].describe())
        
        # Correlation Analysis
        st.header("Correlation Analysis")
        
        # Correlation Matrix
        st.subheader("Correlation Matrix")
        corr_matrix = get_correlation_matrix(df)
        fig = plot_correlation_matrix(corr_matrix)
        st.pyplot(fig)
        
        # High Correlations
        st.subheader("High Correlation Pairs")
        high_corr = get_high_correlation_pairs(df, threshold=0.5)
        if not high_corr.empty:
            st.write(high_corr)
        else:
            st.write("No high correlations found (threshold > 0.5)")
        
        # Target Variable Analysis
        st.header("Target Variable Analysis")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='Diabetes_012')
        plt.title('Distribution of Diabetes Classes')
        st.pyplot(fig)
        
        # Target variable proportions
        st.write("Class Distribution:")
        st.write(df['Diabetes_012'].value_counts(normalize=True).mul(100).round(2))
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.info("Please ensure the dataset is properly processed")

if __name__ == "__main__":
    main()