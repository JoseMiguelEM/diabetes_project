import streamlit as st
from utils.session_state import initialize_session_state, get_data
from components.eda.correlation import get_feature_correlations_with_target
from components.eda.univariate import analyze_numeric_variables, analyze_categorical_variables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.set_page_config(
        page_title="Advanced EDA - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Verificar tipo de usuario
    if st.session_state.user_type != 'Data Analyst':
        st.warning("This page is only accessible to Data Analysts")
        return
    
    st.title('🔬 Advanced Exploratory Analysis')
    
    try:
        df = get_data(use_processed=True)
        
        # Feature Analysis by Target
        st.header("Feature Analysis by Diabetes Status")
        
        # Correlation with target
        st.subheader("Feature Correlations with Diabetes")
        target_correlations = get_feature_correlations_with_target(df, 'Diabetes_012')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        target_correlations.plot(kind='bar')
        plt.title('Feature Correlations with Diabetes')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Detailed Feature Analysis
        st.header("Detailed Feature Analysis")
        
        # Numeric Variables
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_stats = analyze_numeric_variables(df, numeric_cols)
        
        st.subheader("Numeric Variables Analysis")
        for col, stats in numeric_stats.items():
            if col != 'Diabetes_012':  # Excluir variable objetivo
                with st.expander(f"Analysis of {col}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Basic Statistics:")
                        basic_stats = {
                            'Mean': stats['mean'],
                            'Median': stats['median'],
                            'Std Dev': stats['std'],
                            'Min': stats['min'],
                            'Max': stats['max']
                        }
                        st.write(pd.DataFrame([basic_stats]).T)
                    
                    with col2:
                        st.write("Distribution Characteristics:")
                        dist_stats = {
                            'Skewness': stats['skewness'],
                            'Kurtosis': stats['kurtosis'],
                            'Outliers Count': stats['outliers']['count'],
                            'Outliers %': f"{stats['outliers']['percentage']:.2f}%"
                        }
                        st.write(pd.DataFrame([dist_stats]).T)
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Distribution plot
                    sns.histplot(data=df, x=col, hue='Diabetes_012', multiple="stack", ax=ax1)
                    ax1.set_title(f'Distribution of {col} by Diabetes Status')
                    
                    # Box plot
                    sns.boxplot(data=df, y=col, x='Diabetes_012', ax=ax2)
                    ax2.set_title(f'Box Plot of {col} by Diabetes Status')
                    
                    st.pyplot(fig)
        
        # Bivariate Analysis
        st.header("Bivariate Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox('Select first variable', numeric_cols)
        with col2:
            var2 = st.selectbox('Select second variable', 
                              [col for col in numeric_cols if col != var1])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=var1, y=var2, hue='Diabetes_012', alpha=0.6)
        plt.title(f'Relationship between {var1} and {var2}')
        st.pyplot(fig)
        
        # Correlation Analysis by Diabetes Status
        st.header("Correlation Analysis by Diabetes Status")
        
        diabetes_levels = df['Diabetes_012'].unique()
        tabs = st.tabs([f"Diabetes Level {level}" for level in diabetes_levels])
        
        for level, tab in zip(diabetes_levels, tabs):
            with tab:
                subset = df[df['Diabetes_012'] == level]
                corr_matrix = subset.corr()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title(f'Correlation Matrix for Diabetes Level {level}')
                st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.info("Please ensure the dataset is properly processed")

if __name__ == "__main__":
    main()