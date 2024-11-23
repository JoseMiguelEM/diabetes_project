import streamlit as st
from utils.session_state import initialize_session_state, get_data
from data_processing.dataset_processor import DatasetProcessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(
        page_title="Dataset Processing - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Verificar tipo de usuario
    if st.session_state.user_type != 'Data Analyst':
        st.warning("This page is only accessible to Data Analysts")
        return
    
    st.title("🔄 Dataset Processing and Optimization")
    
    st.write("""
    ### Dataset Processing Visualization
    This section shows the preprocessing steps applied to the dataset:
    1. Initial data analysis
    2. Data normalization
    3. Class balancing
    4. Final dataset results
    """)
    
    # Cargar ambos datasets
    df_original = get_data(use_processed=False)
    df_processed = get_data(use_processed=True)
    
    # Initial Data Analysis
    st.subheader("1. Initial Data Analysis")
    
    # Mostrar distribución original
    initial_dist = df_original['Diabetes_012'].value_counts()
    initial_percentages = df_original['Diabetes_012'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Original Class Distribution:")
        st.write(pd.DataFrame({
            'Class': initial_dist.index,
            'Count': initial_dist.values,
            'Percentage': [f"{v:.2f}%" for v in initial_percentages.values]
        }))
    
    with col2:
        fig, ax = plt.subplots()
        plt.pie(
            initial_dist.values,
            labels=initial_dist.index,
            autopct='%1.1f%%'
        )
        plt.title("Original Class Distribution")
        st.pyplot(fig)
    
    # Data Normalization
    st.subheader("2. Data Normalization")
    
    # Visualización de normalización
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("Select column to visualize normalization:", numeric_cols)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original data distribution
    sns.histplot(df_original[selected_col], ax=ax1)
    ax1.set_title("Original Distribution")
    
    # Normalized data distribution
    sns.histplot(df_processed[selected_col], ax=ax2)
    ax2.set_title("Normalized Distribution")
    
    st.pyplot(fig)
    
    # Class Balancing Results
    st.subheader("3. Class Balancing Results")
    
    # Mostrar distribución final
    final_dist = df_processed['Diabetes_012'].value_counts()
    final_percentages = df_processed['Diabetes_012'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Final class distribution:")
        st.write(pd.DataFrame({
            'Class': final_dist.index,
            'Count': final_dist.values,
            'Percentage': [f"{v:.2f}%" for v in final_percentages.values]
        }))
    
    with col2:
        fig, ax = plt.subplots()
        plt.pie(
            final_dist.values,
            labels=final_dist.index,
            autopct='%1.1f%%'
        )
        plt.title("Balanced Class Distribution")
        st.pyplot(fig)
    
    # Final Dataset Statistics
    st.subheader("4. Final Dataset Statistics")
    
    # Mostrar estadísticas del dataset final
    st.write("Statistical summary of the processed dataset:")
    st.write(df_processed.describe())
    
    # Comparación antes/después
    st.write("### Before vs After Processing Comparison")
    
    metrics_comparison = pd.DataFrame({
        'Metric': ['Total Samples', 'Class Balance Ratio', 'Features'],
        'Before': [
            len(df_original),
            f"1:{initial_dist[1]/initial_dist[0]:.2f}:{initial_dist[2]/initial_dist[0]:.2f}",
            len(df_original.columns)
        ],
        'After': [
            len(df_processed),
            f"1:{final_dist[1]/final_dist[0]:.2f}:{final_dist[2]/final_dist[0]:.2f}",
            len(df_processed.columns)
        ]
    })
    
    st.write(metrics_comparison)

if __name__ == "__main__":
    main()