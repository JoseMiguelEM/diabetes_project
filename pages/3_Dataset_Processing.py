# src/pages/dataset_processing.py
import streamlit as st
from data_processing.dataset_processor import DatasetProcessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_dataset_processing_page():
    st.title("🔄 Dataset Processing and Optimization")
    
    st.write("""
    ### Dataset Optimization Process
    This section allows you to process and optimize the dataset following these steps:
    1. Initial data analysis
    2. Data normalization
    3. Class balancing
    4. Final dataset generation
    """)
    
    processor = DatasetProcessor()
    
    # Initial Data Analysis
    st.subheader("1. Initial Data Analysis")
    if st.button("Load and Analyze Original Dataset"):
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
            plt.title("Original Class Distribution")
            st.pyplot(fig)
    
    # Data Normalization
    st.subheader("2. Data Normalization")
    if st.button("Run Data Normalization"):
        df_normalized = processor.normalize_data()
        
        st.write("Sample of normalized data:")
        st.write(df_normalized.head())
        
        # Visualization of normalization
        numeric_cols = df_normalized.select_dtypes(include=['float64', 'int64']).columns
        selected_col = st.selectbox("Select column to visualize:", numeric_cols)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Original data distribution
        sns.histplot(processor.df[selected_col], ax=ax1)
        ax1.set_title("Original Distribution")
        
        # Normalized data distribution
        sns.histplot(df_normalized[selected_col], ax=ax2)
        ax2.set_title("Normalized Distribution")
        
        st.pyplot(fig)
    
    # Class Balancing
    st.subheader("3. Class Balancing")
    if st.button("Run Class Balancing"):
        balanced_dist = processor.balance_classes()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Final class distribution:")
            st.write(pd.DataFrame({
                'Class': balanced_dist['distribution'].keys(),
                'Count': balanced_dist['distribution'].values(),
                'Percentage': [f"{v:.2f}%" for v in balanced_dist['percentages'].values()]
            }))
        
        with col2:
            fig, ax = plt.subplots()
            plt.pie(
                balanced_dist['distribution'].values(),
                labels=balanced_dist['distribution'].keys(),
                autopct='%1.1f%%'
            )
            plt.title("Balanced Class Distribution")
            st.pyplot(fig)
    
    # Final Dataset Generation
    st.subheader("4. Final Dataset Generation")
    if st.button("Generate Final Dataset"):
        if processor.save_final_dataset():
            st.success("Final dataset generated successfully!")
            
            # Show preview of final dataset
            df_final = pd.read_csv('data/dataset-final.csv')
            st.write("Preview of the final dataset:")
            st.write(df_final.head())
            
            # Show dataset statistics
            st.write("### Final Dataset Statistics:")
            st.write(df_final.describe())
        else:
            st.error("Error generating final dataset. Please run previous steps first.")
            st.info("Make sure to run steps 1-3 before generating the final dataset.")