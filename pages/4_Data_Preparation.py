import streamlit as st
from utils.session_state import initialize_session_state, get_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(
        page_title="Data Preparation - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Verificar tipo de usuario
    if st.session_state.user_type != 'Data Analyst':
        st.warning("This page is only accessible to Data Analysts")
        return
    
    st.title('🔄 Data Preparation')
    
    try:
        df = get_data(use_processed=False)
        df_processed = get_data(use_processed=True)
        
        # Original vs Processed Data
        st.header("Data Overview")
        tabs = st.tabs(["Original Data", "Processed Data"])
        
        with tabs[0]:
            st.subheader("Original Dataset")
            st.write(df.head())
            st.write("Shape:", df.shape)
            
            # Basic statistics
            st.subheader("Statistical Summary")
            st.write(df.describe())
            
            # Missing values
            st.subheader("Missing Values")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.write(missing[missing > 0])
            else:
                st.write("No missing values found")
        
        with tabs[1]:
            st.subheader("Processed Dataset")
            st.write(df_processed.head())
            st.write("Shape:", df_processed.shape)
            
            # Basic statistics
            st.subheader("Statistical Summary")
            st.write(df_processed.describe())
            
            # Compare distributions
            st.subheader("Distribution Comparison")
            numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
            selected_col = st.selectbox("Select column:", numeric_cols)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Original distribution
            sns.histplot(data=df, x=selected_col, ax=ax1)
            ax1.set_title(f"Original {selected_col} Distribution")
            
            # Processed distribution
            sns.histplot(data=df_processed, x=selected_col, ax=ax2)
            ax2.set_title(f"Processed {selected_col} Distribution")
            
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please process the dataset first in the Dataset Processing page")

if __name__ == "__main__":
    main()