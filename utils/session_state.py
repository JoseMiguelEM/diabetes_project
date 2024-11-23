# src/utils/session_state.py
import streamlit as st
import os
from components.eda.data_loader import load_dataset

def initialize_session_state():
    """Inicializa el estado global de la aplicación"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.df_processed = None

def get_data(use_processed=True):
    """Obtiene el dataset apropiado según el contexto"""
    try:
        if use_processed and st.session_state.df_processed is not None:
            return st.session_state.df_processed
        elif not use_processed and st.session_state.df is not None:
            return st.session_state.df
            
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(current_dir)
        
        if use_processed:
            filepath = os.path.join(project_root, 'data', 'dataset-final.csv')
            if not os.path.exists(filepath):
                return get_data(use_processed=False)
        else:
            filepath = os.path.join(project_root, 'data', 'dataset.csv')
        
        if not os.path.exists(filepath):
            st.error(f"Dataset file not found in 'data' folder")
            st.stop()
            
        df = load_dataset(filepath)
        
        if use_processed:
            st.session_state.df_processed = df
        else:
            st.session_state.df = df
            
        st.session_state.data_loaded = True
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()