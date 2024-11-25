import streamlit as st
import os
from components.eda.data_loader import load_dataset
from data_processing.dataset_processor import DatasetProcessor
from model_training.model_trainer import ModelTrainer
from utils.project_utils import get_project_root
import pandas as pd
from sklearn.model_selection import train_test_split

def process_and_train_if_needed():
    """Procesa el dataset y entrena el modelo si no existen"""
    project_root = get_project_root()
    processed_path = os.path.join(project_root, 'data', 'dataset-final.csv')
    model_path = os.path.join(project_root, 'models', 'diabetes_model.joblib')
    
    # Procesar dataset si no existe
    if not os.path.exists(processed_path):
        processor = DatasetProcessor()
        results = processor.process_dataset()
        if not results['success']:
            raise Exception("Error processing dataset")
    
    # Entrenar modelo si no existe
    if not os.path.exists(model_path):
        # Cargar dataset procesado
        df = pd.read_csv(processed_path)
        X = df.drop('Diabetes_012', axis=1)
        y = df['Diabetes_012']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar modelo
        trainer = ModelTrainer()
        trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

def initialize_session_state():
    """Inicializa el estado global de la aplicación"""
    if 'data_loaded' not in st.session_state:
        # Procesar dataset y entrenar modelo automáticamente si no existen
        process_and_train_if_needed()
        
        st.session_state.data_loaded = False
        st.session_state.df = None
        st.session_state.df_processed = None
    
    if 'user_type' not in st.session_state:
        st.session_state.user_type = 'General User'

def get_data(use_processed=True):
    """Obtiene el dataset apropiado según el contexto"""
    try:
        if use_processed and st.session_state.df_processed is not None:
            return st.session_state.df_processed
        elif not use_processed and st.session_state.df is not None:
            return st.session_state.df
            
        project_root = get_project_root()
        
        # Siempre procesar el dataset y entrenar si no existen
        process_and_train_if_needed()
        
        if use_processed:
            filepath = os.path.join(project_root, 'data', 'dataset-final.csv')
        else:
            filepath = os.path.join(project_root, 'data', 'dataset.csv')
        
        if not os.path.exists(filepath):
            st.error(f"Dataset file not found at: {filepath}")
            st.info("Please ensure the dataset is in the 'diabetes_project/data' folder")
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
        st.info("Check if the dataset is in the correct location: diabetes_project/data/dataset.csv")
        st.stop()