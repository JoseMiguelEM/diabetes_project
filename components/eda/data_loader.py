# src/components/eda/data_loader.py
import pandas as pd
import numpy as np
import os

def load_dataset(filepath=None):
    """
    Carga el dataset y realiza la preparación inicial
    """
    if filepath is None:
        # Construir ruta relativa desde la ubicación del script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        filepath = os.path.join(project_root, 'data', 'dataset.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo en: {filepath}")
        
    df = pd.read_csv(filepath)
    return df

def check_data_quality(df):
    """
    Verifica la calidad de los datos
    """
    # Información básica
    info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    
    # Estadísticas básicas
    stats = df.describe().to_dict()
    
    return info, stats

def get_feature_types(df):
    """
    Clasifica las variables por tipo
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Separar binarias de numéricas
    binary_features = [col for col in numeric_features 
                      if df[col].nunique() == 2]
    
    # Remover binarias de numéricas
    numeric_features = [col for col in numeric_features 
                       if col not in binary_features]
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'binary': binary_features
    }