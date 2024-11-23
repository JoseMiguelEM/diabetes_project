import pandas as pd
import numpy as np

def get_correlation_matrix(df):
    """
    Calcula la matriz de correlación
    """
    return df.corr()

def get_feature_correlations_with_target(df, target_col, threshold=0.1):
    """
    Obtiene correlaciones con la variable objetivo
    """
    correlations = df.corr()[target_col].sort_values(ascending=False)
    
    # Filtrar correlaciones significativas
    significant_corr = correlations[abs(correlations) > threshold]
    
    return significant_corr

def get_high_correlation_pairs(df, threshold=0.5):
    """
    Encuentra pares de variables con alta correlación
    """
    corr_matrix = df.corr()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    return pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)