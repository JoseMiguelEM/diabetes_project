import pandas as pd
import numpy as np

def analyze_numeric_variables(df, numeric_cols):
    """
    Analiza variables numéricas
    """
    numeric_stats = {}
    
    for col in numeric_cols:
        stats = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        
        # Detectar outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | 
                     (df[col] > (Q3 + 1.5 * IQR))][col]
        
        stats['outliers'] = {
            'count': len(outliers),
            'percentage': (len(outliers)/len(df))*100,
            'range': [outliers.min(), outliers.max()] if len(outliers) > 0 else None
        }
        
        numeric_stats[col] = stats
    
    return numeric_stats

def analyze_categorical_variables(df, cat_cols):
    """
    Analiza variables categóricas
    """
    categorical_stats = {}
    
    for col in cat_cols:
        stats = {
            'unique_values': df[col].nunique(),
            'value_counts': df[col].value_counts().to_dict(),
            'missing_values': df[col].isnull().sum()
        }
        categorical_stats[col] = stats
    
    return categorical_stats

def analyze_binary_variables(df, binary_cols):
    """
    Analiza variables binarias
    """
    binary_stats = {}
    
    for col in binary_cols:
        stats = {
            'value_counts': df[col].value_counts().to_dict(),
            'proportions': df[col].value_counts(normalize=True).to_dict(),
            'mode': df[col].mode()[0]
        }
        binary_stats[col] = stats
    
    return binary_stats