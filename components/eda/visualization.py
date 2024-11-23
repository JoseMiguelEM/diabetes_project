import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_numeric_distribution(df, column):
    """
    Grafica la distribución de una variable numérica
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histograma con KDE
    sns.histplot(data=df, x=column, kde=True, ax=ax1)
    ax1.set_title(f'Distribución de {column}')
    
    # Boxplot
    sns.boxplot(data=df, y=column, ax=ax2)
    ax2.set_title(f'Boxplot de {column}')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(correlation_matrix):
    """
    Grafica la matriz de correlación
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    return plt.gcf()

def plot_class_distribution(y):
    """
    Grafica la distribución de clases
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Distribución de Clases')
    plt.tight_layout()
    return plt.gcf()

def plot_balancing_results(results):
    """
    Grafica los resultados de las técnicas de balanceo
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribución de Clases después del Balanceo', fontsize=16)
    
    for idx, (method, counter) in enumerate(results.items()):
        row = idx // 2
        col = idx % 2
        
        pd.Series(counter).plot(kind='bar', ax=axes[row, col])
        axes[row, col].set_title(f'Método: {method}')
        axes[row, col].set_xlabel('Clase')
        axes[row, col].set_ylabel('Cantidad')
    
    plt.tight_layout()
    return fig