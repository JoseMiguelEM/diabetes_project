import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt

from src.components.eda import (
    data_loader,
    univariate,
    correlation,
    balance,
    visualization
)

def main():
    # 1. Cargar y verificar datos
    print("1. Cargando y verificando datos...")
    df = data_loader.load_dataset()
    info, stats = data_loader.check_data_quality(df)
    
    print(f"Dimensiones del dataset: {info['total_rows']} filas, {info['total_columns']} columnas")
    print("\nValores faltantes por columna:")
    for col, missing in info['missing_values'].items():
        if missing > 0:
            print(f"{col}: {missing}")
    
    # 2. Clasificar variables
    feature_types = data_loader.get_feature_types(df)
    print("\n2. Tipos de variables identificadas:")
    for type_name, features in feature_types.items():
        print(f"\n{type_name.capitalize()}:")
        print(", ".join(features))
    
    # 3. Análisis univariante
    print("\n3. Realizando análisis univariante...")
    
    # Análisis numérico
    numeric_stats = univariate.analyze_numeric_variables(df, feature_types['numeric'])
    print("\nEstadísticas de variables numéricas:")
    for var, stats in numeric_stats.items():
        print(f"\n{var}:")
        print(f"  Media: {stats['mean']:.2f}")
        print(f"  Mediana: {stats['median']:.2f}")
        print(f"  Desviación estándar: {stats['std']:.2f}")
        print(f"  Outliers: {stats['outliers']['count']} ({stats['outliers']['percentage']:.2f}%)")
        
        # Graficar distribución
        fig = visualization.plot_numeric_distribution(df, var)
        fig.savefig(f'outputs/distribution_{var}.png')
        plt.close(fig)
    
    # Análisis categórico
    categorical_stats = univariate.analyze_categorical_variables(df, feature_types['categorical'])
    print("\nEstadísticas de variables categóricas:")
    for var, stats in categorical_stats.items():
        print(f"\n{var}:")
        print(f"  Valores únicos: {stats['unique_values']}")
        print("  Distribución:")
        for val, count in stats['value_counts'].items():
            print(f"    {val}: {count}")
    
    # 4. Análisis de correlaciones
    print("\n4. Analizando correlaciones...")
    corr_matrix = correlation.get_correlation_matrix(df)
    target_correlations = correlation.get_feature_correlations_with_target(df, 'Diabetes_012')
    high_corr_pairs = correlation.get_high_correlation_pairs(df)
    
    # Graficar matriz de correlación
    corr_fig = visualization.plot_correlation_matrix(corr_matrix)
    corr_fig.savefig('outputs/correlation_matrix.png')
    plt.close(corr_fig)
    
    print("\nCorrelaciones más importantes con Diabetes_012:")
    print(target_correlations)
    
    print("\nPares de variables altamente correlacionadas:")
    print(high_corr_pairs)
    
    # 5. Análisis de balance
    print("\n5. Analizando balance de clases...")
    class_balance = balance.analyze_class_balance(df, 'Diabetes_012')
    
    print("\nDistribución de clases:")
    print(class_balance['counts'])
    print("\nProporciones:")
    print(class_balance['proportions'])
    
    # Aplicar técnicas de balanceo
    X = df.drop('Diabetes_012', axis=1)
    y = df['Diabetes_012']
    
    balancing_results = balance.apply_balancing_techniques(X, y)
    
    print("\nResultados de técnicas de balanceo:")
    for method, result in balancing_results.items():
        print(f"\n{method}:")
        print(result)
    
    # Graficar resultados de balanceo
    balance_fig = visualization.plot_balancing_results(balancing_results)
    balance_fig.savefig('outputs/balance_results.png')
    plt.close(balance_fig)

if __name__ == "__main__":
    # Crear directorio de outputs si no existe
    os.makedirs('outputs', exist_ok=True)
    
    # Configurar estilo de las gráficas
    plt.style.use('seaborn')
    
    # Ejecutar análisis
    main()