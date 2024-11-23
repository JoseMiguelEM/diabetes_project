import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import os
from utils.project_utils import get_project_root

class DatasetProcessor:
    def __init__(self):
        project_root = get_project_root()
        self.input_path = os.path.join(project_root, 'data', 'dataset.csv')
        self.output_path = os.path.join(project_root, 'data', 'dataset-final.csv')
        self.df = None
        self.df_normalized = None
        self.df_balanced = None
        self.scalers = {}  # Para almacenar los scalers y poder invertir la normalización si es necesario
        
    def load_data(self):
        """Carga el dataset original"""
        self.df = pd.read_csv(self.input_path)
        return self.analyze_initial_distribution()
    
    def analyze_initial_distribution(self):
        """Analiza la distribución inicial de las clases"""
        class_dist = self.df['Diabetes_012'].value_counts()
        percentages = class_dist / len(self.df) * 100
        return {
            'distribution': class_dist.to_dict(),
            'percentages': percentages.to_dict()
        }
    
    def check_distribution_type(self, column):
        """Determina si una columna tiene distribución estándar"""
        from scipy import stats
        
        # Realizar prueba de normalidad
        statistic, p_value = stats.normaltest(self.df[column])
        
        # Si p_value > 0.05, podemos considerar que sigue una distribución normal
        return p_value > 0.05
    
    def normalize_data(self):
        """Aplica normalización según el tipo de distribución"""
        self.df_normalized = self.df.copy()
        
        # Columnas numéricas que necesitan normalización
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        for column in numeric_columns:
            if column != 'Diabetes_012':  # No normalizar la variable objetivo
                if self.check_distribution_type(column):
                    # Usar StandardScaler para distribución normal
                    scaler = StandardScaler()
                else:
                    # Usar RobustScaler para distribución no normal
                    scaler = RobustScaler()
                
                # Ajustar y transformar los datos
                data_reshaped = self.df_normalized[column].values.reshape(-1, 1)
                self.df_normalized[column] = scaler.fit_transform(data_reshaped)
                
                # Guardar el scaler para uso futuro
                self.scalers[column] = scaler
        
        return self.df_normalized
    
    def balance_classes(self):
        """Realiza el balanceo de clases usando undersampling"""
        X = self.df_normalized.drop('Diabetes_012', axis=1)
        y = self.df_normalized['Diabetes_012']
        
        # Obtener conteos actuales
        current_counts = pd.Series(y).value_counts()
        
        # Encontrar la clase minoritaria
        min_class_count = current_counts.min()
        
        # Calcular las cantidades objetivo basadas en la clase minoritaria
        target_counts = {
            0: int(min_class_count * 1.33),  # 40% del total
            1: min_class_count,              # 30% del total
            2: min_class_count               # 30% del total
        }
        
        # Asegurarse de que no excedemos los conteos originales
        for class_label in target_counts:
            if target_counts[class_label] > current_counts[class_label]:
                target_counts[class_label] = current_counts[class_label]
        
        # Aplicar undersampling
        undersampler = RandomUnderSampler(
            sampling_strategy=target_counts,
            random_state=42
        )
        
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
        
        # Crear el DataFrame balanceado
        self.df_balanced = pd.concat([
            pd.DataFrame(X_balanced, columns=X.columns),
            pd.Series(y_balanced, name='Diabetes_012')
        ], axis=1)
        
        return self.analyze_balanced_distribution()
    
    def analyze_balanced_distribution(self):
        """Analiza la distribución después del balanceo"""
        class_dist = self.df_balanced['Diabetes_012'].value_counts()
        percentages = class_dist / len(self.df_balanced) * 100
        return {
            'distribution': class_dist.to_dict(),
            'percentages': percentages.to_dict()
        }
    
    def save_final_dataset(self):
        """Guarda el dataset procesado"""
        if self.df_balanced is not None:
            # Asegurarse de que el directorio existe
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            # Guardar el dataset
            self.df_balanced.to_csv(self.output_path, index=False)
            # Guardar los scalers para uso futuro
            import joblib
            scalers_path = os.path.join(os.path.dirname(self.output_path), 'scalers.joblib')
            joblib.dump(self.scalers, scalers_path)
            return True
        return False
    
    def process_dataset(self):
        """Ejecuta todo el proceso de preparación del dataset"""
        initial_dist = self.load_data()
        self.normalize_data()
        balanced_dist = self.balance_classes()
        success = self.save_final_dataset()
        
        return {
            'initial_distribution': initial_dist,
            'final_distribution': balanced_dist,
            'success': success
        }

if __name__ == "__main__":
    processor = DatasetProcessor()
    results = processor.process_dataset()
    print("Distribución inicial:", results['initial_distribution'])
    print("Distribución final:", results['final_distribution'])
    print("Dataset procesado y guardado:", results['success'])