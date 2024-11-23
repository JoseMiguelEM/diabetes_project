import pandas as pd
import os
import joblib
from components.model import DiabetesModel
from utils.project_utils import get_project_root
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self):
        self.project_root = get_project_root()
        self.model = DiabetesModel()
        self.training_metrics = None
        self.feature_importance = None
        self.models_dir = os.path.join(self.project_root, 'models')
        
        # Crear directorio de modelos si no existe
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Configurar rutas para métricas y visualizaciones
        self.metrics_path = os.path.join(self.models_dir, 'training_metrics.json')
        self.figures_dir = os.path.join(self.models_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Entrena el modelo y guarda las métricas y visualizaciones
        """
        print("Iniciando entrenamiento del modelo...")
        
        # Entrenar modelo
        metrics = self.model.train(X_train, y_train)
        
        # Evaluar en conjunto de prueba
        y_pred = self.model.predict(X_test)
        test_metrics = self.model.evaluate_model(X_test, y_test)
        
        # Guardar métricas
        self.training_metrics = {
            'training_metrics': metrics,
            'test_metrics': test_metrics,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Generar y guardar visualizaciones
        self._generate_and_save_plots(X_test, y_test, y_pred)
        
        # Guardar modelo y métricas
        self.save_results()
        
        return self.training_metrics
    
    def _generate_and_save_plots(self, X_test, y_test, y_pred):
        """
        Genera y guarda todas las visualizaciones relevantes
        """
        # Matriz de confusión
        cm_fig = self.model.plot_confusion_matrix(y_test, y_pred)
        cm_fig.savefig(os.path.join(self.figures_dir, 'confusion_matrix.png'))
        plt.close(cm_fig)
        
        # Curvas ROC
        roc_fig = self.model.plot_roc_curves(X_test, y_test)
        roc_fig.savefig(os.path.join(self.figures_dir, 'roc_curves.png'))
        plt.close(roc_fig)
        
        # Importancia de características
        imp_fig = self.model.plot_feature_importance()
        imp_fig.savefig(os.path.join(self.figures_dir, 'feature_importance.png'))
        plt.close(imp_fig)
        
        # SHAP summary plot
        shap_fig = self.model.plot_shap_summary(X_test)
        shap_fig.savefig(os.path.join(self.figures_dir, 'shap_summary.png'))
        plt.close(shap_fig)
    
    def save_results(self):
        """
        Guarda el modelo y las métricas
        """
        # Guardar modelo
        self.model.save_model()
        
        # Guardar métricas
        import json
        with open(self.metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=4)
    
    def load_results(self):
        """
        Carga métricas y modelo guardados
        """
        self.model.load_model()
        
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                self.training_metrics = json.load(f)
    
    def get_training_summary(self):
        """
        Retorna un resumen del entrenamiento
        """
        if self.training_metrics is None:
            self.load_results()
        
        if self.training_metrics is None:
            return "No hay métricas de entrenamiento disponibles"
        
        test_metrics = self.training_metrics['test_metrics']
        class_names = ['No Diabetes', 'Prediabetes', 'Diabetes']
        
        summary = {
            'sensitivity_per_class': {
                class_names[i]: test_metrics['sensitivity'][i]
                for i in range(3)
            },
            'specificity_per_class': {
                class_names[i]: test_metrics['specificity'][i]
                for i in range(3)
            },
            'precision_per_class': {
                class_names[i]: test_metrics['precision'][i]
                for i in range(3)
            },
            'recall_per_class': {
                class_names[i]: test_metrics['recall'][i]
                for i in range(3)
            }
        }
        
        return summary
    
    def get_feature_rankings(self):
        """
        Retorna el ranking de importancia de características
        """
        if self.model.model is None:
            self.model.load_model()
        
        return self.model.get_feature_importance()
    
    def get_model_performance_plots(self):
        """
        Retorna las rutas a los gráficos guardados
        """
        plot_paths = {
            'confusion_matrix': os.path.join(self.figures_dir, 'confusion_matrix.png'),
            'roc_curves': os.path.join(self.figures_dir, 'roc_curves.png'),
            'feature_importance': os.path.join(self.figures_dir, 'feature_importance.png'),
            'shap_summary': os.path.join(self.figures_dir, 'shap_summary.png')
        }
        
        # Verificar que todos los archivos existen
        for name, path in plot_paths.items():
            if not os.path.exists(path):
                plot_paths[name] = None
        
        return plot_paths