import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import os

class DiabetesModel:
    def __init__(self):
        self.model = None
        project_root = r"C:\Proyecto vscode\diabetes_project"
        self.models_dir = os.path.join(project_root, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, 'diabetes_model.joblib')
        self.metrics_path = os.path.join(self.models_dir, 'model_metrics.joblib')
        self.feature_names = None
    
    
    def train(self, X, y):
        """
        Entrena el modelo de Random Forest optimizado para diabetes
        """
        # División train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Configurar modelo con pesos ajustados para priorizar falsos positivos
        class_weights = {
            0: 1.0,    # No diabetes
            1: 1.5,    # Prediabetes (penaliza más los falsos negativos)
            2: 2.0     # Diabetes (penaliza aún más los falsos negativos)
        }
        
        # Configurar Random Forest con parámetros optimizados
        self.model = RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Guardar nombres de características
        self.feature_names = X.columns.tolist()
        
        # Entrenar modelo
        self.model.fit(X_train, y_train)
        
        # Calcular y guardar métricas
        metrics = self.evaluate_model(X_test, y_test)
        self.save_metrics(metrics)
        
        # Guardar el modelo entrenado
        self.save_model()
        
        return metrics
    
    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo con múltiples métricas
        """
        # Predicciones
        y_pred = self.model.predict(X_test)
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Métricas por clase
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
        
        # Calcular sensibilidad y especificidad para cada clase
        sensitivities = []
        specificities = []
        
        for class_idx in range(3):
            # Convertir a problema binario para cada clase
            y_true_bin = (y_test == class_idx).astype(int)
            y_pred_bin = (y_pred == class_idx).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
            
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
        
        return {
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivities,
            'specificity': specificities
        }
    
    def predict(self, features: pd.DataFrame):
        """
        Realiza predicción usando el modelo guardado
        """
        if self.model is None:
            self.load_model()
        return self.model.predict(features)
    
    def predict_proba(self, features: pd.DataFrame):
        """
        Retorna probabilidades de predicción
        """
        if self.model is None:
            self.load_model()
        return self.model.predict_proba(features)
    
    def save_model(self):
        """
        Guarda el modelo en disco
        """
        if self.model is not None:
            # Guardar modelo y nombres de características
            joblib.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, self.model_path)
    
    def save_metrics(self, metrics):
        """
        Guarda las métricas del modelo
        """
        joblib.dump(metrics, self.metrics_path)
    
    def load_model(self):
        """
        Carga el modelo desde disco
        """
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.feature_names = data['feature_names']
        else:
            raise FileNotFoundError("Model file not found. Please train the model first.")
    
    def get_feature_importance(self):
        """
        Obtiene la importancia de las características
        """
        if self.model is None:
            self.load_model()
        
        if self.feature_names is None:
            raise ValueError("Feature names not found")
        
        feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=True)
        
        return feature_importance
    
    def plot_feature_importance(self):
        """
        Grafica la importancia de características
        """
        importance = self.get_feature_importance()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.plot(kind='barh')
        plt.title('Feature Importance in Diabetes Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        return fig
    
    def plot_roc_curves(self, X_test, y_test):
        """
        Genera curvas ROC para cada clase
        """
        n_classes = 3
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        y_score = self.predict_proba(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['blue', 'red', 'green']
        classes = ['No Diabetes', 'Prediabetes', 'Diabetes']
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves per Class')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        return fig

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Genera una matriz de confusión visual
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Añadir etiquetas de clase
        ax.set_xticklabels(['No Diabetes', 'Prediabetes', 'Diabetes'])
        ax.set_yticklabels(['No Diabetes', 'Prediabetes', 'Diabetes'])
        
        plt.tight_layout()
        return fig