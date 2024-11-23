# src/components/model.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import ClusterCentroids, NearMiss, TomekLinks

class DiabetesModel:
    def __init__(self):
        self.model = None
        self.model_path = 'models/diabetes_model.pkl'
    
    def train(self, X, y, balance_technique=None):
        """
        Entrena el modelo de Random Forest con opción de balanceo
        """
        # División train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Aplicar técnica de balanceo si se especifica
        if balance_technique:
            if balance_technique == "ClusterCentroids":
                sampler = ClusterCentroids(random_state=42)
            elif balance_technique == "NearMiss":
                sampler = NearMiss(version=1)
            elif balance_technique == "TomekLinks":
                sampler = TomekLinks()
            
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        
        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Calcular score
        score = self.model.score(X_test, y_test)
        
        # Guardar el modelo
        self.save_model()
        
        return score
    
    def predict(self, features: pd.DataFrame):
        """
        Realiza predicción usando el modelo guardado
        """
        if self.model is None:
            self.load_model()
        return self.model.predict(features)
    
    def save_model(self):
        """
        Guarda el modelo en disco
        """
        joblib.dump(self.model, self.model_path)
    
    def load_model(self):
        """
        Carga el modelo desde disco
        """
        self.model = joblib.load(self.model_path)

    def get_feature_importance(self):
        """
        Obtiene la importancia de las características
        """
        if self.model is None:
            self.load_model()
        
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_
        ).sort_values(ascending=False)