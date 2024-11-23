# src/components/model.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class DiabetesModel:
    def __init__(self):
        self.model = None
        self.model_path = 'models/diabetes_model.pkl'
    
    def train(self, X, y):
        """
        Entrena el modelo de Random Forest y retorna el score
        """
        # División train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo
        self.model = RandomForestClassifier(random_state=42)
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
