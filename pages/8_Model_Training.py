import streamlit as st
from utils.session_state import initialize_session_state
import os
from utils.project_utils import get_images_root
def main():
    st.set_page_config(
        page_title="Model Analysis - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    if st.session_state.user_type != 'Data Analyst':
        st.warning("This page is only accessible to Data Analysts")
        return
    
    st.title("🤖 Model Analysis and Performance")
    
    try:
        figures_dir = get_images_root()
        
        # ROC Curves Analysis
        st.header("ROC Curves Analysis")
        roc_path = os.path.join(figures_dir, 'roc_curves.png')
        if os.path.exists(roc_path):
            with open(roc_path, "rb") as file:
                st.image(file.read())
                st.markdown("""
                **Understanding the ROC Curves:**
                - Area Under Curve (AUC) measures model's ability to distinguish between classes
                - No Diabetes (AUC = 0.79): Good discrimination ability
                - Prediabetes (AUC = 0.61): Moderate discrimination
                - Diabetes (AUC = 0.73): Good discrimination ability
                - Curves above diagonal line indicate better-than-random prediction
                """)
        
        # Feature Importance Analysis
        st.header("Feature Importance Analysis")
        importance_path = os.path.join(figures_dir, 'feature_importance.png')
        if os.path.exists(importance_path):
            with open(importance_path, "rb") as file:
                st.image(file.read())
                st.markdown("""
                **Key Feature Insights:**
                - BMI, General Health, and Age are the top predictors
                - High Blood Pressure shows significant importance
                - Lifestyle factors (Physical Activity, Diet) have moderate impact
                - Healthcare access indicators show lower importance
                """)
        
        # Confusion Matrix Analysis
        st.header("Confusion Matrix Analysis")
        matrix_path = os.path.join(figures_dir, 'confusion_matrix.png')
        if os.path.exists(matrix_path):
            with open(matrix_path, "rb") as file:
                st.image(file.read())
                st.markdown("""
                **Matrix Interpretation:**
                - Darker blue indicates higher number of predictions
                - Diagonal elements represent correct predictions
                - Notable observations:
                  * Strong performance in identifying No Diabetes cases (751 correct)
                  * Good accuracy for Diabetes cases (700 correct)
                  * Moderate performance for Prediabetes (151 correct)
                  * Some confusion between Prediabetes and Diabetes classes
                """)
        
    except Exception as e:
        st.error(f"Error loading model analysis: {str(e)}")
        st.info("Please ensure the model has been properly trained and images are present in the figures directory")

if __name__ == "__main__":
    main()