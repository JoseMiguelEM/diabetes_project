import streamlit as st
from utils.session_state import initialize_session_state, get_data
from model_training.model_trainer import ModelTrainer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(
        page_title="Model Analysis - Diabetes Analysis",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Verificar tipo de usuario
    if st.session_state.user_type != 'Data Analyst':
        st.warning("This page is only accessible to Data Analysts")
        return
    
    st.title("🤖 Model Analysis and Performance")
    
    try:
        # Inicializar trainer y cargar resultados
        trainer = ModelTrainer()
        trainer.load_results()
        summary = trainer.get_training_summary()
        
        if isinstance(summary, dict):
            # Mostrar métricas del modelo
            st.header("Model Performance Metrics")
            
            # Mostrar métricas por clase
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sensitivity and Specificity")
                sens_spec_df = pd.DataFrame({
                    'Sensitivity': summary['sensitivity_per_class'],
                    'Specificity': summary['specificity_per_class']
                })
                st.write(sens_spec_df)
            
            with col2:
                st.subheader("Precision and Recall")
                prec_recall_df = pd.DataFrame({
                    'Precision': summary['precision_per_class'],
                    'Recall': summary['recall_per_class']
                })
                st.write(prec_recall_df)
            
            # Mostrar visualizaciones
            st.header("Model Insights")
            
            plot_paths = trainer.get_model_performance_plots()
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Confusion Matrix",
                "ROC Curves",
                "Feature Importance",
                "SHAP Analysis"
            ])
            
            with tab1:
                if plot_paths['confusion_matrix']:
                    st.image(plot_paths['confusion_matrix'])
                    st.write("""
                    The confusion matrix shows the model's performance across all classes.
                    - Diagonal values represent correct predictions
                    - Off-diagonal values represent misclassifications
                    """)
            
            with tab2:
                if plot_paths['roc_curves']:
                    st.image(plot_paths['roc_curves'])
                    st.write("""
                    ROC curves show the tradeoff between sensitivity and specificity.
                    - Higher AUC indicates better model performance
                    - Curves closer to the top-left corner indicate better classification
                    """)
            
            with tab3:
                if plot_paths['feature_importance']:
                    st.image(plot_paths['feature_importance'])
                    st.write("""
                    Feature importance shows which factors most influence the model's predictions.
                    - Longer bars indicate more important features
                    - This helps identify key risk factors for diabetes
                    """)
            
            with tab4:
                if plot_paths['shap_summary']:
                    st.image(plot_paths['shap_summary'])
                    st.write("""
                    SHAP values explain how each feature contributes to predictions.
                    - Color indicates feature value (red = high, blue = low)
                    - Width shows the magnitude of the feature's impact
                    """)
            
            # Conclusiones y recomendaciones
            st.header("Model Insights and Recommendations")
            st.write("""
            ### Key Findings:
            1. The model shows balanced performance across all diabetes classes
            2. Special attention is given to minimizing false negatives in diabetes detection
            3. Feature importance analysis reveals key risk factors
            
            ### Model Usage Guidelines:
            - The model is optimized for early detection of diabetes risk
            - Predictions should be used as a screening tool, not final diagnosis
            - Regular model monitoring and updates are recommended
            """)
            
        else:
            st.error("Model results not found. Please contact system administrator.")
        
    except Exception as e:
        st.error(f"Error loading model analysis: {str(e)}")
        st.info("Please ensure the model has been properly trained")

if __name__ == "__main__":
    main()