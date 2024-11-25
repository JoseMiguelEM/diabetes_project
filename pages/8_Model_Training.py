import streamlit as st
from utils.session_state import initialize_session_state, get_data
from model_training.model_trainer import ModelTrainer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        trainer = ModelTrainer()
        trainer.load_results()
        summary = trainer.get_training_summary()
        
        if isinstance(summary, dict):
            # Mostrar métricas del modelo
            st.header("Model Performance Metrics")
            
            # Mostrar métricas por clase
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sensitivity (Recall) and Specificity")
                sens_spec_df = pd.DataFrame({
                    'Sensitivity': summary['sensitivity_per_class'],
                    'Specificity': summary['specificity_per_class']
                })
                st.write(sens_spec_df)
                
                # Calcular y mostrar falsos negativos por clase
                st.subheader("False Negatives by Class")
                fn_df = pd.DataFrame({
                    'False Negative Rate': {
                        class_name: (1 - sens) * 100  # Convertir a porcentaje
                        for class_name, sens in summary['sensitivity_per_class'].items()
                    }
                })
                st.write("Percentage of cases missed by the model:")
                st.write(fn_df.style.format("{:.2f}%"))
            
            with col2:
                st.subheader("Precision and Recall")
                prec_recall_df = pd.DataFrame({
                    'Precision': summary['precision_per_class'],
                    'Recall': summary['recall_per_class']
                })
                st.write(prec_recall_df)
                
                # Calcular y mostrar F1-Score
                st.subheader("F1-Score by Class")
                f1_scores = {
                    class_name: 2 * (prec * rec) / (prec + rec)
                    for (class_name, prec), (_, rec) in zip(
                        summary['precision_per_class'].items(),
                        summary['recall_per_class'].items()
                    )
                }
                st.write(pd.DataFrame({'F1-Score': f1_scores}))
            
            # Matriz de Confusión
            st.header("Confusion Matrix")
            if 'confusion_matrix' in summary:
                fig, ax = plt.subplots(figsize=(10, 8))
                conf_matrix = summary['confusion_matrix']
                
                # Calcular porcentajes
                conf_matrix_percent = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
                
                sns.heatmap(conf_matrix_percent, 
                           annot=True, 
                           fmt='.1f', 
                           cmap='Blues',
                           xticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'],
                           yticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'])
                plt.title('Confusion Matrix (Percentages)')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
                
                # Explicación de la matriz
                st.write("""
                **Interpretation:**
                - Diagonal values show correct predictions (%)
                - Off-diagonal values show misclassifications (%)
                - Higher values in diagonal = better performance
                """)
            
            # Visualizaciones del modelo
            st.header("Model Visualizations")
            
            plot_paths = trainer.get_model_performance_plots()
            
            tab1, tab2 = st.tabs([
                "ROC Curves",
                "Feature Importance"
            ])
            
            with tab1:
                if plot_paths['roc_curves']:
                    st.image(plot_paths['roc_curves'])
                    st.write("""
                    ROC curves show model's ability to distinguish between classes:
                    - Higher AUC indicates better classification
                    - Curves closer to top-left corner = better performance
                    """)
            
            with tab2:
                if plot_paths['feature_importance']:
                    st.image(plot_paths['feature_importance'])
                    st.write("""
                    Feature importance indicates most influential factors:
                    - Longer bars = stronger influence on predictions
                    - Helps identify key diabetes risk factors
                    """)
            
            # Resumen general del rendimiento
            st.header("Overall Performance Summary")
            avg_metrics = {
                'Average Precision': np.mean(list(summary['precision_per_class'].values())),
                'Average Recall': np.mean(list(summary['recall_per_class'].values())),
                'Average F1-Score': np.mean(list(f1_scores.values())),
                'False Negative Rate': np.mean(fn_df['False Negative Rate'])
            }
            
            st.write(pd.DataFrame({
                'Metric': list(avg_metrics.keys()),
                'Value': [f"{v:.2f}%" for v in [x * 100 for x in avg_metrics.values()]]
            }))
            
        else:
            st.error("Model results not found. Please contact system administrator.")
        
    except Exception as e:
        st.error(f"Error loading model analysis: {str(e)}")
        st.info("Please ensure the model has been properly trained")

if __name__ == "__main__":
    main()