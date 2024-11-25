import streamlit as st
from utils.session_state import initialize_session_state, get_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def main():
    st.set_page_config(page_title="Diabetes Statistics", layout="wide")
    initialize_session_state()
    
    st.title("📊 Diabetes Statistics")
    st.markdown("""
        This interactive dashboard explores health indicators related to diabetes,
        providing descriptive statistics, graphs, and analysis.
    """)
    
    try:
        df = get_data(use_processed=False)
        
        st.sidebar.title("Analysis Options")
        option = st.sidebar.radio(
            "Select functionality:",
            ["Interactive Graphs", "Health Indicators", "Age Distribution"]
        )

        if option == "Interactive Graphs":
            show_interactive_graphs(df)
        elif option == "Health Indicators":
            show_health_indicators(df)
        elif option == "Age Distribution":
            show_age_distribution(df)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_interactive_graphs(df):
    st.subheader("📉 Interactive Graphs")
    
    feature_names = {
        'HighBP': 'High Blood Pressure',
        'HighChol': 'High Cholesterol',
        'BMI': 'Body Mass Index',
        'Smoker': 'Smoking Status',
        'Stroke': 'Stroke History',
        'HeartDiseaseorAttack': 'Heart Disease/Attack',
        'PhysActivity': 'Physical Activity',
        'Fruits': 'Fruit Consumption',
        'Veggies': 'Vegetable Consumption',
        'GenHlth': 'General Health',
        'Age': 'Age Group'
    }
    
    selected_feature = st.selectbox(
        "Select a health indicator:", 
        list(feature_names.keys()),
        format_func=lambda x: feature_names[x]
    )

    col1, col2 = st.columns(2)
    fig_height = 6

    with col1:
        st.subheader(f"Distribution of {feature_names[selected_feature]}:")
        fig1, ax1 = plt.subplots(figsize=(6, fig_height))
        feature_counts = df[selected_feature].value_counts(normalize=True) * 100
        
        if selected_feature == 'GenHlth':
            labels = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
        else:
            labels = ['No', 'Yes'] if len(feature_counts) == 2 else feature_counts.index
            
        plt.pie(
            feature_counts,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=[0.1 if i == 0 else 0 for i in range(len(feature_counts))]
        )
        st.pyplot(fig1, use_container_width=True)
        
    with col2:
        st.subheader(f"Distribution of {feature_names[selected_feature]}:")
        fig2, ax2 = plt.subplots(figsize=(6, fig_height))
        feature_counts = df[selected_feature].value_counts()
        plt.bar(
            range(len(feature_counts)),
            feature_counts,
            color="skyblue",
            edgecolor="black"
        )
        
        if selected_feature == 'GenHlth':
            plt.xticks(range(5), ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
        else:
            plt.xticks(range(len(feature_counts)), ['No', 'Yes'] if len(feature_counts) == 2 else feature_counts.index)
        
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # Add Circular Bar Chart for Diabetes Distribution
    st.subheader(f"Diabetes Distribution by {feature_names[selected_feature]}:")
    diabetes_dist = pd.crosstab(df[selected_feature], df['Diabetes_012'], normalize='index') * 100

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    # Set the angles for the bars (360 degrees divided by number of categories)
    categories = diabetes_dist.index
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)

    # Plot bars for each diabetes status
    width = 0.5
    ax3.bar(angles, diabetes_dist[0], width=width, label='No Diabetes', 
            bottom=0, alpha=0.5, color='green')
    ax3.bar(angles, diabetes_dist[1], width=width, label='Prediabetes',
            bottom=diabetes_dist[0], alpha=0.5, color='yellow')
    ax3.bar(angles, diabetes_dist[2], width=width, label='Diabetes',
            bottom=diabetes_dist[0] + diabetes_dist[1], alpha=0.5, color='red')

    # Set the labels
    if selected_feature == 'GenHlth':
        ax3.set_xticks(angles)
        ax3.set_xticklabels(['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
    else:
        ax3.set_xticks(angles)
        ax3.set_xticklabels(['No', 'Yes'] if len(categories) == 2 else categories)

    ax3.set_ylim(0, 100)
    ax3.set_ylabel('Percentage')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    # Add explanation text
    st.subheader("""
    **Chart Interpretation:**""")
    st.write("""
    - Green: Percentage of people without diabetes
    - Yellow: Percentage of people with prediabetes
    - Red: Percentage of people with diabetes
    
    The height of each colored section represents the percentage of people in that diabetes category for each value of the selected health indicator.
    """)
def show_health_indicators(df):
    st.subheader("🏥 Health Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # BMI Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='BMI', bins=30)
        plt.title("BMI Distribution")
        st.pyplot(fig)
        
        # Physical Activity vs Diabetes
        fig, ax = plt.subplots(figsize=(10, 6))
        df_phys = df.groupby(['PhysActivity', 'Diabetes_012']).size().unstack()
        df_phys.plot(kind='bar', stacked=True)
        plt.title("Physical Activity vs Diabetes")
        plt.xlabel("Physical Activity")
        plt.ylabel("Count")
        plt.legend(['No Diabetes', 'Prediabetes', 'Diabetes'])
        st.pyplot(fig)
    
    with col2:
        # Blood Pressure vs Diabetes
        fig, ax = plt.subplots(figsize=(10, 6))
        df_bp = df.groupby(['HighBP', 'Diabetes_012']).size().unstack()
        df_bp.plot(kind='bar', stacked=True)
        plt.title("High Blood Pressure vs Diabetes")
        plt.xlabel("High Blood Pressure")
        plt.ylabel("Count")
        plt.legend(['No Diabetes', 'Prediabetes', 'Diabetes'])
        st.pyplot(fig)
        
        # General Health Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='GenHlth')
        plt.title("General Health Distribution")
        plt.xlabel("Health Level (1: Excellent, 5: Poor)")
        st.pyplot(fig)

def show_age_distribution(df):
    st.subheader("👥 Age Distribution Analysis")
    
    # Age distribution by diabetes status
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='Diabetes_012', y='Age')
    plt.title("Age Distribution by Diabetes Status")
    plt.xlabel("Diabetes Status (0: No Diabetes, 1: Prediabetes, 2: Diabetes)")
    plt.ylabel("Age Category")
    st.pyplot(fig)
    
    # Age pyramid by diabetes
    age_diabetes = pd.crosstab(df['Age'], df['Diabetes_012'])
    fig, ax = plt.subplots(figsize=(12, 8))
    age_diabetes.plot(kind='barh', stacked=True)
    plt.title("Age Distribution by Diabetes Status")
    plt.xlabel("Count")
    plt.ylabel("Age Category")
    plt.legend(['No Diabetes', 'Prediabetes', 'Diabetes'])
    st.pyplot(fig)

if __name__ == "__main__":
    main()