# Home.py (archivo principal)
import streamlit as st
from utils.session_state import initialize_session_state
from pages.home import show_general_home, show_technical_home

def setup_page_config():
    """Configuración inicial de la página"""
    st.set_page_config(
        page_title="Diabetes Analysis System",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def setup_styles():
    """Configuración de estilos CSS"""
    st.markdown("""
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    setup_page_config()
    setup_styles()
    initialize_session_state()

    # Selector de tipo de usuario en el sidebar
    with st.sidebar:
        st.title("User Type")
        user_type = st.radio(
            "Select user type:",
            options=["General User", "Data Analyst"],
            key="user_type",
            horizontal=True
        )

    # Mostrar la página principal según el tipo de usuario
    if user_type == "General User":
        show_general_home()
    else:
        show_technical_home()

    # Footer
    st.markdown("---")
    st.markdown("Developed for Data Analysis Course 2024-2")

if __name__ == "__main__":
    main()