#frontend.py
import streamlit as st
import hashlib
import pandas as pd
import numpy as np
import time
import toml
import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from streamlit_option_menu import option_menu
from simulation import simulation_results
from data import dataframes
from check_submissions import run_tests, display_test_results
from operators import operators

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)

def rerun():
    st.markdown("<script>window.location.reload();</script>", unsafe_allow_html=True)

def init_connection():
    db_secrets = st.secrets["database"]
    user = db_secrets["USER"]
    password = db_secrets["PASSWORD"]
    host = db_secrets["HOST"]
    port = db_secrets["PORT"]
    database = db_secrets["DATABASE"]

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_account(username, password):
    session = init_connection()
    try:
        if session.query(User).filter_by(username=username).first() is not None:
            return False
        new_user = User(username=username, password=hash_password(password))
        session.add(new_user)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()

def login(username, password):
    session = init_connection()
    try:
        user = session.query(User).filter_by(username=username).first()
        if user and user.password == hash_password(password):
            return True
        return False
    finally:
        session.close()

st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="SaigonQuant Alpha")

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Style adjustments for a modern, clean interface */
    .css-18e3th9 {
        max-width: 95% !important;
    }
    .stButton > button {
        background-color: #4CAF50 !important;  /* Green button color */
        color: white !important;
    }
    .st-expanderHeader {
        font-size: 18px;
        font-weight: bold;
    }
    .st-tabs-header {
        color: #1a1a1a !important;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown(
    """
    <style>
        .footer-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 100;
        }
        .footer-container a {
            margin: 0 10px;
            color: #7289da;
            font-size: 24px;
            text-decoration: none;
        }
        .footer-container a:hover {
            color: #4CAF50; /* Green hover color */
        }
    </style>
    <div class="footer-container">
        <a href="https://discord.com/channels/1290146471727075348/1290146472406421516" target="_blank">
            <i class="fab fa-discord"></i>
        </a>
        <a href="https://www.linkedin.com/in/trantrungchien/" target="_blank">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://www.facebook.com/chien.trung.357622/" target="_blank">
            <i class="fab fa-facebook"></i>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey && event.key === 'Enter') {
            document.querySelector('button[title="Run"]').click();
        }
    });
    </script>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login(login_username, login_password):
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.markdown("<h1 style='text-align: center;'>Sign Up</h1>", unsafe_allow_html=True)
        register_username = st.text_input("Username", key="register_username")
        register_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

        if st.button("Sign Up"):
            if register_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(register_password) == 0 or len(register_username) == 0:
                st.error("Username and password cannot be empty.")
            else:
                if create_account(register_username, register_password):
                    st.success("Account created successfully! You can now login.")
                else:
                    st.error("Username already taken or registration failed.")

else:
    selected = option_menu(
        menu_title=None,
        options=["Simulate", "Alphas", "Learn", "Data", "Operators", "Team", "Community"],
        icons=["graph-up-arrow", "lightning", "book", "database", "calculator", "people", "chat-left-dots"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "5!important", "background-color": "#23272a"},
            "icon": {"color": "#ffffff", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "color": "#ffffff"},
            "nav-link-selected": {"background-color": "#7289da"},
        }
    )

    st.markdown("<h1 style='text-align: center; color: #7289da;'>Saigon Quant Alpha</h1>", unsafe_allow_html=True)

    if selected == "Simulate":
        st.sidebar.markdown("<h2 style='text-align: center; color: #000000;'>Settings</h2>", unsafe_allow_html=True)

        if 'saved_settings' not in st.session_state:
            st.session_state.saved_settings = {
                'region': 'VN',
                'universe': 'VN30',
                'neutralization': 'None',
                'decay': 4,
                'truncation': 0.08,
                'pasteurization': 'False',
                'delay': 0
            }

        saved_settings = st.session_state.saved_settings

        region = st.sidebar.selectbox("Region", ['VN', 'US'], index=['VN', 'US'].index(saved_settings['region']))

        if region == 'VN':
            universe_options = ['VN30', 'VN100', 'VNALL']
        else:
            universe_options = ['US1000']

        if saved_settings['universe'] not in universe_options:
            saved_settings['universe'] = universe_options[0]

        universe = st.sidebar.selectbox("Universe", universe_options, index=universe_options.index(saved_settings['universe']))

        if region == 'VN':
            neutral = ['None']
        else:
            neutral = ['Sub-Industry', 'Industry', 'Market', 'Sector']

        if saved_settings['neutralization'] not in neutral:
            saved_settings['neutralization'] = neutral[0]

        neutralization = st.sidebar.selectbox("Neutralization", neutral, index=neutral.index(saved_settings['neutralization']))
        decay = st.sidebar.slider("Decay", min_value=0, max_value=20, value=saved_settings['decay'])
        truncation = st.sidebar.text_input("Truncation", saved_settings['truncation'])
        pasteurization = st.sidebar.selectbox("Pasteurization", ['False'], index=['False'].index(saved_settings['pasteurization']))
        delay = st.sidebar.selectbox("Delay", [0, 1])

        if st.sidebar.button("Apply"):
            st.session_state.saved_settings = {
                'region': region,
                'universe': universe,
                'neutralization': neutralization,
                'decay': decay,
                'truncation': truncation,
                'pasteurization': pasteurization,
                'delay': delay
            }
            st.sidebar.success("Settings saved successfully!")

        col1, col2 = st.columns([10, 10])

        with col1:
            st.header("Write Your Alpha")
            formula = st.text_area("Alpha", "")

            if st.button("Run", key="run_button", help="Press Ctrl + Enter to run the simulation."):
                with st.spinner("Running simulation..."):
                    time.sleep(2)
                    fig, summary = simulation_results(formula, saved_settings)
                st.success("Simulation completed!")

        with col2:
            st.markdown("<h1 style='text-align: center;'>Simulation Results</h1>", unsafe_allow_html=True)
            if 'fig' in locals():
                st.plotly_chart(fig)
            else:
                st.write("Simulation results will appear here.")

        if 'summary' in locals():
            overall_metrics = summary.loc["All"]

            overall_sharpe = overall_metrics["Sharpe"]
            overall_turnover = overall_metrics["Turnover (%)"]
            overall_returns = overall_metrics["Returns (%)"]
            overall_fitness = overall_metrics["Fitness"]
            overall_drawdown = overall_metrics["Drawdown (%)"]
            overall_margin = overall_metrics["Margin (%)"]

            metrics_col, test_col = st.columns([2, 1])

            with metrics_col:
                metrics_col.subheader("Overall Metrics")
                col1, col2, col3, col4, col5, col6 = metrics_col.columns(6)

                col1.metric("Sharpe", f"{overall_sharpe:.2f}")
                col2.metric("Turnover", f"{overall_turnover:.2f}%")
                col3.metric("Returns", f"{overall_returns:.2f}%")
                col4.metric("Fitness", f"{overall_fitness:.2f}")
                col5.metric("Drawdown", f"{overall_drawdown:.2f}%")
                col6.metric("Margin", f"{overall_margin:.2f}%")

                metrics_col.subheader("Yearly Performance Breakdown")
                metrics_col.dataframe(
                    summary.style.format(
                        {
                            "Sharpe": "{:.2f}",
                            "Turnover (%)": "{:.2f}",
                            "Returns (%)": "{:.2f}",
                            "Fitness": "{:.2f}",
                            "Drawdown (%)": "{:.2f}",
                            "Margin (%)": "{:.2f}",
                            "Long Side": "{:.0f}",
                            "Short Side": "{:.0f}", 
                        }
                    )
                )

            test_results = run_tests(overall_metrics)
            display_test_results(test_results, test_col)

    elif selected == "Alphas":
        st.title("Alphas Page")
        st.write("Content for Alphas page.")

    elif selected == "Learn":
        st.title("Learning Resources")
        st.write("Content for Learning Resources page.")

    elif selected == "Data":
        st.subheader("Select Region and Universe")

        region = st.selectbox("Region", ['VN', 'US'], key="region_selectbox")

        universe_options = ['VN30', 'VN100', 'VNALL'] if region == 'VN' else ['US1000']

        universe = st.selectbox("Universe", universe_options, key="universe_selectbox")

        st.subheader(f"Available Data Fields:")

        if 'dataframes' in globals() and universe in dataframes:
            universe_data = dataframes[universe]

            columns = st.columns(5)

            for idx, dataset_name in enumerate(universe_data.keys()):
                col = columns[idx % 5]
                col.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ccc; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px; 
                        background-color: #f9f9f9;
                        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);">
                        <strong>{dataset_name}</strong>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.warning("No data available for the selected universe or an error occurred.")

    elif selected == "Competitions":
        st.title("Competitions Page")
        st.write("Content for Competitions page.")

    elif selected == "Operators":
        for operator_type, operator_dict in operators.items():
            st.markdown(f"### {operator_type}")
            columns = st.columns(5)
            for idx, operator_name in enumerate(operator_dict.keys()):
                col = columns[idx % 5]
                col.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ccc; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px; 
                        background-color: #f9f9f9;
                        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);">
                        <strong>{operator_name}</strong>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    elif selected == "Team":
        st.title("Team Information")
        st.write("Content for Team page.")

    elif selected == "Community":
        st.title("Community Page")
        st.write("Content for Community page.")

    st.markdown("<div class='top-right'>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.logged_in = False
        rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='color: #cccccc;'>Powered by SaigonQuant. Developed for Alpha generation and trading insights.</p>
            <p style='color: #7289da; font-size: 14px;'>© 2024 SaigonQuant</p>
        </div>
        """, unsafe_allow_html=True
    )