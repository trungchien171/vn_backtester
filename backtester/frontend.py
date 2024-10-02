#frontend.py
import streamlit as st
import hashlib
import pandas as pd
import numpy as np
import time
import toml
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload
from streamlit_option_menu import option_menu
from simulation import simulation_results
from data import dataframes
from check_submissions import run_tests, display_test_results
from operators import operators

def authenticate_gdrive():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["google_drive"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=credentials)

def load_user_data(drive_service):
    file_id = get_user_data_file_id(drive_service)
    if file_id:
        request = drive_service.files().get_media(fileId=file_id)
        file_data = io.BytesIO()
        downloader = MediaIoBaseDownload(file_data, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_data.seek(0)
        return pd.read_csv(file_data)
    else:
        return pd.DataFrame(columns=["username", "password"])

def get_user_data_file_id(drive_service):
    results = drive_service.files().list(q="name='user_data.csv'", fields="files(id, name)").execute()
    items = results.get("files", [])
    if not items:
        return None
    return items[0]["id"]

def save_user_data(drive_service, df):
    file_id = get_user_data_file_id(drive_service)
    file_metadata = {"name": "user_data.csv"}
    file_data = io.BytesIO()
    df.to_csv(file_data, index=False)
    file_data.seek(0)

    media = MediaIoBaseUpload(file_data, mimetype="text/csv", resumable=True)
    if file_id:
        drive_service.files().update(fileId=file_id, media_body=media).execute()
    else:
        drive_service.files().create(body=file_metadata, media_body=media).execute()
def rerun():
    st.rerun()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_account(username, password, drive_service, user_data):
    if username in user_data["username"].values:
        return False
    new_entry = pd.DataFrame({"username": [username], "password": [hash_password(password)]})
    user_data = pd.concat([user_data, new_entry], ignore_index=True)
    save_user_data(drive_service, user_data)
    return True

def login(username, password, user_data):
    if username not in user_data["username"].values:
        return False, "Username does not exist. Please sign up first."
    stored_password = user_data.loc[user_data["username"] == username, "password"].values[0]
    if stored_password == hash_password(password):
        return True, "Logged in successfully!"
    else:
        return False, "Incorrect password."

drive_service = authenticate_gdrive()
user_data = load_user_data(drive_service)

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
            top: 20px;  /* Position at the top */
            right: 20px; /* Align to the right */
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

hide_st_style = """

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

"""
st.markdown(hide_st_style, unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            login_success, message = login(login_username, login_password, user_data)
            if login_success:
                st.session_state.logged_in = True
                st.success(message)
                rerun()
            else:
                st.error(message)

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
                if create_account(register_username, register_password, drive_service, user_data):
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
        st.session_state.login_success = False
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