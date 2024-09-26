#frontend.py
import streamlit as st
import plost
import pandas as pd
import time
from streamlit_option_menu import option_menu
from backend import *


# CSS tuỳ chỉnh cho giao diện đẹp hơn
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    /* Sidebar settings */
    .css-1d3b3hu {
        background-color: #f8f9fa !important;
        padding: 10px;
        border-right: 1px solid #e9ecef;
    }

    /* Main content container: widened to 95% */
    .css-18e3th9 {
        padding: 0 !important;
        max-width: 100% !important;  /* Increased from 90% to 95% */
        margin-left: auto;
        margin-right: auto;
    }

    /* Custom button styles */
    .stButton>button {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 5px !important;
    }

    /* Custom DataFrame styling */
    .dataframe {
        border: 1px solid #dee2e6 !important;
    }

    /* Reduced general padding for a more compact layout */
    .css-1avcm0n {
        padding: 0.5rem !important;  /* Reduced padding */
    }

    /* Center the page title */
    .css-1v0mbdj {
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True
)


# Thêm thanh điều hướng ở đầu trang
selected = option_menu(
    menu_title=None,  # Bỏ tiêu đề
    options=["Simulate", "Alphas", "Learn", "Data", "Operators", "Team", "Community"],  # Các tùy chọn menu
    icons=["graph-up-arrow", "lightning", "book", "database", "calculator", "people", "chat-left-dots"],  # Các biểu tượng tương ứng
    menu_icon="cast",  # Biểu tượng menu
    default_index=0,  # Mặc định trang đầu tiên
    orientation="horizontal",  # Sắp xếp theo chiều ngang
    styles={
        "container": {"padding": "5!important", "background-color": "#23272a"},
        "icon": {"color": "#ffffff", "font-size": "18px"},  # Màu biểu tượng đổi thành trắng
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "color": "#ffffff"},
        "nav-link-selected": {"background-color": "#7289da"},
    }
)

# Sidebar setup
st.sidebar.markdown("<h2 style='text-align: center; color: #000000;'>Settings</h2>", unsafe_allow_html=True)

# Xử lý phần cài đặt trang 'Simulate'
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
pasteurization = st.sidebar.selectbox("Pasteurization", ['True', 'False'], index=['True', 'False'].index(saved_settings['pasteurization']))
delay = st.sidebar.selectbox("Delay", [0, 1])

# Nút Apply để lưu cài đặt
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

# Nội dung của từng trang
if selected == "Simulate":
    st.markdown("<h1 style='text-align: center; color: #7289da;'>Saigon Quant Alpha</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([10, 10])

    with col1:
        universe_variables = list(dataframes[universe].keys())
        selected_variable = st.selectbox("Select a variable to add to the formula", universe_variables)
        st.header("Write Your Alpha")
        formula = st.text_area("Alpha", "close")
        if st.button("Run"):
            st.write("Simulating...")
            with st.spinner("Running simulation..."):
                time.sleep(2)
                result = simulation_results(formula, saved_settings)
                st.success("Simulation completed!")

    # Column for simulation results
    with col2:
        st.markdown("<h1 style='text-align: center;'>Simulation Results</h1>", unsafe_allow_html=True)
        if 'result' in locals():
            st.dataframe(result)
            st.write("Simulation results will appear here.")

    st.subheader("Analysis")
    st.write("This area can be used for detailed performance metrics and analysis.")

    # Footer
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='color: #cccccc;'>Powered by SaigonQuant. Developed for Alpha generation and trading insights.</p>
            <p style='color: #7289da; font-size: 14px;'>© 2024 SaigonQuant</p>
        </div>
        """, unsafe_allow_html=True
    )

elif selected == "Alphas":
    st.title("Alphas Page")
    st.write("Content for Alphas page.")

elif selected == "Learn":
    st.title("Learning Resources")
    st.write("Content for Learning Resources page.")

elif selected == "Data":
    st.title("Data Page")
    st.write("Content for Data page.")

elif selected == "Competitions":
    st.title("Competitions Page")
    st.write("Content for Competitions page.")

elif selected == "Team":
    st.title("Team Information")
    st.write("Content for Team page.")

elif selected == "Community":
    st.title("Community Page")
    st.write("Content for Community page.")
