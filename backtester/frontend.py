import streamlit as st
import pandas as pd
import time
import fields
allowed_variables = {name: getattr(fields, name) for name in dir(fields) 
                     if not name.startswith('__') and isinstance(getattr(fields, name), pd.DataFrame)}
from streamlit_option_menu import option_menu

# CSS tuỳ chỉnh cho giao diện đẹp hơn
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding: 0 !important;
    }
    .css-1d391kg {
        background-color: #2c2f33 !important;
        color: #ffffff !important;
    }
    .stTextInput input {
        background-color: #2c2f33 !important;
        color: #ffffff !important;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #7289da;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #5b6eae;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .stProgress .st-bo {
        background-color: #7289da;
    }
    .stTextArea textarea {
        background-color: #2c2f33 !important;
        color: white !important;
        border-radius: 8px;
    }
    .stSlider>div>div>div {
        background-color: #7289da;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #2c2f33 !important;
        color: #ffffff !important;
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
        'decay': 0.5,
        'truncation': 0.08,
        'pasteurization': 'True'
    }

saved_settings = st.session_state.saved_settings

region = st.sidebar.selectbox("Region", ['VN', 'US'], index=['VN', 'US'].index(saved_settings['region']))

if region == 'VN':
    universe_options = ['VN30', 'HNX30', 'VNALL']
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
decay = st.sidebar.text_input("Decay", saved_settings['decay'])
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
        'pasteurization': pasteurization
    }
    st.sidebar.success("Settings saved successfully!")

# Nội dung của từng trang
if selected == "Simulate":
    st.markdown("<h1 style='text-align: center; color: #7289da;'>Saigon Quant Alpha</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='text-align: center; color: #000000;'>Write Your Formula</h3>", unsafe_allow_html=True)
        selected_variable = st.selectbox("Select a variable to add to the formula", list(allowed_variables.keys()))
        code = st.text_area("Write your formula", "close")

        if st.button("Simulate"):
            progress_bar = st.progress(0)
            st.write(f"Simulating...")

            try:
                result = eval(code, {"__builtins__": None}, allowed_variables)
                st.write(f"Simulation Result")
                st.dataframe(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")

            for percent_complete in range(100):
                time.sleep(0.05)
                progress_bar.progress(percent_complete + 1)

    with col2:
        st.markdown("<h3 style='text-align: center; color: #000000;'>Simulation Results</h3>", unsafe_allow_html=True)
        st.write("Simulation results will appear here.")

    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='color: #cccccc;'>Powered by Saigon Quant. Developed for Alpha generation and trading insights.</p>
            <p style='color: #7289da; font-size: 14px;'>© 2024 Saigon Quant</p>
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
