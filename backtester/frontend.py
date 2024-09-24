import streamlit as st
import time
from backend import DataRetrieval

if 'saved_settings' not in st.session_state:
    st.session_state.saved_settings = {
        'region': 'VN',
        'universe': 'VN30',
        'neutralization': 'None',
        'decay': 0.5,
        'truncation': 0.08,
        'pasteurization': 'True'
    }

st.sidebar.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #F0F4FA;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #008CBA;
        color: white;
        border: none;
        border-radius: 15px;
        padding: 10px 25px;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #005F73;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='text-align: center; color: #008CBA; font-family: Arial, sans-serif; margin-bottom: 50px;'>
        <i class="fas fa-chart-line"></i> Saigon Quant Alpha
    </h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<h2 style='text-align: center;'>Settings</h2>", unsafe_allow_html=True)

saved_settings = st.session_state.saved_settings

region = st.sidebar.selectbox("Region", ['VN', 'US'], index=['VN', 'US'].index(saved_settings['region']))

if region == 'VN':
    universe_options = ['VN30', 'HNX30', 'VNALL']
else:
    universe_options = ['US1000']

universe = st.sidebar.selectbox("Universe", universe_options, index=universe_options.index(saved_settings['universe']))

if region == 'VN':
    neutral = ['None']
else:
    neutral = ['Sub-Industry', 'Industry', 'Market', 'Sector']

neutralization = st.sidebar.selectbox("Neutralization", neutral, index=neutral.index(saved_settings['neutralization']))
decay = st.sidebar.text_input("Decay", saved_settings['decay'])
truncation = st.sidebar.text_input("Truncation", saved_settings['truncation'])
pasteurization = st.sidebar.selectbox("Pasteurization", ['True', 'False'], index=['True', 'False'].index(saved_settings['pasteurization']))
delay = st.sidebar.selectbox("Delay", [0, 1])

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

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='text-align: center; color: #008CBA;'>Write Your Formula</h3>", unsafe_allow_html=True)
    code = st.text_area("Write your formula", "close - open")

    if st.button("Simulate"):
        progress_bar = st.progress(0)
        st.write(f"Simulating...")

        for percent_complete in range(100):
            time.sleep(0.05)
            progress_bar.progress(percent_complete + 1)
        st.write(f"Result of simulation: [Placeholder]")

with col2:
    st.markdown("<h3 style='text-align: center; color: #008CBA;'>Simulation Results</h3>", unsafe_allow_html=True)
    st.write("Simulation results will appear here.")

st.markdown(
    """
    <div style='text-align: center; margin-top: 50px;'>
        <p style='color: grey; font-family: Arial, sans-serif;'>Powered by Saigon Quant. Developed for Alpha generation and trading insights.</p>
        <p style='color: #008CBA; font-size: 14px;'>© 2024 Saigon Quant</p>
    </div>
    """, unsafe_allow_html=True
)
