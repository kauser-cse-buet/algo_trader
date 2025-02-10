import streamlit as st
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Title of the App
st.title(config['app_name'])

# App Description with Robot Icon
st.markdown(f"""
### Welcome to 🤖 **{config['app_name']}**, {config['app_description']} designed to help you make smart investment decisions effortlessly.
""")

# Sidebar for user inputs
st.sidebar.header("Investor Profile")

# User Inputs
investment_amount = st.sidebar.number_input("Investment Amount ($)", min_value=config['min_investment'], step=config['investment_step'])
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", config['risk_tolerance'])
investment_horizon = st.sidebar.slider("Investment Horizon (Years)", config['investment_horizon']['min'], config['investment_horizon']['max'], config['investment_horizon']['default'])
preferred_sectors = st.sidebar.multiselect(
    "Preferred Sectors", 
    config['preferred_sectors']
)

# Button to submit inputs
if st.sidebar.button("Generate Investment Strategy"):
    # Display user inputs
    st.subheader("Your Investment Preferences")
    st.write(f"**Investment Amount:** ${investment_amount}")
    st.write(f"**Risk Tolerance:** {risk_tolerance}")
    st.write(f"**Investment Horizon:** {investment_horizon} years")
    st.write(f"**Preferred Sectors:** {', '.join(preferred_sectors) if preferred_sectors else 'None'}")

    # Mock output (This would be replaced with actual AI/ML recommendations)
    st.subheader("Recommended Investment Strategy")
    st.write("Based on your preferences, Stock Buddy recommends a diversified portfolio with a focus on:")
    if preferred_sectors:
        st.write(f"- **Sectors:** {', '.join(preferred_sectors)}")
    else:
        st.write("- A balanced mix across all major sectors")

    st.write(f"- **Risk Management:** Tailored to your selected risk tolerance")
    st.write("- **Rebalancing:** Quarterly reviews to optimize returns")
else:
    st.write("Use the sidebar to input your investment preferences.")
