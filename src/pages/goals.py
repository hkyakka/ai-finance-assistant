import streamlit as st
import pandas as pd
import numpy as np

def render():
    st.header("Financial Goals")

    risk_appetite = st.select_slider(
        'Select your risk appetite',
        options=['Conservative', 'Moderate', 'Aggressive'],
        value='Moderate'
    )

    st.subheader("Portfolio Projection")

    # Adjust projection based on risk appetite
    if risk_appetite == 'Conservative':
        projection_factor = 0.05
    elif risk_appetite == 'Moderate':
        projection_factor = 0.08
    else: # Aggressive
        projection_factor = 0.12

    # Placeholder projection chart
    years = np.arange(1, 21)
    initial_investment = 100000
    projected_value = initial_investment * (1 + projection_factor) ** years

    chart_data = pd.DataFrame({
        'Year': years,
        'Projected Value': projected_value
    })

    st.line_chart(chart_data.rename(columns={'Year':'index'}).set_index('index'))

    st.markdown(f"**Your selected risk appetite: {risk_appetite}**")
    st.write("The chart above shows a hypothetical projection of your portfolio's growth over the next 20 years based on your selected risk appetite.")
