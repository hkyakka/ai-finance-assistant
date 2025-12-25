import streamlit as st
import pandas as pd
import numpy as np
import datetime

def render():
    st.header("Market View")

    ticker = st.text_input("Enter a stock ticker to analyze", "AAPL")

    if ticker:
        st.subheader(f"Analysis for {ticker}")

        # Placeholder for data freshness badge
        st.markdown(f"Data as of: **{datetime.date.today().strftime('%Y-%m-%d')}** <span style='color:green; font-weight:bold;'>&#11044;</span> Fresh", unsafe_allow_html=True)

        # Placeholder for trend line chart
        st.subheader("Price Trend")
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Open', 'High', 'Low'])
        st.line_chart(chart_data)

    else:
        st.info("Please enter a ticker symbol to see the market data.")
