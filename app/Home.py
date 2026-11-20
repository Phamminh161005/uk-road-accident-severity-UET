"""Entry point: streamlit run app/Home.py"""

import streamlit as st

st.set_page_config(page_title="Tai nạn giao thông Vương quốc Anh", layout="wide")

pages = [
    st.Page("pages/home.py", title="Giới thiệu", default=True),
    st.Page("pages/overview.py", title="Tổng quan"),
]
st.navigation(pages).run()
