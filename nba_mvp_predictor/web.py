import streamlit as st

from nba_mvp_predictor import conf

def run():
    st.set_page_config(
        page_title=conf.web.tab_title,
        page_icon=":basketball:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.title(conf.web.page_title)
    st.sidebar.markdown(conf.web.sidebar_top_text)
    st.sidebar.markdown(conf.web.sidebar_bottom_text)
    st.title(conf.web.page_title)
