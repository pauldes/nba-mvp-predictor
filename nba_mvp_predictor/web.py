import streamlit as st

from nba_mvp_predictor import conf


def run():
    print(conf)
    st.set_page_config(
        page_title=conf.web.page_title,
        # page_icon="nba_mvp_predictor/static/favicon.ico",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.title(conf.web.page_title)
