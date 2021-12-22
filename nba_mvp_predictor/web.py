from datetime import datetime
import os
import re

import streamlit as st
import pandas

from nba_mvp_predictor import conf, logger
from nba_mvp_predictor import load, evaluate, artifacts, download, analytics

# Constants
PAGE_PREDICTIONS = "Current year predictions"
PAGE_PERFORMANCE = "Model performance analysis"
CONFIDENCE_MODE_SOFTMAX = "Softmax-based"
CONFIDENCE_MODE_SHARE = "Share-based"

pandas.set_option("display.precision", 2)

def build_predictions():
    predictions = pandas.read_csv(
        "./data/predictions-artifact.csv.zip",
        sep=conf.data.predictions.sep,
        encoding=conf.data.predictions.encoding,
        compression="zip",
        index_col=0,
        dtype={},
    )
    predictions = predictions.set_index("PLAYER", drop=True)
    return predictions

@st.cache(ttl=3600) #1h cache
def download_predictions():
    date, url = artifacts.get_last_artifact("predictions.csv")
    logger.debug(f"Downloading history from {url}")
    download.download_data_from_url_to_file(url, "./data/predictions-artifact.csv.zip", auth=artifacts.get_github_auth())

@st.cache(ttl=3600) #1h cache
def download_history():
    date, url = artifacts.get_last_artifact("history.csv")
    logger.debug(f"Downloading history from {url}")
    download.download_data_from_url_to_file(url, "./data/history-artifact.csv.zip", auth=artifacts.get_github_auth())

def build_history():
    download_history()
    history = pandas.read_csv(
        "./data/history-artifact.csv.zip",
        sep=conf.data.history.sep,
        encoding=conf.data.history.encoding,
        compression="zip",
        index_col=False,
        dtype={},
    )
    history = history.rename(
        columns={"DATE": "date", "PLAYER": "player", "PRED": "prediction"}
    )
    history.date = pandas.to_datetime(history.date, format="%d-%m-%Y")
    return history


def prepare_history(stats, keep_top_n, confidence_mode, compute_probs_based_on_top_n):
    keep_players = stats.sort_values(by=["date", "prediction"], ascending=False)[
        "player"
    ].to_list()[:keep_top_n]
    for date in stats.date.unique():
        stats.loc[stats.date == date, "rank"] = stats.loc[
            stats.date == date, "prediction"
        ].rank(ascending=False)
        if confidence_mode == CONFIDENCE_MODE_SOFTMAX:
            stats.loc[stats.date == date, "chance"] = (
                evaluate.softmax(stats[stats.date == date]["prediction"]) * 100
            )
            # stats.loc[dataset.rank <= compute_probs_based_on_top_n, "chance"] = evaluate.softmax(dataset[dataset.rank <= compute_probs_based_on_top_n]["prediction"]) * 100
        else:
            stats.loc[stats.date == date, "chance"] = (
                evaluate.share(stats[stats.date == date]["prediction"]) * 100
            )
            # stats.loc[dataset.rank <= compute_probs_based_on_top_n, "chance"] = evaluate.share(dataset[dataset.rank <= compute_probs_based_on_top_n]["prediction"]) * 100
    stats = stats[stats["player"].isin(keep_players)]
    stats = stats.fillna(0.0)
    return stats

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#@st.cache
def inject_google_analytics_tag():
    index_path = os.path.dirname(st.__file__)+'/static/index.html'
    code = analytics.get_google_analytic_code()
    with open(index_path, 'r') as f:
        data = f.read()
        if len(re.findall('G-', data))==0:
            with open(index_path, 'w') as ff:
                newdata = re.sub('<head>','<head>' + code, data)
                ff.write(newdata)


def run():
    st.set_page_config(
        page_title=conf.web.tab_title,
        page_icon=":basketball:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.title(conf.web.page_title)
    inject_google_analytics_tag()
    local_css("./nba_mvp_predictor/custom.css")
    current_season = (
        datetime.now().year + 1 if datetime.now().month > 9 else datetime.now().year
    )
    st.markdown(
        f"""
    *Predicting the NBA Most Valuable Player for the {current_season-1}-{str(current_season)[-2:]} season using machine learning.*
    """
    )

    navigation_page = st.sidebar.radio(
        "Navigate to", [PAGE_PREDICTIONS, PAGE_PERFORMANCE]
    )
    st.sidebar.markdown(conf.web.sidebar_top_text)
    st.sidebar.markdown(conf.web.sidebar_bottom_text)

    predictions = build_predictions()
    initial_columns = list(predictions.columns)

    if navigation_page == PAGE_PREDICTIONS:

        st.header("Current year predictions")

        col1, col2 = st.columns(2)
        col1.subheader("Predicted top 3")
        col2.subheader("Prediction parameters")
        confidence_mode = col2.radio(
            "MVP probability estimation method",
            [CONFIDENCE_MODE_SHARE, CONFIDENCE_MODE_SOFTMAX],
        )
        compute_probs_based_on_top_n = col2.slider(
            "Number of players used to estimate probability",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
        )
        if confidence_mode == CONFIDENCE_MODE_SOFTMAX:
            predictions.loc[
                predictions.PRED_RANK <= compute_probs_based_on_top_n, "MVP probability"
            ] = (
                evaluate.softmax(
                    predictions[predictions.PRED_RANK <= compute_probs_based_on_top_n][
                        "PRED"
                    ]
                )
                * 100
            )
        else:
            predictions.loc[
                predictions.PRED_RANK <= compute_probs_based_on_top_n, "MVP probability"
            ] = (
                evaluate.share(
                    predictions[predictions.PRED_RANK <= compute_probs_based_on_top_n][
                        "PRED"
                    ]
                )
                * 100
            )
        predictions.loc[
            predictions.PRED_RANK > compute_probs_based_on_top_n, "MVP probability"
        ] = 0.0
        predictions["MVP probability"] = predictions["MVP probability"].map(
            "{:,.2f}%".format
        )
        predictions["MVP rank"] = predictions["PRED_RANK"]
        show_columns = ["MVP probability", "MVP rank"] + initial_columns[:]
        predictions = predictions[show_columns]

        top_3 = predictions["MVP probability"].head(3).to_dict()
        emojis = ["🥇", "🥈", "🥉"]

        for n, player_name in enumerate(top_3):
            title_level = "###" + n * "#"
            col1.markdown(
                f"""
            ##### {emojis[n]} **{player_name}**
            *{top_3[player_name]} chance to win MVP*
            """
            )


        #show_top_n = compute_probs_based_on_top_n
        show_top_n = min([compute_probs_based_on_top_n, 10])

        st.subheader(f"Predicted top {show_top_n}")

        col1, col2 = st.columns([2,3])
        col2.markdown("Player statistics")
        col2.dataframe(
            data=predictions.head(show_top_n), width=None, height=300,
        )
        
        barchart_data = predictions.head(show_top_n).copy()
        barchart_data["player"] = barchart_data.index
        barchart_data["chance"] = barchart_data["MVP probability"].str[:-1]
        barchart_data["chance"] = pandas.to_numeric(barchart_data["chance"])

        col1.markdown("Chart")
        col1.vega_lite_chart(
            barchart_data,
            {
                "mark": {
                    "type": "bar",
                    "point": True,
                    "tooltip": True,
                },
                "encoding": {
                    "x": {
                        "field": "chance",
                        "type": "quantitative",
                        "title":" MVP chance (%)",
                    },
                    "y": {
                        "field": "player",
                        "type": "ordinal",
                        "title": None,
                        "sort": "-x"
                    },
                    #"color": {"field": "player", "type": "nominal"},
                },
            },
            height=300,
            use_container_width=True,
        )

        st.subheader("Predictions history")
        col1, col2 = st.columns(2)
        keep_top_n = col2.slider(
            "Number of players to show",
            min_value=3,
            max_value=compute_probs_based_on_top_n,
            value=5,
            step=1,
        )
        variable_to_draw_dict = {
            "MVP chance (%)": "chance",
            "Predicted MVP share": "prediction",
        }
        variable_to_draw = col1.radio(
            "Variable to draw", list(variable_to_draw_dict.keys())
        )
        history = build_history()
        prepared_history = prepare_history(
            history, keep_top_n, confidence_mode, compute_probs_based_on_top_n
        )

        st.vega_lite_chart(
            prepared_history,
            {
                "mark": {
                    "type": "line",
                    "interpolate": "monotone",
                    "point": True,
                    "tooltip": True,
                },
                "encoding": {
                    "x": {"timeUnit": "yearmonthdate", "field": "date"},
                    "y": {
                        "field": variable_to_draw_dict[variable_to_draw],
                        "type": "quantitative",
                        "title": variable_to_draw,
                    },
                    "color": {"field": "player", "type": "nominal"},
                },
            },
            height=400,
            use_container_width=True,
        )

    elif navigation_page == PAGE_PERFORMANCE:
        st.header("Model performance analysis")
        st.warning("Work in progress")
