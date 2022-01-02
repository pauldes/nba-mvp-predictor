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
    download_predictions()
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

def mvp_found_pct(performances):
    metrics = (
        performances["Predicted MVP"] == performances["True MVP"]
    ).sum() / len(performances)
    metrics = int(metrics * 100)
    return str(metrics) + " %"

def avg_real_mvp_rank(performances):
    metrics = (performances["True rank of predicted MVP"]).mean()
    return "%.1f" % metrics

def avg_pred_mvp_rank(performances):
    metrics = (performances["Predicted rank of true MVP"]).mean()
    return "%.1f" % metrics

def build_performances():
    download_performances()
    performances = pandas.read_csv(
        "./data/performances-artifact.csv.zip",
        sep=conf.data.performances.sep,
        encoding=conf.data.performances.encoding,
        compression="zip",
        index_col=0,
        dtype={},
    )
    performances.index = performances.index.rename("Season")
    performances.REAL_RANK = performances.REAL_RANK.astype("Int32")
    performances.PRED_RANK = performances.PRED_RANK.astype("Int32")
    performances = performances.rename(
        columns={
            "Pred. MVP": "Predicted MVP",
            "REAL_RANK": "True rank of predicted MVP",
            "PRED_RANK": "Predicted rank of true MVP",
            "PRED": "Predicted award share",
            "TRUTH": "True award share"
        }
    )
    performances = performances.sort_index(ascending=False)
    performances.loc[performances["True MVP"]==performances["Predicted MVP"], "Model is right"] = "✔️" #"☑️"
    performances.loc[performances["True MVP"]!=performances["Predicted MVP"], "Model is right"] = "❌"
    performances = performances[["True MVP", "Model is right", "Predicted MVP", "True rank of predicted MVP", "Predicted rank of true MVP"]]
    return performances

@st.cache(ttl=3600) #1h cache
def download_performances():
    date, url = artifacts.get_last_artifact("performances.csv")
    logger.debug(f"Downloading performances from {url}")
    download.download_data_from_url_to_file(url, "./data/performances-artifact.csv.zip", auth=artifacts.get_github_auth())


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
    today_date = pandas.Timestamp(datetime.today().date())
    history["days_ago"] = (today_date - history.date).dt.days.astype(int)
    return history


def prepare_history(stats, keep_top_n, confidence_mode, compute_probs_based_on_top_n, keep_last_days=None):
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
    if keep_last_days is not None:
        stats = stats[stats.days_ago <= keep_last_days]
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
    predictions = build_predictions()
    history = build_history()
    performances = build_performances()
    current_season = (
        datetime.now().year + 1 if datetime.now().month > 9 else datetime.now().year
    )
    last_update = history.date.max().date()
    st.markdown(
        f"""
    *Predicting the NBA Most Valuable Player for the {current_season-1}-{str(current_season)[-2:]} season using machine learning.*
    *Last update : {last_update}.*
    """
    )

    navigation_page = st.sidebar.radio(
        "Navigate to", [PAGE_PREDICTIONS, PAGE_PERFORMANCE]
    )
    st.sidebar.markdown(conf.web.sidebar_top_text)
    st.sidebar.markdown(conf.web.sidebar_bottom_text)

    
    initial_columns = list(predictions.columns)

    if navigation_page == PAGE_PREDICTIONS:

        st.header("Current year predictions")

        col1, col2 = st.columns(2)
        col1.subheader("Predicted top 3")
        col2.subheader("Prediction parameters")
        confidence_mode = col2.radio(
            "Method used to estimate MVP probability",
            [CONFIDENCE_MODE_SHARE, CONFIDENCE_MODE_SOFTMAX],
        )
        compute_probs_based_on_top_n = col2.slider(
            "Number of players used to estimate probability",
            min_value=5,
            max_value=15,
            value=10,
            step=5,
            format="%d players"
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
            *{top_3[player_name]} probability to win MVP*
            """
            )


        show_top_n = compute_probs_based_on_top_n
        #show_top_n = min([compute_probs_based_on_top_n, 10])

        st.subheader(f"Predicted top {show_top_n}")

        col1, col2 = st.columns(2)
        #col2.markdown("Player statistics")
        cols = [col for col in predictions.columns if "MVP" not in col and "PRED" not in col]
        col2.dataframe(
            data=predictions.head(show_top_n)[cols], width=None, height=300,
        )
        
        predictions["player"] = predictions.index
        predictions["chance"] = predictions["MVP probability"].str[:-1]
        predictions["chance"] = pandas.to_numeric(predictions["chance"])

        #col1.markdown("Chart")
        col1.vega_lite_chart(
            predictions.head(show_top_n),
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
                        "title":" MVP probability (%)",
                    },
                    "y": {
                        "field": "player",
                        "type": "nominal",
                        "title": None,
                        "sort": "-x"
                    },
                    "color": {
                        "field": "chance",
                        "type": "quantitative",
                        "title":None,
                        "legend":None,
                        "scale": {"scheme": "purplebluegreen"},
                    },
                },
            },
            height=350,
            use_container_width=True,
        )

        st.subheader("Predictions history")
        col1, col2, col3 = st.columns([2, 3, 3])
        keep_top_n = col2.slider(
            "Number of players to show",
            min_value=3,
            max_value=compute_probs_based_on_top_n,
            value=5,
            step=1,
            format="%d players"
        )
        
        variable_to_draw_dict = {
            "MVP probability (%)": "chance",
            "Predicted MVP share": "prediction",
        }
        variable_to_draw = col1.radio(
            "Variable to draw", list(variable_to_draw_dict.keys())
        )
        
        num_past_days = col3.slider(
            "Show history for last",
            min_value=max(int(history.days_ago.min()), 3),
            max_value=int(history.days_ago.max()),
            value=min(int(history.days_ago.max()), 30),
            step=1,
            format="%d days"
        )
        prepared_history = prepare_history(
            history, keep_top_n, confidence_mode, compute_probs_based_on_top_n, keep_last_days=num_past_days
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
                    "x": {
                        "timeUnit": "yearmonthdate",
                        "field": "date",
                        "title":"Date"
                    },
                    "y": {
                        "field": variable_to_draw_dict[variable_to_draw],
                        "type": "quantitative",
                        "title": variable_to_draw,
                    },
                    "color": {
                        "field": "player",
                        "type": "nominal",
                        "scale": {"scheme": "category20"},
                        "legend":{
                            "orient":"bottom-left",
                        }
                    },
                },
            },
            height=400,
            use_container_width=True,
        )

    elif navigation_page == PAGE_PERFORMANCE:
        st.header("Model performance analysis")
        col1, col2 = st.columns(2)
        #col1
        percentage = mvp_found_pct(performances)
        num_test_seasons = len(performances)
        avg_real_rank = avg_real_mvp_rank(performances)
        avg_pred_rank = avg_pred_mvp_rank(performances)

        col1.markdown(f"##### All {num_test_seasons} seasons ({performances.index.min()}-{performances.index.max()})")
        col1.markdown(
            f"""
        - **{percentage}** of MVPs correctly found
        - The true MVP is ranked **{avg_real_rank}** by the model in average
        - The true rank of the predicted MVP is **{avg_pred_rank}** in average
        """
        )
        #col2
        performances_last10 = performances.head(10)
        percentage = mvp_found_pct(performances_last10)
        num_test_seasons = len(performances_last10)
        avg_real_rank = avg_real_mvp_rank(performances_last10)
        avg_pred_rank = avg_pred_mvp_rank(performances_last10)
        col2.markdown(f"##### Last {num_test_seasons} seasons ({performances_last10.index.min()}-{performances_last10.index.max()})")
        col2.markdown(
            f"""
        - **{percentage}** of MVPs correctly found
        - The true MVP is ranked **{avg_real_rank}** by the model in average
        - The true rank of the predicted MVP is **{avg_pred_rank}** in average
        """
        )
        # dataframe
        #st.markdown(f"##### Details")
        st.dataframe(data=performances, width=None, height=None)
        st.markdown(
        """
        Predictions of the model are made on the unseen season using holdout method.
        """
        )

    else:
        st.error("Unknown page selected.")
