import os
import re
from datetime import date, datetime

import numpy
import pandas
import streamlit as st

from nba_mvp_predictor import analytics, artifacts, conf, download, evaluate, logger

# Constants
PAGE_PREDICTIONS = "Current predictions"
PAGE_EXPLICABILITY = "Explain predictions"
PAGE_PERFORMANCE = "Model performance"
CONFIDENCE_MODE_SOFTMAX = "Softmax-based"
CONFIDENCE_MODE_SHARE = "Share-based"
SEASON_END_DATE = date(year=2024, month=4, day=10)

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
    metrics = (performances["Predicted MVP"] == performances["True MVP"]).sum() / len(
        performances
    )
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
            "TRUTH": "True award share",
        }
    )
    performances = performances.sort_index(ascending=False)
    performances.loc[
        performances["True MVP"] == performances["Predicted MVP"], "Model is right"
    ] = "✔️"
    performances.loc[
        performances["True MVP"] != performances["Predicted MVP"], "Model is right"
    ] = "❌"
    performances = performances[
        [
            "True MVP",
            "Model is right",
            "Predicted MVP",
            "True rank of predicted MVP",
            "Predicted rank of true MVP",
        ]
    ]
    return performances


@st.cache_data(ttl=3600)  # 1h cache
def download_performances():
    date, url = artifacts.get_last_artifact("performances.csv")
    logger.debug(f"Downloading performances from {url}")
    download.download_data_from_url_to_file(
        url, "./data/performances-artifact.csv.zip", auth=artifacts.get_github_auth()
    )


@st.cache_data(ttl=3600)  # 1h cache
def download_predictions():
    date, url = artifacts.get_last_artifact("predictions-2024.csv")
    logger.debug(f"Downloading predictions from {url}")
    download.download_data_from_url_to_file(
        url, "./data/predictions-artifact.csv.zip", auth=artifacts.get_github_auth()
    )


@st.cache_data(ttl=3600)  # 1h cache
def download_shap_values():
    date, url = artifacts.get_last_artifact("shap_values-2024.csv")
    logger.debug(f"Downloading shap values from {url}")
    download.download_data_from_url_to_file(
        url, "./data/shap_values-2024.csv.zip", auth=artifacts.get_github_auth()
    )


def build_shap_values():
    use_local_file = False
    if use_local_file:
        shap_values = pandas.read_csv(
            "./data/shap_values-2024.csv",
            sep=conf.data.shap_values.sep,
            encoding=conf.data.shap_values.encoding,
            compression=None,
            index_col=0,
            dtype={},
        )
    else:
        download_shap_values()
        shap_values = pandas.read_csv(
            "./data/shap_values-2024.csv.zip",
            sep=conf.data.shap_values.sep,
            encoding=conf.data.shap_values.encoding,
            compression="zip",
            index_col=0,
            dtype={},
        )
    return shap_values


@st.cache_data(ttl=3600)  # 1h cache
def download_history():
    date, url = artifacts.get_last_artifact("history-2024.csv")
    logger.debug(f"Downloading history from {url}")
    download.download_data_from_url_to_file(
        url, "./data/history-artifact.csv.zip", auth=artifacts.get_github_auth()
    )


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
    last_date_in_season = pandas.Timestamp(
        min([datetime.today().date(), SEASON_END_DATE])
    )
    history["days_ago"] = (last_date_in_season - history.date).dt.days.astype(int)
    history = history[history.date.dt.date <= SEASON_END_DATE]
    return history


def prepare_history(
    stats,
    keep_top_n,
    confidence_mode,
    compute_probs_based_on_top_n,
    keep_last_days=None,
):
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
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


def inject_google_analytics_tag():
    index_path = os.path.dirname(st.__file__) + "/static/index.html"
    code = analytics.get_google_analytic_code()
    with open(index_path, "r") as f:
        data = f.read()
        if len(re.findall("G-", data)) == 0:
            with open(index_path, "w") as ff:
                newdata = re.sub("<head>", "<head>" + code, data)
                ff.write(newdata)


def remove_trailing_sequence(string, sequence):
    if string.endswith(sequence):
        return string[: -len(sequence)]
    else:
        return string


def format_feature_name(feature_name) -> str:
    # Rename _advanced features - they are unique anyway
    feature_name = remove_trailing_sequence(feature_name, "_advanced")
    feature_name = feature_name.replace("_", " ")
    return feature_name


def run():
    st.set_page_config(
        page_title=conf.web.tab_title,
        page_icon=":basketball:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.title(conf.web.page_title)
    if conf.enable_google_analytics:
        inject_google_analytics_tag()
    local_css("./nba_mvp_predictor/custom.css")
    if conf.web.enable_web:
        try:
            predictions = build_predictions()
            building_predictions_succeeded = True
        except (OSError, Exception) as e:
            logger.error(f"Failed to build predictions {e}", exc_info=False)
            predictions = pandas.DataFrame(
                columns=[
                    "PLAYER",
                    "POS",
                    "AGE",
                    "TEAM",
                    "G",
                    "GS",
                    "MP",
                    "FG_per_game",
                    "FGA_per_game",
                    "FG%",
                    "3P_per_game",
                    "3PA_per_game",
                    "3P%",
                    "2P_per_game",
                    "2PA_per_game",
                    "2P%",
                    "EFG%_per_game",
                    "FT_per_game",
                    "FTA_per_game",
                    "FT%",
                    "ORB_per_game",
                    "DRB_per_game",
                    "TRB_per_game",
                    "AST_per_game",
                    "STL_per_game",
                    "BLK_per_game",
                    "TOV_per_game",
                    "PF_per_game",
                    "PTS_per_game",
                    "SEASON",
                    "FG_per_36min",
                    "FGA_per_36min",
                    "3P_per_36min",
                    "3PA_per_36min",
                    "2P_per_36min",
                    "2PA_per_36min",
                    "FT_per_36min",
                    "FTA_per_36min",
                    "ORB_per_36min",
                    "DRB_per_36min",
                    "TRB_per_36min",
                    "AST_per_36min",
                    "STL_per_36min",
                    "BLK_per_36min",
                    "TOV_per_36min",
                    "PF_per_36min",
                    "PTS_per_36min",
                    "FG_per_100poss",
                    "FGA_per_100poss",
                    "3P_per_100poss",
                    "3PA_per_100poss",
                    "2P_per_100poss",
                    "2PA_per_100poss",
                    "FT_per_100poss",
                    "FTA_per_100poss",
                    "ORB_per_100poss",
                    "DRB_per_100poss",
                    "TRB_per_100poss",
                    "AST_per_100poss",
                    "STL_per_100poss",
                    "BLK_per_100poss",
                    "TOV_per_100poss",
                    "PF_per_100poss",
                    "PTS_per_100poss",
                    "ORTG_per_100poss",
                    "DRTG_per_100poss",
                    "PER_advanced",
                    "TS%_advanced",
                    "3PAR_advanced",
                    "FTR_advanced",
                    "ORB%_advanced",
                    "DRB%_advanced",
                    "TRB%_advanced",
                    "AST%_advanced",
                    "STL%_advanced",
                    "BLK%_advanced",
                    "TOV%_advanced",
                    "USG%_advanced",
                    "OWS_advanced",
                    "DWS_advanced",
                    "WS_advanced",
                    "WS/48_advanced",
                    "OBPM_advanced",
                    "DBPM_advanced",
                    "BPM_advanced",
                    "VORP_advanced",
                    "MVP_VOTES_SHARE",
                    "MVP_WINNER",
                    "MVP_PODIUM",
                    "MVP_CANDIDATE",
                    "W",
                    "L",
                    "W/L%",
                    "GB",
                    "PW",
                    "PL",
                    "PS/G",
                    "PA/G",
                    "CONF",
                    "CONF_RANK",
                    "PRED",
                    "PRED_RANK",
                ]
            )
            building_predictions_succeeded = False
        try:
            history = build_history()
            last_update = str(history.date.max().date())
            building_history_succeeded = True
        except (OSError, Exception) as e:
            logger.error(f"Failed to build history {e}", exc_info=False)
            history = pandas.DataFrame(columns=["DATE", "PLAYER", "PRED"])
            last_update = "N/A"
            building_history_succeeded = False
        try:
            performances = build_performances()
            building_performances_succeeded = True
        except (OSError, Exception) as e:
            logger.error(f"Failed to build performances {e}", exc_info=False)
            performances = pandas.DataFrame(
                columns=[
                    "DATE",
                    "SEASON",
                    "TRUTH",
                    "PRED",
                    "AE",
                    "Pred. MVP",
                    "True MVP",
                    "REAL_RANK",
                    "PRED_RANK",
                    "Real MVP rank" "PLAYER",
                    "PRED",
                ]
            )
            building_performances_succeeded = False
        try:
            shap_values = build_shap_values()
            building_shap_values_succeeded = True
        except (OSError, Exception) as e:
            logger.error(f"Failed to build shap values {e}", exc_info=False)
            shap_values = pandas.DataFrame(
                columns=[
                    "player",
                    "2P_per_game",
                    "FGA_per_100poss",
                    "TRB_per_game",
                    "FTA_per_game",
                    "FG_per_36min",
                    "PS/G",
                    "AST_per_36min",
                    "FG_per_100poss",
                    "PER_advanced",
                    "PA/G",
                    "FT_per_36min",
                    "PTS_per_36min",
                    "AST%_advanced",
                    "EFG%_per_game",
                    "2P%",
                    "AST_per_game",
                    "MP",
                    "ORB_per_36min",
                    "PL",
                    "DRTG_per_100poss",
                    "DRB_per_game",
                    "WS/48_advanced",
                    "DBPM_advanced",
                    "BLK_per_100poss",
                    "2P_per_100poss",
                    "CONF_RANK",
                    "FG_per_game",
                    "FG%",
                    "FTR_advanced",
                    "ORB_per_game",
                    "2PA_per_game",
                    "2PA_per_36min",
                    "PW",
                    "GB",
                    "OWS_advanced",
                    "FGA_per_game",
                    "BLK_per_game",
                    "STL_per_game",
                    "FTA_per_36min",
                    "G",
                    "BPM_advanced",
                    "VORP_advanced",
                    "OBPM_advanced",
                    "W",
                    "W/L%",
                    "WS_advanced",
                    "PF_per_36min",
                    "PF_per_100poss",
                    "FT_per_game",
                    "FT%",
                    "PTS_per_game",
                    "L",
                    "TRB_per_100poss",
                    "TS%_advanced",
                    "STL_per_100poss",
                    "DWS_advanced",
                    "FGA_per_36min",
                    "PTS_per_100poss",
                    "DRB_per_36min",
                    "POS_C",
                    "POS_PF",
                    "POS_PG",
                    "POS_SF",
                    "POS_SG",
                    "CONF_EASTERN_CONF",
                    "CONF_WESTERN_CONF",
                ]
            )
            building_shap_values_succeeded = False
        current_season = (
            datetime.now().year + 1 if datetime.now().month > 9 else datetime.now().year
        )

        st.markdown(
            f"""
        *Predicting the NBA Most Valuable Player for the {current_season-1}-{str(current_season)[-2:]} season using machine learning.*
        *Last update : {last_update}.*
        """
        )

        navigation_page = st.sidebar.radio(
            "Navigate to", [PAGE_PREDICTIONS, PAGE_EXPLICABILITY, PAGE_PERFORMANCE]
        )
        st.sidebar.markdown(conf.web.sidebar_top_text)
        st.sidebar.markdown(conf.web.sidebar_bottom_text)

        initial_columns = list(predictions.columns)

        st.header(str(navigation_page))

        if navigation_page == PAGE_PREDICTIONS:
            if building_predictions_succeeded:
                col1, col2 = st.columns(2)
                col1.subheader("Predicted top 3")
                col2.subheader("Prediction parameters")
                confidence_mode = col2.radio(
                    "Method used to estimate MVP probability",
                    [CONFIDENCE_MODE_SHARE, CONFIDENCE_MODE_SOFTMAX],
                )
                compute_probs_based_on_top_n = col2.slider(
                    "Number of players used to estimate probability",
                    min_value=3,
                    max_value=10,
                    value=10,
                    step=1,
                    format="%d players",
                )
                if confidence_mode == CONFIDENCE_MODE_SOFTMAX:
                    predictions.loc[
                        predictions.PRED_RANK <= compute_probs_based_on_top_n,
                        "MVP probability",
                    ] = (
                        evaluate.softmax(
                            predictions[
                                predictions.PRED_RANK <= compute_probs_based_on_top_n
                            ]["PRED"]
                        )
                        * 100
                    )
                else:
                    predictions.loc[
                        predictions.PRED_RANK <= compute_probs_based_on_top_n,
                        "MVP probability",
                    ] = (
                        evaluate.share(
                            predictions[
                                predictions.PRED_RANK <= compute_probs_based_on_top_n
                            ]["PRED"]
                        )
                        * 100
                    )
                predictions.loc[
                    predictions.PRED_RANK > compute_probs_based_on_top_n,
                    "MVP probability",
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
                # show_top_n = min([compute_probs_based_on_top_n, 10])

                st.subheader(f"Predicted top {show_top_n}")
                col1 = st.container()
                predictions["player"] = predictions.index
                predictions["chance"] = predictions["MVP probability"].str[:-1]
                predictions["chance"] = pandas.to_numeric(predictions["chance"])
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
                                "title": " MVP probability (%)",
                            },
                            "y": {
                                "field": "player",
                                "type": "nominal",
                                "title": None,
                                "sort": "-x",
                            },
                            "color": {
                                "field": "chance",
                                "type": "quantitative",
                                "title": None,
                                "legend": None,
                                "scale": {"scheme": "purplebluegreen"},
                            },
                        },
                    },
                    height=350,
                    use_container_width=True,
                )

            else:
                st.warning(
                    "This section is unavailable because loading predictions file failed."
                )

            st.subheader("Predictions history")

            if building_history_succeeded:
                col1, col2, col3 = st.columns([2, 3, 3])
                keep_top_n = col2.slider(
                    "Number of players to show",
                    min_value=3,
                    max_value=compute_probs_based_on_top_n,
                    value=5,
                    step=1,
                    format="%d players",
                )
                variable_to_draw_dict = {
                    "MVP probability (%)": "chance",
                    "Predicted MVP share": "prediction",
                }
                variable_to_draw = col1.radio(
                    "Variable to draw", list(variable_to_draw_dict.keys())
                )
                slider_min_value = max(int(history.days_ago.min()), 3)
                slider_max_value = int(history.days_ago.max())
                logger.debug("Slider minimum value: %s", slider_min_value)
                logger.debug("Slider maximum value: %s", slider_max_value)
                if slider_min_value < slider_max_value:
                    num_past_days = col3.slider(
                        "Show history for last",
                        min_value=slider_min_value,
                        max_value=slider_max_value,
                        value=min(int(history.days_ago.max()), 30),
                        step=1,
                        format="%d days",
                    )
                else:
                    logger.warning("Could not build history range slider")
                    logger.warning("Most likely there is not enough history")
                    num_past_days = 1

                prepared_history = prepare_history(
                    history,
                    keep_top_n,
                    confidence_mode,
                    compute_probs_based_on_top_n,
                    keep_last_days=num_past_days,
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
                                "title": "Date",
                                "axis": {
                                    "format": "%b %d",
                                    "labelAngle": 0,
                                },
                            },
                            "y": {
                                "field": variable_to_draw_dict[variable_to_draw],
                                "type": "quantitative",
                                "title": variable_to_draw,
                                "axis": {
                                    "format": ".0f",
                                },
                            },
                            "color": {
                                "field": "player",
                                "type": "nominal",
                                "scale": {"scheme": "category20"},
                                "legend": {
                                    "orient": "top",  # "top-left",
                                    "fillColor": "#525050",
                                },
                            },
                        },
                    },
                    height=400,
                    use_container_width=True,
                )

            else:
                st.warning(
                    "This section is unavailable because loading history file failed."
                )

        elif navigation_page == PAGE_PERFORMANCE:
            if building_performances_succeeded:
                col1, col2 = st.columns(2)
                # col1
                percentage = mvp_found_pct(performances)
                num_test_seasons = len(performances)
                avg_real_rank = avg_real_mvp_rank(performances)
                avg_pred_rank = avg_pred_mvp_rank(performances)

                col1.markdown(
                    f"##### All {num_test_seasons} seasons ({performances.index.min()}-{performances.index.max()})"
                )
                col1.markdown(
                    f"""
                - **{percentage}** of MVPs correctly found
                - The true MVP is ranked **{avg_real_rank}** by the model in average
                - The true rank of the predicted MVP is **{avg_pred_rank}** in average
                """
                )
                # col2
                performances_last10 = performances.head(10)
                percentage = mvp_found_pct(performances_last10)
                num_test_seasons = len(performances_last10)
                avg_real_rank = avg_real_mvp_rank(performances_last10)
                avg_pred_rank = avg_pred_mvp_rank(performances_last10)
                col2.markdown(
                    f"##### Last {num_test_seasons} seasons ({performances_last10.index.min()}-{performances_last10.index.max()})"
                )
                col2.markdown(
                    f"""
                - **{percentage}** of MVPs correctly found
                - The true MVP is ranked **{avg_real_rank}** by the model in average
                - The true rank of the predicted MVP is **{avg_pred_rank}** in average
                """
                )

                st.dataframe(data=performances, width=None, height=None)
                st.markdown(
                    """
                Predictions of the model are made on the unseen season using holdout method.
                Players with no MVP vote are considered as ranked 10th for simplification.
                """
                )

            else:
                st.warning(
                    "This page is unavailable because loading perfomances file failed."
                )

        elif navigation_page == PAGE_EXPLICABILITY:
            if building_shap_values_succeeded:
                # Remove binary features - should no be trusted for SHAP
                shap_values = shap_values[
                    [
                        f
                        for f in shap_values.columns
                        if "ERN_CONF" not in f and "POS_" not in f
                    ]
                ]
                shap_values = shap_values.rename(columns=format_feature_name)

                st.subheader("Local explanation")
                st.markdown(
                    "To understand which stats have the strongest impact on the model prediction for the MVP share one player."
                )
                selected_player = st.selectbox(
                    "Select a player", predictions.index.to_list()[:10]
                )
                a = shap_values.loc[selected_player, :]
                features_positive_impact = a[a > 0.0]
                features_negative_impact = a[a < 0.0]

                num_stats = 3
                top_features_positive_impact = (
                    features_positive_impact.sort_values(ascending=False)
                    .index[:num_stats]
                    .to_list()
                )
                top_features_positive_impact_values = (
                    features_positive_impact.sort_values(ascending=False).values[
                        :num_stats
                    ]
                )
                top_features_negative_impact = (
                    features_negative_impact.sort_values(ascending=True)
                    .index[:num_stats]
                    .to_list()
                )
                top_features_negative_impact_values = (
                    features_negative_impact.sort_values(ascending=True).values[
                        :num_stats
                    ]
                )

                st.markdown(
                    "👍 Stats with the strongest **positive impact** on the model prediction for this player:"
                )
                for i, col in enumerate(st.columns(num_stats)):
                    col.success(
                        f"""
                        **{(top_features_positive_impact[i])}**   
                        *+{round(top_features_positive_impact_values[i], 2)} MVP share*
                        """
                    )
                st.markdown(
                    "👎 Stats with the strongest **negative impact** on the model prediction for this player:"
                )
                for i, col in enumerate(st.columns(num_stats)):
                    col.error(
                        f"""
                        **{top_features_negative_impact[i]}**    
                        *{round(top_features_negative_impact_values[i], 2)} MVP share*
                        """
                    )

                st.subheader("Global explanation")
                st.markdown(
                    "To understand which stats have an impact on the model prediction for the MVP share of the top-10 players."
                )

                vals = numpy.array(shap_values.values).mean(0)
                vals_abs = numpy.abs(shap_values.values).mean(0)
                shap_importance = pandas.DataFrame(
                    list(zip(shap_values.columns, vals, vals_abs)),
                    columns=[
                        "col_name",
                        "feature_importance_vals",
                        "abs_feature_importance_vals",
                    ],
                )
                keep_to_n_features: int = st.slider(
                    "Number of stats to show",
                    min_value=10,
                    max_value=len(shap_importance),
                    value=10,
                    step=1,
                    format="%d stats",
                    label_visibility="hidden",
                )
                shap_importance = shap_importance.sort_values(
                    by=["abs_feature_importance_vals"], ascending=False
                ).head(keep_to_n_features)
                chart_height = 28 * keep_to_n_features
                col1, col2 = st.columns([10, 9])
                shap_importance = shap_importance.sort_values(
                    by=["feature_importance_vals"], ascending=True
                )
                col1.markdown("**Average impact on the predicted MVP share**")
                col1.vega_lite_chart(
                    shap_importance,
                    {
                        "mark": {
                            "type": "bar",
                            "point": True,
                            "tooltip": True,
                        },
                        "title": {"text": None},
                        "encoding": {
                            "x": {
                                "field": "feature_importance_vals",
                                "type": "quantitative",
                                "title": "Average impact (MVP share)",
                            },
                            "y": {
                                "field": "col_name",
                                "type": "nominal",
                                "title": None,
                                "sort": "-x",
                            },
                            "color": {
                                "field": "feature_importance_vals",
                                "type": "quantitative",
                                "title": None,
                                "legend": None,
                                "scale": {"scheme": "redyellowgreen"},
                            },
                        },
                    },
                    height=chart_height,
                    use_container_width=True,
                )

                shap_importance = shap_importance.sort_values(
                    by=["abs_feature_importance_vals"], ascending=True
                )
                col2.markdown("**Average absolute impact (positive or negative)**")
                col2.vega_lite_chart(
                    shap_importance,
                    {
                        "mark": {
                            "type": "bar",
                            "point": True,
                            "tooltip": True,
                        },
                        "title": {"text": None},
                        "encoding": {
                            "x": {
                                "field": "abs_feature_importance_vals",
                                "type": "quantitative",
                                "title": "Average absolute impact (MVP share)",
                            },
                            "y": {
                                "field": "col_name",
                                "type": "nominal",
                                "title": None,
                                "sort": "-x",
                                "axis": {"labelLimit": -1},
                            },
                            "color": {
                                "field": "abs_feature_importance_vals",
                                "type": "quantitative",
                                "title": None,
                                "legend": None,
                                "scale": {"scheme": "purplebluegreen"},
                            },
                        },
                    },
                    height=chart_height,
                    use_container_width=True,
                )
            else:
                st.warning(
                    "This page is unavailable because loading explainability file failed."
                )
        else:
            st.error("Unknown page selected.")
    else:
        st.info(conf.web.disabled_web_text)
