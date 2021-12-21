import streamlit as st
import pandas

from nba_mvp_predictor import conf
from nba_mvp_predictor import load, evaluate

# Constants
PAGE_PREDICTIONS = "Current year predictions"
PAGE_PERFORMANCE = "Model performance analysis"
CONFIDENCE_MODE_SOFTMAX = "Softmax-based"
CONFIDENCE_MODE_SHARE = "Share-based"

def build_history():
    history = load.load_history()
    history = history.rename(columns={"DATE":"date", "PLAYER":"player", "PRED":"prediction"})
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

    predictions = load.load_predictions()
    predictions = predictions.set_index("PLAYER", drop=True)
    initial_columns = list(predictions.columns)

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
        max_value=50,
        value=10,
        step=5,
    )
    if confidence_mode == CONFIDENCE_MODE_SOFTMAX:
        predictions.loc[
            predictions.PRED_RANK <= compute_probs_based_on_top_n, "MVP probability"
        ] = (
            evaluate.softmax(
                predictions[predictions.PRED_RANK <= compute_probs_based_on_top_n]["PRED"]
            )
            * 100
        )
    else:
        predictions.loc[
            predictions.PRED_RANK <= compute_probs_based_on_top_n, "MVP probability"
        ] = (
            evaluate.share(
                predictions[predictions.PRED_RANK <= compute_probs_based_on_top_n]["PRED"]
            )
            * 100
        )
    predictions.loc[
        predictions.PRED_RANK > compute_probs_based_on_top_n, "MVP probability"
    ] = 0.0
    predictions["MVP probability"] = predictions["MVP probability"].map("{:,.2f}%".format)
    predictions["MVP rank"] = predictions["PRED_RANK"]
    show_columns = ["MVP probability", "MVP rank"] + initial_columns[:]
    predictions = predictions[show_columns]

    top_3 = predictions["MVP probability"].head(3).to_dict()
    emojis = ["🥇", "🥈", "🥉"]

    for n, player_name in enumerate(top_3):
        title_level = "###" + n * "#"
        col1.markdown(
            f"""
        #### {emojis[n]} **{player_name}**

        *{top_3[player_name]} chance to win MVP*
        """
        )

    st.subheader(f"Predicted top {compute_probs_based_on_top_n}")
    st.dataframe(
        data=predictions.head(compute_probs_based_on_top_n), width=None, height=None
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