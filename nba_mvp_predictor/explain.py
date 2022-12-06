import joblib

import shap
import pandas
import numpy
from matplotlib import pyplot

from nba_mvp_predictor import conf, logger
from nba_mvp_predictor import load, download


def explain_model():
    """Explain model predictions."""
    model = load.load_model()
    model_input = load.load_model_input()
    predictions = load.load_predictions()
    predictions = predictions.sort_values(by="PRED_RANK", ascending=True)
    player_season_team_list = predictions.index.to_list()
    # Analyze SHAP values on 10 top players
    sample_size = 10
    logger.debug(f"SHAP values will be computed for : {sample_size} top players")
    sample = model_input[model_input.index.isin(player_season_team_list[:sample_size])]
    # Compare to a population of 100 players
    population_size = 100
    logger.debug(f"Number of players in predictions : {len(player_season_team_list)}")
    # Old method : not ideal because only players with positive shares are kept
    population = model_input[
        model_input.index.isin(player_season_team_list[:population_size])
    ]
    # New method : Sample 100 players randomly.
    population = model_input.sample(population_size)
    logger.debug(f"Population size for SHAP : {population_size}")

    explainer = shap.Explainer(model.predict, population, algorithm="auto")
    shap_values = explainer(sample)
    top10_shap_values = {}
    sample["player"] = sample.index
    sample["player"] = sample["player"].map(predictions["PLAYER"])
    sample = sample.reset_index(drop=True)
    feature_names = shap_values.feature_names
    shap_df = pandas.DataFrame(
        shap_values.values, columns=feature_names, index=sample.player
    )
    shap_df.to_csv(
        conf.data.shap_values.path,
        sep=conf.data.shap_values.sep,
        encoding=conf.data.shap_values.encoding,
        compression=conf.data.shap_values.compression,
        index=True,
    )
