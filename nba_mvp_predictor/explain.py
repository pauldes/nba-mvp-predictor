import joblib

import shap
import pandas
import numpy
from matplotlib import pyplot

from nba_mvp_predictor import conf, logger
from nba_mvp_predictor import load, download, predict, train

def get_model_input_alternative():
    # Data needs to be downloaded prior
    train.make_bronze_data()
    train.make_silver_data()
    Xs = []
    for season in [1980, 1990, 2000, 2010, 2020]:
        X = predict.prepare_data_for_prediction_from_silver(season)
        logger.debug(f"Season {season} - {X.shape} entries")
        Xs.append(X)
    X = pandas.concat(Xs, sort=False)
    logger.debug(f"All seasons - {X.shape} entries")
    return X

def explain_model():
    """Explain model predictions."""
    model = load.load_model()
    model_input_old_seasons = get_model_input_alternative()
    model_input_current_season = load.load_model_input()
    model_input = pandas.concat(
        [model_input_old_seasons, model_input_current_season], sort=False
    )
    predictions = load.load_predictions()
    predictions = predictions.sort_values(by="PRED_RANK", ascending=True)
    player_season_team_list = predictions.index.to_list()
    # Analyze SHAP values on 10 top players
    sample_size = 10
    logger.debug(f"SHAP values will be computed for : {sample_size} top players")
    sample = model_input[
        model_input.index.isin(player_season_team_list[:sample_size])
    ]
    # Compare to a population of 500 players
    population_size = 500
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
