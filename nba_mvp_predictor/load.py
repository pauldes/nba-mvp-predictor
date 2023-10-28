import json

import joblib
import pandas

from nba_mvp_predictor import conf, scrappers


def load_model():
    """Load the model.

    Returns:
        sklearn.base.BaseEstimator: The model to use for prediction
    """
    return joblib.load(conf.data.model.path)


def load_player_stats(nrows: int = None):
    return pandas.read_csv(
        conf.data.player_stats.path,
        sep=conf.data.player_stats.sep,
        encoding=conf.data.player_stats.encoding,
        compression=conf.data.player_stats.compression,
        index_col=0,
        nrows=nrows,
    )


def load_mvp_votes(nrows: int = None):
    return pandas.read_csv(
        conf.data.mvp_votes.path,
        sep=conf.data.mvp_votes.sep,
        encoding=conf.data.mvp_votes.encoding,
        compression=conf.data.mvp_votes.compression,
        index_col=0,
        nrows=nrows,
    )


def load_team_standings(nrows: int = None):
    return pandas.read_csv(
        conf.data.team_standings.path,
        sep=conf.data.team_standings.sep,
        encoding=conf.data.team_standings.encoding,
        compression=conf.data.team_standings.compression,
        index_col=0,
        nrows=nrows,
    )


def load_bronze_data(nrows: int = None):
    return pandas.read_csv(
        conf.data.bronze.path,
        sep=conf.data.bronze.sep,
        encoding=conf.data.bronze.encoding,
        compression=conf.data.bronze.compression,
        index_col=0,
        nrows=nrows,
    )


def load_silver_data(nrows: int = None):
    return pandas.read_csv(
        conf.data.silver.path,
        sep=conf.data.silver.sep,
        encoding=conf.data.silver.encoding,
        compression=conf.data.silver.compression,
        index_col=0,
        nrows=nrows,
    )


def load_gold_data(nrows: int = None):
    return pandas.read_csv(
        conf.data.gold.path,
        sep=conf.data.gold.sep,
        encoding=conf.data.gold.encoding,
        compression=conf.data.gold.compression,
        index_col=0,
        nrows=nrows,
    )


def load_predictions(nrows: int = None):
    return pandas.read_csv(
        conf.data.predictions.path,
        sep=conf.data.predictions.sep,
        encoding=conf.data.predictions.encoding,
        compression=conf.data.predictions.compression,
        index_col=0,
        nrows=nrows,
        dtype={},
    )


def load_history(nrows: int = None):
    return pandas.read_csv(
        conf.data.history.path,
        sep=conf.data.history.sep,
        encoding=conf.data.history.encoding,
        compression=conf.data.history.compression,
        index_col=False,
        nrows=nrows,
        dtype={},
    )


def load_features():
    with open(
        conf.data.features.path, encoding=conf.data.features.encoding
    ) as json_file:
        features_dict = json.load(json_file)
    return features_dict


def load_model_input(nrows: int = None):
    return pandas.read_csv(
        conf.data.model_input.path,
        sep=conf.data.model_input.sep,
        encoding=conf.data.model_input.encoding,
        compression=conf.data.model_input.compression,
        index_col=0,
        nrows=nrows,
        dtype={},
    )


def load_shap_values(nrows: int = None):
    return pandas.read_csv(
        conf.data.shap_values.path,
        sep=conf.data.shap_values.sep,
        encoding=conf.data.shap_values.encoding,
        compression=conf.data.shap_values.compression,
        index_col=0,
        nrows=nrows,
        dtype={},
    )
