import joblib

import pandas

from nba_mvp_predictor import conf
from nba_mvp_predictor import scrappers


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
