import pandas

from nba_mvp_predictor import conf
from nba_mvp_predictor import scrappers


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
