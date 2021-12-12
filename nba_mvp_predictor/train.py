from sklearn import (
    dummy,
    tree,
    model_selection,
    metrics,
    preprocessing,
    linear_model,
    ensemble,
    neural_network,
)

from nba_mvp_predictor import conf, logger
from nba_mvp_predictor import load

def consolidate_data():
    player_stats = load.load_player_stats()
    mvp_votes = load.load_mvp_votes()
    team_standings = load.load_team_standings()
    if mvp_votes.duplicated(subset=["PLAYER", "TEAM", "SEASON"]).sum() > 0:
        logger.warning("Duplicated rows in MVP votes!")
    bronze = player_stats.reset_index(
        drop=False
        ).merge(
            mvp_votes, how="left", on=["PLAYER", "TEAM", "SEASON"]
        ).set_index(
            player_stats.index.name
        )
    if team_standings.duplicated(subset=["TEAM", "SEASON"]).sum() > 0:
        logger.warning("Duplicated rows in team standings!")
    bronze = bronze.reset_index(
            drop=False
        ).merge(
            team_standings, how="inner", on=["TEAM", "SEASON"]
        ).set_index(
            bronze.index.name
        )
    for col in ["MVP_WINNER", "MVP_PODIUM", "MVP_CANDIDATE"]:
        bronze[col] = bronze[col].fillna(False)
    for col in ["MVP_VOTES_SHARE"]:
        bronze[col] = bronze[col].fillna(0.0)
    logger.info(f'MVPs found in data : {bronze[bronze["MVP_WINNER"] == True]["SEASON"].nunique()}')
    bronze.to_csv(
        conf.data.bronze.path,
        sep=conf.data.bronze.sep,
        encoding=conf.data.bronze.encoding,
        compression=conf.data.bronze.compression,
        index=True,
    )

def preprocess_data():
    nrows = 10


def train_model():
    consolidate_data()