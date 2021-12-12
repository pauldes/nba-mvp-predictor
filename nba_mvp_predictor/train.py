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

from nba_mvp_predictor import conf
from nba_mvp_predictor import load


def train_model():
    nrows = 10
    player_stats = load.load_player_stats(nrows=nrows)
    print(player_stats.sample(3))
    mvp_votes = load.load_mvp_votes(nrows=nrows)
    print(mvp_votes.sample(3))
    team_standings = load.load_team_standings(nrows=nrows)
    print(team_standings.sample(3))
