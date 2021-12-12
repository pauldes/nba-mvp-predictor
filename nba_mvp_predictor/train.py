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
    player_stats = load.load_player_stats()
    print(player_stats.sample(3))
