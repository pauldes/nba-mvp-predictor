from nba_mvp_predictor import conf, logger
from nba_mvp_predictor import load

def load_model_make_predictions():
    model = load.load_model()

def make_predictions():
    try:
        load_model_make_predictions()
    except Exception as e:
        logger.error(f"Predicting failed : {e}")