from datetime import datetime

from nba_mvp_predictor import conf, logger
from nba_mvp_predictor import load, preprocess

def load_model_make_predictions():
    model = load.load_model()
    data = load.load_silver_data()
    data = data.fillna(0.0)
    current_season = datetime.now().year + 1 if datetime.now().month>9 else datetime.now().year
    logger.debug(f"Current season : {current_season}")
    data = data[data.SEASON==current_season]

    # TODO get automatically from training step.. or keep all
    cat = ["POS", "CONF"]
    # TODO get automatically from training step.. or keep all
    num = [
        'AGE', 'G', 'MP', 'FG_per_game', 'FGA_per_game', 'FG%', '3P%', '2P_per_game', '2PA_per_game', '2P%', 'EFG%_per_game', 'FT_per_game', 'FTA_per_game', 'FT%', 'ORB_per_game', 'DRB_per_game', 'TRB_per_game', 'AST_per_game', 'STL_per_game', 'BLK_per_game', 'PF_per_game', 'PTS_per_game', 'FG_per_36min', 'FGA_per_36min', '3P_per_36min', 'FT_per_36min', 'ORB_per_36min', 'DRB_per_36min', 'TRB_per_36min', 'AST_per_36min', 'BLK_per_36min', 'PF_per_36min', 'PTS_per_36min', 'FG_per_100poss', 'FGA_per_100poss', '2P_per_100poss', '2PA_per_100poss', 'FTA_per_100poss', 'STL_per_100poss', 'PF_per_100poss', 'PTS_per_100poss', 'DRTG_per_100poss', 'PER_advanced', 'TS%_advanced', '3PAR_advanced', 
        'FTR_advanced', 'AST%_advanced', 'BLK%_advanced', 'OWS_advanced', 'DWS_advanced', 'WS_advanced', 'WS/48_advanced', 'OBPM_advanced', 'DBPM_advanced', 'BPM_advanced', 'VORP_advanced', 'GS', 'TOV_per_game', 'TOV_per_100poss', 'ORTG_per_100poss', 'TOV%_advanced', 'USG%_advanced', 'W', 'L', 'W/L%', 'GB', 'PW', 'PL', 'PS/G', 'PA/G', 'CONF_RANK'
    ]
    # TODO get automatically from training step
    features = [
       'BLK%_advanced', 'AGE', 'PTS_per_100poss', 'PA/G', 'CONF_RANK', 'MP', 'DBPM_advanced', '2PA_per_game', 'DRB_per_36min', 'PF_per_100poss', 'OBPM_advanced', 'FGA_per_100poss', 'STL_per_game', '3PAR_advanced', 'L', '3P%', 'BLK_per_game', 'PS/G', '2P%', 'FG_per_game', 'PTS_per_game', 'GB', 'FT%', 'TS%_advanced', 'FG_per_100poss', 'PTS_per_36min', 'FG%', 'BPM_advanced', 'FTA_per_game', 'ORB_per_36min', 'PF_per_game', '2PA_per_100poss', 'PW', 'TRB_per_game', 'TRB_per_36min', 'FGA_per_36min', 'OWS_advanced', 'DRB_per_game', 'W', 'VORP_advanced', 'G', 'FG_per_36min', 'FT_per_36min', 'WS/48_advanced', 'FGA_per_game', '2P_per_100poss', 'WS_advanced', 'PF_per_36min', '3P_per_36min', 
       'DRTG_per_100poss', 'DWS_advanced', 'STL_per_100poss', 'ORB_per_game', 'W/L%', 'AST_per_game', 'EFG%_per_game', 'FT_per_game', 'BLK_per_36min', 'AST%_advanced', 'PL', '2P_per_game', 'AST_per_36min', 'FTA_per_100poss', 'PER_advanced', 'FTR_advanced', 'POS_C', 'POS_PF', 'POS_PG', 'POS_SF', 'POS_SG', 'CONF_EASTERN_CONF', 'CONF_WESTERN_CONF'
    ]
    min_max_scaling = False
    data_processed_features_only, _ = preprocess.scale_per_value_of(
        data, cat, num, data["SEASON"], min_max_scaler=min_max_scaling
    )
    X = data_processed_features_only[features]
    predictions = model.predict(X)
    data.loc[:, "PRED"] = predictions
    data.loc[:, "PRED"] = data["PRED"].clip(lower=0.0)
    data.loc[:, "PRED_RANK"] = data["PRED"].rank(ascending=False)
    data = data.sort_values(by="PRED", ascending=False).head(50)
    data.to_csv(
        conf.data.predictions.path,
        sep=conf.data.predictions.sep,
        encoding=conf.data.predictions.encoding,
        compression=conf.data.predictions.compression,
        index=True,
    )

def make_predictions():
    try:
        load_model_make_predictions()
    except Exception as e:
        logger.error(f"Predicting failed : {e}")