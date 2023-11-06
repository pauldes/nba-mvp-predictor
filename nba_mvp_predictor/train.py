import json
from datetime import datetime

import joblib
import numpy
import pandas
from sklearn import base, metrics, model_selection

from nba_mvp_predictor import analyze, conf, load, logger, model, preprocess

_MIN_TARGET_CORRELATION = 0.05
_MAX_FEATURES_CORRELATION = 0.95


def make_bronze_data():
    """Make bronze training data from raw downloaded data."""
    player_stats = load.load_player_stats()
    mvp_votes = load.load_mvp_votes()
    team_standings = load.load_team_standings()
    if mvp_votes.duplicated(subset=["PLAYER", "TEAM", "SEASON"]).sum() > 0:
        logger.warning("Duplicated rows in MVP votes!")
    bronze = (
        player_stats.reset_index(drop=False)
        .merge(mvp_votes, how="left", on=["PLAYER", "TEAM", "SEASON"])
        .set_index(player_stats.index.name)
    )
    if team_standings.duplicated(subset=["TEAM", "SEASON"]).sum() > 0:
        logger.warning("Duplicated rows in team standings!")
    bronze = (
        bronze.reset_index(drop=False)
        .merge(team_standings, how="inner", on=["TEAM", "SEASON"])
        .set_index(bronze.index.name)
    )
    # Add a feature: PREVIOUS_SEASON_MVP_WINNER
    previous_season_winners = bronze[bronze["MVP_WINNER"] == True][
        ["PLAYER", "TEAM", "SEASON"]
    ]
    previous_season_winners["SEASON"] = previous_season_winners["SEASON"] + 1
    previous_season_winners["PREVIOUS_SEASON_MVP_WINNER"] = True
    # Add a feature: PREVIOUS_SEASON_MVP_PODIUM_NOT_WINNER
    previous_season_podium_not_winners = bronze[
        (bronze["MVP_WINNER"] == False) & (bronze["MVP_PODIUM"] == True)
    ][["PLAYER", "TEAM", "SEASON"]]
    previous_season_podium_not_winners["SEASON"] = (
        previous_season_podium_not_winners["SEASON"] + 1
    )
    previous_season_podium_not_winners["PREVIOUS_SEASON_MVP_PODIUM_NOT_WINNER"] = True
    # Merge these two features
    bronze = (
        bronze.reset_index(drop=False)
        .merge(
            previous_season_podium_not_winners,
            how="left",
            on=["PLAYER", "TEAM", "SEASON"],
        )
        .set_index(bronze.index.name)
        .reset_index(drop=False)
        .merge(previous_season_winners, how="left", on=["PLAYER", "TEAM", "SEASON"])
        .set_index(bronze.index.name)
    )

    for col in [
        "MVP_WINNER",
        "MVP_PODIUM",
        "MVP_CANDIDATE",
        "PREVIOUS_SEASON_MVP_WINNER",
        "PREVIOUS_SEASON_MVP_PODIUM_NOT_WINNER",
    ]:
        bronze[col] = bronze[col].fillna(False).astype(bool)
    for col in ["MVP_VOTES_SHARE"]:
        bronze[col] = bronze[col].fillna(0.0)
    logger.info(
        f'MVPs found in data : {bronze[bronze["MVP_WINNER"] == True]["SEASON"].nunique()}'
    )
    bronze.to_csv(
        conf.data.bronze.path,
        sep=conf.data.bronze.sep,
        encoding=conf.data.bronze.encoding,
        compression=conf.data.bronze.compression,
        index=True,
    )


def make_silver_data():
    """Make silver training data from bronze data."""
    data = load.load_bronze_data()
    logger.debug(
        f"Before filters: {len(data)} players - {len(data[data.MVP_CANDIDATE])} MVP candidates - {len(data[data.MVP_WINNER])} winners"
    )
    data_copy = data.copy()
    # Apply filters
    # 50% games played
    # 28 minutes per game
    # 2 FG attemptes
    # Team ranked 12th in conference at least
    for season in data.SEASON.unique():
        max_g = data[data.SEASON == season]["G"].max()
        treshold = 0.5 * max_g
        data = data[
            (data.SEASON != season) | ((data.SEASON == season) & (data.G >= treshold))
        ]
    data = data[data["FGA_per_game"] >= 2]
    data = data[data["CONF_RANK"] <= 12]
    data = data[data["MP"] >= 28.0]

    removed_players = data_copy.loc[~data_copy.index.isin(data.index)]
    removed_mvp_candidates = removed_players[removed_players.MVP_CANDIDATE]
    logger.debug(f"{len(removed_mvp_candidates)} MVP candidates removed due to filters")
    if len(removed_mvp_candidates) > 0:
        logger.info(
            "Removed candidates: %s", ", ".join(removed_mvp_candidates.index.unique())
        )
    logger.debug(
        f"After filters: {len(data)} players - {len(data[data.MVP_CANDIDATE])} MVP candidates - {len(data[data.MVP_WINNER])} winners"
    )
    data.to_csv(
        conf.data.silver.path,
        sep=conf.data.silver.sep,
        encoding=conf.data.silver.encoding,
        compression=conf.data.silver.compression,
        index=True,
    )


def make_gold_data_and_train_model():
    """Make gold training data from silver data"""
    data = load.load_silver_data()
    not_features = [
        "PLAYER",
        "MVP_VOTES_SHARE",
        "MVP_WINNER",
        "MVP_PODIUM",
        "MVP_CANDIDATE",
        "TEAM",
        "SEASON",
    ]  # Conf and season maybe should be used
    features = [col for col in data.columns if col not in not_features]
    num_features = list(preprocess.get_numerical_columns(data[features]))
    cat_features = list(preprocess.get_categorical_columns(data[features]))
    logger.debug(f"Not features ( {len(not_features)}): {', '.join(not_features)}")
    logger.debug(
        f"Numerical features ( {len(num_features)}): {', '.join(num_features)}"
    )
    logger.debug(
        f"Categorical features ( {len(cat_features)}): {', '.join(cat_features)}"
    )

    # Maximum two players per team ?
    # Include team ? Include already_won, won_last_year..? Include won_over_other_contenders.. ? Include has_top_performance ?

    corr_treshold = _MAX_FEATURES_CORRELATION
    num_features = list(
        analyze.get_columns_with_inter_correlations_under(
            data[num_features], corr_treshold
        )
    )
    data = data[num_features + cat_features + not_features]
    # Add MVP rank
    for season in data.SEASON.unique():
        data.loc[data.SEASON == season, "MVP_RANK"] = data[data.SEASON == season][
            "MVP_VOTES_SHARE"
        ].rank(ascending=False, method="min")
    ranks_reference = data.MVP_RANK.copy()
    not_features.append("MVP_RANK")

    target = "MVP_VOTES_SHARE"

    selected_num_features = list(num_features)
    selected_cat_features = list(cat_features)
    min_max_scaling = False  # else it's stdev

    if min_max_scaling:
        standardized_type = "min_max"
    else:
        standardized_type = "std"

    data_processed_features_only, data_raw = preprocess.scale_per_value_of(
        data,
        selected_cat_features,
        selected_num_features,
        data["SEASON"],
        min_max_scaler=min_max_scaling,
    )
    selected_cat_features_numerized = [
        f
        for f in data_processed_features_only.columns
        if f not in selected_num_features
    ]
    data_not_features = data[not_features]

    data_processed = pandas.concat(
        [data_processed_features_only, data_not_features], axis=1
    )

    data_processed.to_csv(
        conf.data.gold.path,
        sep=conf.data.gold.sep,
        encoding=conf.data.gold.encoding,
        compression=conf.data.gold.compression,
        index=True,
    )

    data = load.load_gold_data()

    current_season = (
        datetime.now().year + 1 if datetime.now().month > 9 else datetime.now().year
    )
    logger.debug(f"Current season : {current_season}")
    data = data[data.SEASON < current_season]
    percent_test_seasons = 0.2
    num_test_seasons = int(data.SEASON.nunique() * percent_test_seasons)
    test_seasons = sorted(data.SEASON.unique())[-num_test_seasons:]
    trainval_seasons = sorted(data.SEASON.unique())[:-num_test_seasons]
    logger.debug(f"Test seasons : {test_seasons[0]} to {test_seasons[-1]}")
    logger.debug(f"Trainval seasons : {trainval_seasons[0]} to {trainval_seasons[-1]}")
    data_test = data[data.SEASON.isin(test_seasons)]
    data_trainval = data[data.SEASON.isin(trainval_seasons)]
    data_all = data.copy()

    # n_features = 50
    # treshold = None
    n_features = None
    treshold = _MIN_TARGET_CORRELATION

    data_for_corr_analysis = data_trainval[
        selected_num_features
    ]  # we make the choice of not looking into numerized cat features

    method = "pearson"
    top_corr_pearson = filter_by_correlation_with_target(
        pandas.concat([data_for_corr_analysis, data_trainval[target]], axis=1),
        target,
        method=method,
        n_features=n_features,
        treshold=treshold,
    )
    method = "kendall"
    top_corr_kendall = filter_by_correlation_with_target(
        pandas.concat([data_for_corr_analysis, data_trainval[target]], axis=1),
        target,
        method=method,
        n_features=n_features,
        treshold=treshold,
    )
    method = "spearman"
    top_corr_spearman = filter_by_correlation_with_target(
        pandas.concat([data_for_corr_analysis, data_trainval[target]], axis=1),
        target,
        method=method,
        n_features=n_features,
        treshold=treshold,
    )

    selected_features_pearson = top_corr_pearson.index.tolist()
    selected_features_kendall = top_corr_kendall.index.tolist()
    selected_features_spearman = top_corr_spearman.index.tolist()

    selected_features = (
        selected_features_pearson
        + selected_features_kendall
        + selected_features_spearman
    )
    selected_features = list(set(selected_features))
    logger.debug(f"Selected features : {len(selected_features)}")

    selected_features = [
        f for f in selected_features if data_trainval[f].isna().sum() == 0
    ]

    X_trainval = data_trainval[selected_features + selected_cat_features_numerized]
    y_trainval = data_trainval[target]

    X_test = data_test[selected_features + selected_cat_features_numerized]
    y_test = data_test[target]

    X_all = data_all[selected_features + selected_cat_features_numerized]
    y_all = data_all[target]

    features_dict = {
        "cat": selected_cat_features,
        "num": selected_num_features,
        "model": selected_features + selected_cat_features_numerized,
    }
    with open("data/features.json", "w") as outfile:
        json.dump(features_dict, outfile, indent=2)

    regressors = [model.get_model()]

    splits = 3
    repeats = 2
    splitter = model_selection.RepeatedKFold(
        n_splits=splits, n_repeats=repeats, random_state=0
    )

    logger.debug("Fitting model...")

    for step, regressor in enumerate(regressors):
        regressor_name = str(regressor.__class__.__name__)
        non_default_params = str(regressor).split("(")[1].split(")")[0]

        logger.debug(
            f"Model {step + 1} of {len(regressors)}: {regressor_name} {non_default_params}"
        )

        train_MAEs = []
        train_MSEs = []
        train_MAXs = []
        val_MAEs = []
        val_MSEs = []
        val_MSLEs = []
        val_MAPEs = []
        val_MAXs = []

        # End run if ened abnormally
        # try:
        #     mlflow.end_run()
        # except Exception as e:
        #     pass

        # mlflow.start_run(experiment_id=1)

        # mlflow.log_param("model", regressor_name)
        # mlflow.log_param("non_default_params", non_default_params)
        # mlflow.log_param("standardized_type", standardized_type)
        # mlflow.log_param("num_features", len(selected_num_features))
        # mlflow.log_param("cat_features", len(selected_cat_features))
        # SMOGN (SMOTE for regression) ?

        for step, (train_index, val_index) in enumerate(
            splitter.split(X_trainval, y_trainval)
        ):
            logger.debug(f"Step {step + 1} of {splits * repeats}")
            X_train = X_trainval.iloc[train_index, :]
            X_val = X_trainval.iloc[val_index, :]
            y_train = y_trainval.iloc[train_index]
            y_val = y_trainval.iloc[val_index]
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_val)
            y_pred_train = regressor.predict(X_train)
            train_MAEs.append(metrics.mean_absolute_error(y_train, y_pred_train))
            train_MSEs.append(metrics.mean_squared_error(y_train, y_pred_train))
            train_MAXs.append(metrics.max_error(y_train, y_pred_train))
            val_MAEs.append(metrics.mean_absolute_error(y_val, y_pred))
            val_MSEs.append(metrics.mean_squared_error(y_val, y_pred))
            # val_MSLEs.append(metrics.mean_squared_log_error(y_val, y_pred))
            val_MAPEs.append(metrics.mean_absolute_percentage_error(y_val, y_pred))
            val_MAXs.append(metrics.max_error(y_val, y_pred))
            # mlflow.log_metric(key="val_MAE", value=metrics.mean_absolute_error(y_val, y_pred), step=step)
            # mlflow.log_metric(key="val_MSE", value=metrics.mean_squared_error(y_val, y_pred), step=step)
            # #mlflow.log_metric(key="val_MSLE", value=metrics.mean_squared_log_error(y_val, y_pred), step=step)
            # mlflow.log_metric(key="val_MAPE", value=metrics.mean_absolute_percentage_error(y_val, y_pred), step=step)
            # mlflow.log_metric(key="train_MAE", value=metrics.mean_absolute_error(y_train, y_pred_train), step=step)
            # mlflow.log_metric(key="train_MSE", value=metrics.mean_squared_error(y_train, y_pred_train), step=step)
            # Add a MSE/MAE on MVP candidates
            # Add a metrics on MVP or MVP top 3
            # Clip predictions between 0.0 and 1.0 !

        logger.debug("Training MAE: %f", numpy.mean(train_MAEs))
        logger.debug("Training MSE: %f", numpy.mean(train_MSEs))
        logger.debug("Training MaxAE: %f", numpy.mean(train_MAXs))
        logger.debug("Validation MAE: %f", numpy.mean(val_MAEs))
        logger.debug("Validation MSE: %f", numpy.mean(val_MSEs))
        logger.debug("Validation MAPE: %f", numpy.mean(val_MAPEs))
        logger.debug("Validation MaxAE: %f", numpy.mean(val_MAXs))

        # mlflow.end_run()

    logger.debug("Performing test seasons analysis...")

    """
    regressor.fit(X_trainval, y_trainval)
    y_pred_test = regressor.predict(X_test)
    results = y_test.rename("TRUTH").to_frame()
    results.loc[:, "PRED"] = y_pred_test
    results.loc[:, "AE"] = (results["TRUTH"] - results["PRED"]).abs()
    results = results.merge(data_test[["SEASON"]], left_index=True, right_index=True)
    real_winners = data_test.sort_values(by=target, ascending=False).drop_duplicates(
        subset=["SEASON"], keep="first"
    )[["SEASON"]]
    real_winners["True MVP"] = real_winners.index
    real_winners = real_winners.set_index("SEASON", drop=True)
    winners = results.sort_values(by="PRED", ascending=False).drop_duplicates(
        subset=["SEASON"], keep="first"
    )
    winners["Pred. MVP"] = winners.index
    winners = winners.set_index("SEASON", drop=True)
    winners = winners.merge(real_winners, left_index=True, right_index=True)
    winners.loc[:, "REAL_RANK"] = winners["Pred. MVP"].map(ranks_reference)
    winners = winners.sort_index(ascending=True)
    print(winners)
    print(numpy.mean(results.AE))
    print(results.AE.max())
    print(numpy.mean(results.AE ** 2))
    winners["Real MVP rank"] = 1
    print("Pourcentage de MVP bien trouvé sur le jeu de test :")
    print((winners["Pred. MVP"] == winners["True MVP"]).sum() / len(winners))
    print("Rang réel moyen du MVP prédit:")
    print((winners["REAL_RANK"]).mean())
    """

    logger.debug("Performing all season analysis...")
    all_winners = pandas.DataFrame()
    for season in data_all.SEASON.unique():
        season_regressor = base.clone(regressor)
        logger.debug(f"Season {season}")
        data_all_train = data_all[data_all.SEASON != season]
        data_all_test = data_all[data_all.SEASON == season]
        X_all_train = data_all_train[
            selected_features + selected_cat_features_numerized
        ]
        y_all_train = data_all_train[target]
        X_all_test = data_all_test[selected_features + selected_cat_features_numerized]
        y_all_test = data_all_test[target]
        season_regressor.fit(X_all_train, y_all_train)
        y_pred_all_test = season_regressor.predict(X_all_test)

        results = y_all_test.rename("TRUTH").to_frame()
        results.loc[:, "PRED"] = y_pred_all_test
        results.loc[:, "AE"] = (results["TRUTH"] - results["PRED"]).abs()
        results = results.merge(
            data_all_test[["SEASON"]], left_index=True, right_index=True
        )
        # Export detailed results
        # results.sort_values(by="PRED", ascending=False).head(10).to_csv("./data/temp/"+str(season)+"_results.csv")
        real_winners = data_all_test.sort_values(
            by=target, ascending=False
        ).drop_duplicates(subset=["SEASON"], keep="first")[["SEASON"]]
        real_winners["True MVP"] = real_winners.index
        real_winners = real_winners.set_index("SEASON", drop=True)
        winners = results.sort_values(by="PRED", ascending=False).drop_duplicates(
            subset=["SEASON"], keep="first"
        )
        predicted_ranks_reference = results.PRED.rank(ascending=False, method="min")
        winners["Pred. MVP"] = winners.index
        winners = winners.set_index("SEASON", drop=True)
        winners = winners.merge(real_winners, left_index=True, right_index=True)
        winners.loc[:, "REAL_RANK"] = winners["Pred. MVP"].map(ranks_reference)
        winners.loc[:, "PRED_RANK"] = winners["True MVP"].map(predicted_ranks_reference)
        winners = winners.sort_index(ascending=True)
        all_winners = pandas.concat([all_winners, winners])

    logger.debug("Mean absolute error: %f", numpy.mean(results.AE))
    logger.debug("Max absolute error: %f", results.AE.max())
    logger.debug("Mean squared error: %f", numpy.mean(results.AE**2))
    all_winners["Real MVP rank"] = 1
    # To avoid extremely high values and skewed means, limit rank to 10
    all_winners["PRED_RANK"] = all_winners["PRED_RANK"].clip(upper=10)
    all_winners["REAL_RANK"] = all_winners["REAL_RANK"].clip(upper=10)
    logger.info(
        "Pourcentage de MVP bien trouvé sur le jeu de test : %f",
        (all_winners["Pred. MVP"] == all_winners["True MVP"]).sum() / len(all_winners),
    )
    logger.info("Rang réel moyen du MVP prédit: %f", (all_winners["REAL_RANK"]).mean())
    all_winners["Pred. MVP"] = all_winners["Pred. MVP"].map(data_all["PLAYER"])
    all_winners["True MVP"] = all_winners["True MVP"].map(data_all["PLAYER"])
    all_winners.to_csv(
        conf.data.performances.path,
        sep=conf.data.performances.sep,
        encoding=conf.data.performances.encoding,
        compression=conf.data.performances.compression,
        index=True,
    )

    final_regressor = base.clone(regressor)
    final_regressor.fit(X_all, y_all)
    joblib.dump(final_regressor, conf.data.model.path)


def filter_by_correlation_with_target(
    data, target, method="pearson", n_features=None, treshold=None
):
    logger.info("Method : %s", method)
    if n_features is not None and treshold is None:
        top_corr = analyze.get_columns_correlation_with_target(
            data, target, method=method
        )[:n_features]
    elif n_features is None and treshold is not None:
        top_corr = analyze.get_columns_correlation_with_target(
            data, target, method=method
        )
        top_corr = top_corr[top_corr > treshold]
    else:
        raise Exception("Invalid arguments")

    return top_corr


def train_model():
    try:
        make_bronze_data()
        make_silver_data()
        # make_gold_data()
        # train_on_gold()
        make_gold_data_and_train_model()
    except Exception as e:
        logger.error(f"Training model failed : {e}")
