from typing import List

from nba_mvp_predictor import conf, logger
from nba_mvp_predictor import scrappers


def download_data(
    seasons: List[int] = None,
    scrapper: scrappers.Scrapper = scrappers.BasketballReferenceScrapper(),
):
    logger.info("Downloading player stats...")
    try:
        download_player_stats(seasons=seasons, scrapper=scrapper)
    except Exception as e:
        logger.error(f"Downloading player stats failed : {e}")
    logger.info("Downloading MVP votes...")
    try:
        download_mvp_votes(seasons=seasons, scrapper=scrapper)
    except Exception as e:
        logger.error(f"Downloading MVP votes failed : {e}")
    logger.info("Downloading team standings...")
    try:
        download_team_standings(seasons=seasons, scrapper=scrapper)
    except Exception as e:
        logger.error(f"Downloading team standings failed : {e}")


def download_player_stats(seasons: List[int], scrapper: scrappers.Scrapper):
    # We do not retrieve totals stats since we want to be able to predict at any moment in the season
    # That's not a big deal since we will have total games played, stats per game and per minute (will be highly correlated)
    # We could have normalized totals within the season if we'd have really want to use them
    data = scrapper.get_player_stats(
        subset_by_seasons=seasons,
        subset_by_stat_types=["per_game", "per_36min", "per_100poss", "advanced"],
    )
    data.to_csv(
        conf.data.player_stats.path,
        sep=conf.data.player_stats.sep,
        encoding=conf.data.player_stats.encoding,
        compression=conf.data.player_stats.compression,
        index=True,
    )


def download_mvp_votes(seasons: List[int], scrapper: scrappers.Scrapper):
    data = scrapper.get_mvp(
        subset_by_seasons=seasons,
    )
    data.to_csv(
        conf.data.mvp_votes.path,
        sep=conf.data.mvp_votes.sep,
        encoding=conf.data.mvp_votes.encoding,
        compression=conf.data.mvp_votes.compression,
        index=True,
    )


def download_team_standings(seasons: List[int], scrapper: scrappers.Scrapper):
    data = scrapper.get_team_standings(
        subset_by_seasons=seasons,
    )
    data.to_csv(
        conf.data.team_standings.path,
        sep=conf.data.team_standings.sep,
        encoding=conf.data.team_standings.encoding,
        compression=conf.data.team_standings.compression,
        index=True,
    )
