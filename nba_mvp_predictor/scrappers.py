import datetime
import time
from abc import ABC, abstractmethod
from io import StringIO
from os import path
from typing import ClassVar
from urllib.parse import urljoin

import pandas
from bs4 import BeautifulSoup
from curl_cffi import requests as _br_http

from nba_mvp_predictor import logger, utils

"""
1955-56 through 1979-1980: Voting was done by players. Rules prohibited player from voting
for himself or any teammate.
1980-81 to present: Voting conducted by media.
"""

_TEAM_NAMES_PATH = path.join(path.dirname(__file__), "team_names.yaml")


class Scrapper(ABC):
    """Abstract interface for scraping Basketball Reference."""

    #: First season end year (BR-style label, e.g. 1974 = 1973–74). Each concrete subclass must
    #: assign this.
    FIRST_SEASON_END: ClassVar[int]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is Scrapper:
            return
        if "FIRST_SEASON_END" not in cls.__dict__:
            raise TypeError(
                f"{cls.__qualname__} must define class attribute FIRST_SEASON_END "
                "(int: first season end year, e.g. 1974 for 1973–74)."
            )

    def __init__(self):
        self.team_names = utils.get_dict_from_yaml(_TEAM_NAMES_PATH)

    @classmethod
    @abstractmethod
    def retrieve_mvp_votes(cls, season):
        pass

    @classmethod
    @abstractmethod
    def wait_between_request(cls) -> None:
        """Pause before each outbound HTTP request (rate limiting / politeness)."""

    @classmethod
    @abstractmethod
    def fetch_single_season_league_stat_table(cls, season, stat_type):
        """One season, one stat mode (e.g. per_game), one league-wide table."""
        pass

    @abstractmethod
    def get_team_standings(self, subset_by_seasons: list[int] | None = None):
        pass

    @abstractmethod
    def get_team_standings_on_date(self, day: int, month: int, year: int):
        pass

    def get_mvp(self, subset_by_seasons: list[int] | None = None):
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        if month > 9:
            year += 1
        allowed_seasons = range(self.__class__.FIRST_SEASON_END, year)
        if subset_by_seasons is not None:
            seasons = [
                season for season in subset_by_seasons if season in allowed_seasons
            ]
        else:
            seasons = allowed_seasons
        total_dfs = []
        for season in seasons:
            logger.info(f"Retrieving MVP of season {season}...")
            results = self.__class__.retrieve_mvp_votes(season)
            results.loc[:, "player_season_team"] = (
                results["PLAYER"].str.replace(" ", "")
                + "_"
                + results["SEASON"]
                + "_"
                + results["TEAM"]
            )
            results = results.set_index("player_season_team", drop=True)
            total_dfs.append(results)
        return pandas.concat(total_dfs, join="outer", axis="index", ignore_index=False)

    def build_multi_season_league_player_stats(
        self,
        subset_by_teams: list[str] | None = None,
        subset_by_seasons: list[int] | None = None,
        subset_by_stat_types: list[str] | None = None,
    ):
        """
        Merge many seasons and many stat modes into one wide player-level DataFrame.

        Defaults: all teams, all allowed seasons, all stat modes.
        """
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        if month > 9:
            year += 1
        allowed_stat_types = [
            "totals",
            "per_game",
            "per_36min",
            "per_100poss",
            "advanced",
        ]
        allowed_seasons = self._allowed_season_end_years(year)
        allowed_teams = list(set(self.team_names.values()))

        if subset_by_teams is not None:
            subset_by_teams = [str(s).upper() for s in subset_by_teams]

        if subset_by_seasons is not None:
            seasons = [
                season for season in subset_by_seasons if season in allowed_seasons
            ]
        else:
            seasons = list(allowed_seasons)
        if subset_by_stat_types is not None:
            subset_by_stat_types = [str(s).lower() for s in subset_by_stat_types]
            stat_types = [
                stat_type
                for stat_type in subset_by_stat_types
                if stat_type in allowed_stat_types
            ]
        else:
            stat_types = allowed_stat_types

        season_dfs = []
        for season in seasons:
            do_not_suffix = [
                "PLAYER",
                "POS",
                "AGE",
                "TEAM",
                "SEASON",
                "G",
                "GS",
                "FG%",
                "3P%",
                "FT%",
                "2P%",
                "eFG%",
                "MP",
            ]
            stat_type_dfs = []
            for stat_type in stat_types:
                logger.info(f"Retrieving {stat_type} stats for season {season}...")
                try:
                    stat_type_df = self.__class__.fetch_single_season_league_stat_table(
                        season, stat_type
                    )
                except Exception as e:
                    logger.error(
                        "Could not retrieve data. Are you sure NBA was played in season %s? %s",
                        season,
                        e,
                    )
                else:
                    stat_type_df.columns = [
                        col + "_" + str(stat_type) if col not in do_not_suffix else col
                        for col in stat_type_df.columns
                    ]
                    stat_type_df.loc[:, "player_season_team"] = (
                        stat_type_df["PLAYER"].str.replace(" ", "")
                        + "_"
                        + stat_type_df["SEASON"]
                        + "_"
                        + stat_type_df["TEAM"]
                    )
                    stat_type_df = stat_type_df.set_index(
                        "player_season_team", drop=True
                    )
                    stat_type_df = stat_type_df.dropna(axis="columns", how="all")
                    if stat_type_df.index.duplicated().any():
                        duplicated_indexes = stat_type_df.index[
                            stat_type_df.index.duplicated()
                        ].tolist()
                        logger.warning(
                            "Duplicate index values found in %s stats for season %s: %s. Removing duplicates.",
                            stat_type,
                            season,
                            duplicated_indexes,
                        )
                        stat_type_df = stat_type_df[
                            ~stat_type_df.index.duplicated(keep="first")
                        ]
                    stat_type_dfs.append(stat_type_df)

            season_df = pandas.concat(
                stat_type_dfs, join="outer", axis="columns", ignore_index=False
            )
            season_df = season_df.loc[:, ~season_df.columns.duplicated()]
            season_dfs.append(season_df)

        full_df = pandas.concat(
            season_dfs, join="outer", axis="index", ignore_index=False
        )

        if subset_by_teams is not None:
            full_df = full_df[full_df.TEAM.isin(subset_by_teams)]

        return full_df

    def _allowed_season_end_years(self, calendar_year_upper: int):
        """Season end years from ``FIRST_SEASON_END`` through ``calendar_year_upper`` inclusive."""
        return range(self.__class__.FIRST_SEASON_END, calendar_year_upper + 1)


class BasketballReferenceScrapper(Scrapper):
    FIRST_SEASON_END = 1974
    BR_ORIGIN = "https://www.basketball-reference.com"
    BR_IMPERSONATE_DEFAULT = "firefox147"
    BR_REQUEST_TIMEOUT_SECONDS = 60.0

    @classmethod
    def wait_between_request(cls) -> None:
        # Basketball Reference asks for at least ~3 seconds between requests.
        time.sleep(utils.sample_uniform_seconds(3.0, 4.0))

    @classmethod
    def get_request(cls, uri):
        url = urljoin(f"{cls.BR_ORIGIN}/", uri)
        cls.wait_between_request()
        impersonate = cls.BR_IMPERSONATE_DEFAULT
        logger.debug("Requesting %s (impersonate=%s)...", url, impersonate)
        r = _br_http.get(
            url,
            impersonate=impersonate,
            timeout=cls.BR_REQUEST_TIMEOUT_SECONDS,
        )
        if r.status_code == 200:
            return r
        logger.error("Failed to get %s", url)
        retry_after = r.headers.get("Retry-After", "")
        if r.status_code == 429:
            message = f"(too many requests, retry after {retry_after})"
        else:
            message = f"(status code {r.status_code})"
        raise ConnectionError(
            f"Could not connect to BR and get data, status code : {message}"
        )

    @classmethod
    def retrieve_mvp_votes(cls, season):
        season = str(season)
        uri = f"awards/awards_{season}.html"
        r = cls.get_request(uri)
        soup = BeautifulSoup(r.content, "html.parser")
        table_mvp = soup.find("table", id="mvp")
        table_nba_mvp = soup.find("table", id="nba_mvp")
        if table_mvp is not None:
            table = table_mvp
        elif table_nba_mvp is not None:
            table = table_nba_mvp
        else:
            raise Exception("No table found for MVP data for season", season)
        data = pandas.read_html(StringIO(str(table)), header=1)[0]
        data.columns = [str(col).upper() for col in data.columns]
        data.loc[:, "SEASON"] = season
        data = data.rename(columns={"SHARE": "MVP_VOTES_SHARE"})
        data = data.rename(columns={"TM": "TEAM"})
        data = data[["PLAYER", "TEAM", "SEASON", "MVP_VOTES_SHARE", "RANK"]]
        data.loc[:, "PLAYER"] = data["PLAYER"].str.replace(
            "[ _'.*]",
            "",
            regex=True,
        )
        data.loc[:, "MVP_WINNER"] = False
        data["RANK"] = (
            data["RANK"]
            .astype(str)
            .str.replace("[^0-9]", "", regex=True)
            .astype(int, errors="raise")
        )
        data.loc[data["RANK"] == 1, "MVP_WINNER"] = True
        data.loc[:, "MVP_PODIUM"] = False
        data.loc[data["RANK"].isin([1, 2, 3]), "MVP_PODIUM"] = True
        data.loc[:, "MVP_CANDIDATE"] = True
        data = data.drop("RANK", axis="columns")
        return data

    @classmethod
    def fetch_single_season_league_stat_table(cls, season, stat_type):
        season = str(season)
        stat_type = str(stat_type).lower()
        url_mapper = {
            "totals": "totals",
            "per_game": "per_game",
            "per_36min": "per_minute",
            "per_100poss": "per_poss",
            "advanced": "advanced",
        }
        not_stats = ["RK", "AWARDS"]
        stat_type = url_mapper[stat_type]
        uri = f"leagues/NBA_{season}_{stat_type}.html"
        r = cls.get_request(uri)
        soup = BeautifulSoup(r.content, "html.parser")
        table = soup.find("table")
        data = pandas.read_html(StringIO(str(table)))[0]
        data = data.loc[data.Player != "Player", :]
        data.columns = [str(col).upper() for col in data.columns]
        data.loc[:, "SEASON"] = season
        data.loc[:, "PLAYER"] = data["PLAYER"].str.replace(
            "[ _'.*]",
            "",
            regex=True,
        )
        data = data.rename(columns={"TM": "TEAM"})
        data = data.drop(
            [col for col in data.columns if col in not_stats], axis="columns"
        )
        for col in data.columns:
            if col.startswith("3P"):
                data[col] = data[col].fillna(0.0)
        return data

    @classmethod
    def get_standings(cls, date=None):
        """Ported from https://github.com/vishaalagartha/basketball_reference_scraper."""
        if date is None:
            date = datetime.datetime.now()
        else:
            date = pandas.to_datetime(date)
        d = {}
        uri = f"friv/standings.fcgi?month={date.month}&day={date.day}&year={date.year}"
        r = cls.get_request(uri)

        soup = BeautifulSoup(r.content, "html.parser")
        e_table = soup.find("table", attrs={"id": "standings_e"})
        w_table = soup.find("table", attrs={"id": "standings_w"})
        e_df = pandas.DataFrame(
            columns=["TEAM", "W", "L", "W/L%", "GB", "PW", "PL", "PS/G", "PA/G"]
        )
        w_df = pandas.DataFrame(
            columns=["TEAM", "W", "L", "W/L%", "GB", "PW", "PL", "PS/G", "PA/G"]
        )
        if e_table and w_table:
            e_df = pandas.read_html(StringIO(str(e_table)))[0]
            w_df = pandas.read_html(StringIO(str(w_table)))[0]
            e_df.rename(columns={"Eastern Conference": "TEAM"}, inplace=True)
            w_df.rename(columns={"Western Conference": "TEAM"}, inplace=True)
        d["EASTERN_CONF"] = e_df
        d["WESTERN_CONF"] = w_df
        return d

    def get_team_standings(self, subset_by_seasons: list[int] | None = None):
        """Assumptions : the season is over by June 1st."""
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        if month > 9:
            year += 1
        allowed_seasons = range(self.__class__.FIRST_SEASON_END, year + 1)
        if subset_by_seasons is not None:
            seasons = [
                season for season in subset_by_seasons if season in allowed_seasons
            ]
        else:
            seasons = allowed_seasons
        total_dfs = []
        for season in seasons:
            logger.info(f"Retrieving standings of season {season}...")
            date = "06-01-" + str(season)
            dfs = []
            results = self.get_standings(date=date)
            for conference, data in results.items():
                logger.debug("Standings data columns: %s", ", ".join(data.columns))
                data = data.dropna(axis="index", how="any")
                logger.debug(
                    "First column name before renaming: %s", data.columns.values[0]
                )
                data = data.rename(columns={data.columns[0]: "TEAM"})
                data.loc[:, "TEAM"] = (
                    data["TEAM"].str.upper().str.replace("[^A-Z]", "", regex=True)
                )
                team_names = {}
                for raw, short in self.team_names.items():
                    raw = "".join(filter(str.isalpha, raw)).upper()
                    team_names[raw] = short
                data = data[~data["TEAM"].str.contains("DIVISION")]
                data["W/L%"] = data["W/L%"].astype("float32")
                data["W"] = data["W"].astype("int32")
                data["L"] = data["L"].astype("int32")
                data = data.sort_values(by="W/L%", ascending=False)
                data = data.reset_index(drop=True)
                data.loc[:, "CONF_RANK"] = data.index + 1
                logger.debug("Conference : %s", conference)
                data.loc[:, "CONF"] = (
                    conference.replace(" ", "_").upper().replace("CONFERENCE", "CONF")
                )

                unmapped_teams = [
                    team
                    for team in data["TEAM"].unique()
                    if team not in team_names.keys()
                ]
                data.loc[:, "TEAM"] = data["TEAM"].map(team_names)
                if data["TEAM"].isna().sum() > 0:
                    raise ValueError("Unknown/unmapped teams : %s", unmapped_teams)
                data["GB"] = (
                    data["GB"].str.replace("—", "0.0").astype(float, errors="raise")
                )
                data.loc[:, "TEAM_SEASON"] = data["TEAM"] + "_" + str(season)
                data.loc[:, "SEASON"] = season
                data = data.set_index("TEAM_SEASON", drop=True)
                dfs.append(data)
            all_conf_df = pandas.concat(
                dfs, join="outer", axis="index", ignore_index=False
            )
            total_dfs.append(all_conf_df)
        return pandas.concat(total_dfs, join="outer", axis="index", ignore_index=False)

    def get_team_standings_on_date(self, day: int, month: int, year: int):
        uri = f"friv/standings.fcgi?month={month}&day={day}&year={year}&lg_id=NBA"
        r = self.get_request(uri)

        soup = BeautifulSoup(r.content, "html.parser")
        table_east = soup.find("table", {"id": "standings_e"})
        data_east = pandas.read_html(StringIO(str(table_east)))[0]
        table_west = soup.find("table", {"id": "standings_w"})
        data_west = pandas.read_html(StringIO(str(table_west)))[0]

        results = {
            "West": data_west,
            "East": data_east,
        }

        dfs = []

        for conference, data in results.items():
            logger.debug("Standings data columns: %s", ", ".join(data.columns))
            data = data.dropna(axis="index", how="any")
            logger.debug(
                "First column name before renaming: %s", data.columns.values[0]
            )
            data = data.rename(columns={data.columns[0]: "TEAM"})
            data.loc[:, "TEAM"] = (
                data["TEAM"].str.upper().str.replace("[^A-Z]", "", regex=True)
            )
            team_names = {}
            for raw, short in self.team_names.items():
                raw = "".join(filter(str.isalpha, raw)).upper()
                team_names[raw] = short
            data = data[~data["TEAM"].str.contains("DIVISION")]

            data = data.sort_values(by="W/L%", ascending=False)
            data = data.reset_index(drop=True)
            data.loc[:, "CONF_RANK"] = data.index + 1

            logger.debug("Conference : %s", conference)
            data.loc[:, "CONF"] = (
                conference.replace(" ", "_").upper().replace("CONFERENCE", "CONF")
            )
            unmapped_teams = [
                team for team in data["TEAM"].unique() if team not in team_names.keys()
            ]
            data.loc[:, "TEAM"] = data["TEAM"].map(team_names)
            if data["TEAM"].isna().sum() > 0:
                raise ValueError("Unknown/unmapped teams : %s", unmapped_teams)
            data["GB"] = (
                data["GB"].str.replace("—", "0.0").astype(float, errors="raise")
            )
            data = data.astype(
                {
                    "W": "int32",
                    "L": "int32",
                    "CONF_RANK": "int32",
                    "W/L%": "float64",
                }
            )
            data = data.set_index("TEAM", drop=True)
            dfs.append(data)
        return pandas.concat(dfs, join="outer", axis="index", ignore_index=False)
