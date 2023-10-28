import datetime
import time
from abc import ABC, abstractmethod
from os import path

import pandas
import requests
from bs4 import BeautifulSoup

from nba_mvp_predictor import conf, logger, utils

""" 
1955-56 through 1979-1980: Voting was done by players. Rules prohibited player from voting for himself or any teammate.
1980-81 to present: Voting conducted by media. 
"""


class Scrapper(ABC):
    """Abstract class working as an interface for scrapper classes."""

    @abstractmethod
    def retrieve_mvp_votes(season: int):
        pass

    @abstractmethod
    def get_mvp(self, subset_by_seasons: list = None):
        pass

    @abstractmethod
    def get_roster_stats_v2(season, stat_type):
        pass

    @abstractmethod
    def get_team_standings(self, subset_by_seasons: list = None):
        pass

    @abstractmethod
    def get_player_stats(
        self,
        subset_by_teams: list = None,
        subset_by_seasons: list = None,
        subset_by_stat_types: list = None,
    ):
        pass

    @abstractmethod
    def get_team_standings_on_date(self, day: int, month: int, year: int):
        pass


class BasketballReferenceScrapper(Scrapper):
    def __init__(self):
        self.team_names = utils.get_dict_from_yaml(
            "./nba_mvp_predictor/team_names.yaml"
        )

    @classmethod
    def get_request(cls, uri):
        root_url = "https://www.basketball-reference.com/"
        url = path.join(root_url, uri)
        logger.debug("Waiting time...")
        # Wait some time to avoid BR anti-bot measures (HTTP 429)
        utils.wait_random_time(2, 10)
        logger.debug("Requesting %s...", url)
        r = requests.get(url)
        if r.status_code == 200:
            return r
        else:
            logger.error("Failed to get %s", url)
            retry_after = r.headers["Retry-After"]
            if r.status_code == 429:
                message = f"(too many requests, retry after {retry_after})"
            else:
                message = f"(status code {r.status_code}"
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
        data = pandas.read_html(str(table), header=1)[0]
        data.columns = [str(col).upper() for col in data.columns]
        data.loc[:, "SEASON"] = season
        data = data.rename(columns={"SHARE": "MVP_VOTES_SHARE"})
        data = data.rename(columns={"TM": "TEAM"})
        data = data[["PLAYER", "TEAM", "SEASON", "MVP_VOTES_SHARE", "RANK"]]
        data.loc[:, "PLAYER"] = data["PLAYER"].str.replace(
            # "[^A-Za-z]", "", regex=True
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

    def get_mvp(self, subset_by_seasons: list = None):
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        if month > 9:
            # Season starts after september
            year += 1
        allowed_seasons = range(1974, year)
        if subset_by_seasons is not None:
            seasons = [
                season for season in subset_by_seasons if season in allowed_seasons
            ]
        else:
            seasons = allowed_seasons
        total_dfs = []
        for season in seasons:
            logger.info(f"Retrieving MVP of season {season}...")
            results = self.retrieve_mvp_votes(season)
            results.loc[:, "player_season_team"] = (
                results["PLAYER"].str.replace(" ", "")
                + "_"
                + results["SEASON"]
                + "_"
                + results["TEAM"]
            )
            results = results.set_index("player_season_team", drop=True)
            total_dfs.append(results)
        all_df = pandas.concat(
            total_dfs, join="outer", axis="index", ignore_index=False
        )
        return all_df

    @classmethod
    def get_roster_stats_v2(cls, season, stat_type):
        """
        Return all players stats for one season.
        Season : end year of season, as int or str
        Available stat types (case insensitive) : 'totals', 'per_game', 'per_36min', 'per_100poss', 'advanced'
        """
        season = str(season)
        stat_type = str(stat_type).lower()
        url_mapper = {
            "totals": "totals",
            "per_game": "per_game",
            "per_36min": "per_minute",
            "per_100poss": "per_poss",
            "advanced": "advanced",
        }
        stat_type = url_mapper[stat_type]
        uri = f"leagues/NBA_{season}_{stat_type}.html"
        r = cls.get_request(uri)
        soup = BeautifulSoup(r.content, "html.parser")
        table = soup.find("table")
        data = pandas.read_html(str(table))[0]
        data = data.loc[data.Player != "Player", :]
        data.columns = [str(col).upper() for col in data.columns]
        data.loc[:, "SEASON"] = season
        data.loc[:, "PLAYER"] = data["PLAYER"].str.replace(
            # "[^-'a-zA-ZÀ-ÿ]", "", regex=True
            "[ _'.*]",
            "",
            regex=True,
        )
        data = data.rename(columns={"TM": "TEAM"})
        data = data.drop("RK", axis="columns")
        for col in data.columns:
            if col.startswith("3P"):
                data[col] = data[col].fillna(0.0)
        return data

    @classmethod
    def get_standings(cls, date=None):
        """Ported from https://github.com/vishaalagartha/basketball_reference_scraper.


        Args:
            date (_type_, optional): _description_. Defaults to None.

        Raises:
            ConnectionError: _description_

        Returns:
            _type_: _description_
        """
        if date is None:
            date = datetime.now()
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
            e_df = pandas.read_html(str(e_table))[0]
            w_df = pandas.read_html(str(w_table))[0]
            e_df.rename(columns={"Eastern Conference": "TEAM"}, inplace=True)
            w_df.rename(columns={"Western Conference": "TEAM"}, inplace=True)
        d["EASTERN_CONF"] = e_df
        d["WESTERN_CONF"] = w_df
        return d

    def get_team_standings(self, subset_by_seasons: list = None):
        """Assumptions : the season is over by June 1st.
        TODO : Use the season dataset to find last game date.
        """
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        if month > 9:
            # Season starts after september
            year += 1
        allowed_seasons = range(1974, year + 1)
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
                logger.debug(f"Standings data columns: {', '.join(data.columns)}")
                data = data.dropna(axis="index", how="any")
                logger.debug(
                    f"First column name before renaming: {data.columns.values[0]}"
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
                logger.debug(f"Conference : {conference}")
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
        all_conf_df = pandas.concat(
            total_dfs, join="outer", axis="index", ignore_index=False
        )
        return all_conf_df

    def get_player_stats(
        self,
        subset_by_teams: list = None,
        subset_by_seasons: list = None,
        subset_by_stat_types: list = None,
    ):
        """
        Get a set of stats.
        Defaults to all teams, all seasons, all stat types.
        """
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        if month > 9:
            # Season starts after september
            year += 1
        allowed_stat_types = [
            "totals",
            "per_game",
            "per_36min",
            "per_100poss",
            "advanced",
        ]
        allowed_seasons = range(1974, year + 1)
        allowed_teams = list(set(self.team_names.values()))

        if subset_by_teams is not None:
            subset_by_teams = [str(s).upper() for s in subset_by_teams]

        if subset_by_seasons is not None:
            seasons = [
                season for season in subset_by_seasons if season in allowed_seasons
            ]
        else:
            seasons = allowed_seasons
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
            time.sleep(5)
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
                    stat_type_df = self.get_roster_stats_v2(season, stat_type)
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve data. Are you sure NBA was played in season {season}? {e}"
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

    def get_team_standings_on_date(self, day: int, month: int, year: int):
        uri = f"friv/standings.fcgi?month={month}&day={day}&year={year}&lg_id=NBA"
        r = self.get_request(uri)

        soup = BeautifulSoup(r.content, "html.parser")
        table_east = soup.find("table", {"id": "standings_e"})
        data_east = pandas.read_html(str(table_east))[0]
        table_west = soup.find("table", {"id": "standings_w"})
        data_west = pandas.read_html(str(table_west))[0]

        results = {
            "West": data_west,
            "East": data_east,
        }

        dfs = []

        for conference, data in results.items():
            logger.debug(f"Standings data columns: {', '.join(data.columns)}")
            data = data.dropna(axis="index", how="any")
            logger.debug(f"First column name before renaming: {data.columns.values[0]}")
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
            # data.loc[:, "CONF_RANK"] = data["W/L%"].rank(ascending=False, method='max')

            logger.debug(f"Conference : {conference}")
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
        all_conf_df = pandas.concat(dfs, join="outer", axis="index", ignore_index=False)
        return all_conf_df
