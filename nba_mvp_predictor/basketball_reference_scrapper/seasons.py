"""
This file is imported from
https://github.com/vishaalagartha/basketball_reference_scraper
Using this package as a dependency fails (requirements conflict)
"""

import pandas as pd
from bs4 import BeautifulSoup
from requests import get


def get_schedule(season, playoffs=False):
    months = [
        "October",
        "November",
        "December",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
    ]
    if season == 2020:
        months = [
            "October-2019",
            "November",
            "December",
            "January",
            "February",
            "March",
            "July",
            "August",
            "September",
            "October-2020",
        ]
    df = pd.DataFrame()
    for month in months:
        r = get(
            f"https://www.basketball-reference.com/leagues/NBA_{season}_games-{month.lower()}.html"
        )
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, "html.parser")
            table = soup.find("table", attrs={"id": "schedule"})
            if table:
                month_df = pd.read_html(str(table))[0]
                df = pd.concat([df, month_df])

    df = df.reset_index()

    cols_to_remove = [i for i in df.columns if "Unnamed" in i]
    cols_to_remove += [i for i in df.columns if "Notes" in i]
    cols_to_remove += [i for i in df.columns if "Start" in i]
    cols_to_remove += [i for i in df.columns if "Attend" in i]
    cols_to_remove += [i for i in df.columns if "Arena" in i]
    cols_to_remove += ["index"]
    df = df.drop(cols_to_remove, axis=1)
    df.columns = ["DATE", "VISITOR", "VISITOR_PTS", "HOME", "HOME_PTS"]

    if season == 2020:
        df = df[df["DATE"] != "Playoffs"]
        df["DATE"] = df["DATE"].apply(lambda x: pd.to_datetime(x))
        df = df.sort_values(by="DATE")
        df = df.reset_index().drop("index", axis=1)
        playoff_loc = df[df["DATE"] == pd.to_datetime("2020-08-17")].head(n=1)
        if len(playoff_loc.index) > 0:
            playoff_index = playoff_loc.index[0]
        else:
            playoff_index = len(df)
        if playoffs:
            df = df[playoff_index:]
        else:
            df = df[:playoff_index]
    else:
        # account for 1953 season where there's more than one "playoffs" header
        if season == 1953:
            df.drop_duplicates(subset=["DATE", "HOME", "VISITOR"], inplace=True)
        playoff_loc = df[df["DATE"] == "Playoffs"]
        if len(playoff_loc.index) > 0:
            playoff_index = playoff_loc.index[0]
        else:
            playoff_index = len(df)
        if playoffs:
            df = df[playoff_index + 1 :]
        else:
            df = df[:playoff_index]
        df["DATE"] = df["DATE"].apply(lambda x: pd.to_datetime(x))
    return df
