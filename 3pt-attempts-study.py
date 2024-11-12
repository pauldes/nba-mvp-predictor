import time
from matplotlib import pyplot
import pandas
import requests
import seaborn

NUM_CURRENT_LEADERS = 3
NUM_ALL_TIME_LEADERS = 3

def get_player_slug(player: str) -> str:
    """ Get BBR player slug from player name.
    
    Stephen Curry -> curryst01
    Luka Dončić -> doncilu01
    """
    first_name, last_name = player.split(" ")
    # Get up to 5 first characters of last name
    last_name = last_name[:5].lower()
    # Get up to 2 first characters of first name
    first_name = first_name[:2].lower()
    return f"{last_name}{first_name}01"

def get_current_leaders(top_n=3, current_season=2025):
    url = f"https://www.basketball-reference.com/leagues/NBA_{current_season}_totals.html#totals_stats::fg3a"
    print('Loading', url, '...')
    r = requests.get(url)
    if r.status_code == 200:
        data = pandas.read_html(r.text)
    else:
        raise Exception(
            "Failed to fetch data"
            f"from {url}."
            f"Status code: {r.status_code}"
            f"Response: {r.text}"
        )
    data = data[0][
        [
            "Player",
            "3PA",
        ]
    ].sort_values("3PA", ascending=False).head(top_n)
    data['Season'] = current_season
    data['3PA'] = data['3PA'].astype(int)
    return data

def get_game_logs(player: str, season: int) -> pandas.DataFrame:
    """ Get game logs for a player in a season.
    """
    slug = get_player_slug(player)
    url = f"https://www.basketball-reference.com/players/{slug[0]}/{slug}/gamelog/{season}"
    print('Loading', url, '...')
    r = requests.get(url)
    if r.status_code == 200:
        data = pandas.read_html(r.text)
    else:
        raise Exception(
            "Failed to fetch data"
            f"from {url}."
            f"Status code: {r.status_code}"
            f"Response: {r.text}"
        )
    data = data[7][
        [
            "Rk",
            "3P",
            "3PA",
        ]
    ].sort_values("Rk", ascending=True).rename(
        columns={
            "Rk": "Game",
        }
    )
    # Keep rows where "Game" is a number
    data = data[data["Game"].astype(str).str.isnumeric()]
    data = data[data["3P"].astype(str).str.isnumeric()]
    # Cast to int
    data["Game"] = data["Game"].astype(int)
    data["3P"] = data["3P"].astype(int)
    data["3PA"] = data["3PA"].astype(int)
    # Add cumulative 3PA
    data = data.sort_values("Game", ascending=True)
    data["Cumulative 3PA"] = data["3PA"].cumsum()
    return data

def get_leaders(top_n=10):
    url = "https://www.basketball-reference.com/leaders/fg3a_season.html"
    print('Loading', url, '...')
    r = requests.get(url)
    if r.status_code == 200:
        data = pandas.read_html(r.text)
    else:
        raise Exception(
            "Failed to fetch data"
            f"from {url}."
            f"Status code: {r.status_code}"
            f"Response: {r.text}"
        )
    return data[0][
        [
            "Player",
            "Season",
            "3PA",
        ]
    ].sort_values("3PA", ascending=False).head(top_n)

def plot_results(data: pandas.DataFrame):
    seaborn.set_theme()
    fig, ax = pyplot.subplots(figsize=(10, 5))
    seaborn.lineplot(
        x="Game",
        y="Cumulative 3PA",
        hue="PlayerSeason",
        data=data,
        palette="tab10",
        ax=ax,
    )
    ax.set_title(f"{NUM_CURRENT_LEADERS} current and {NUM_ALL_TIME_LEADERS} all-time leaders in 3PA")
    ax.set_xlabel("Game")
    ax.set_ylabel("3PA")
    # Set legend at bottom right
    ax.legend(loc="lower right")
    seaborn.despine()
    fig.savefig(f"data/3pt_attempts.png")

def main():
    data = []
    current_leaders = get_current_leaders(top_n=NUM_CURRENT_LEADERS, current_season=2025)
    for leader in current_leaders.to_dict(orient="records"):
        time.sleep(1.8)
        game_log = get_game_logs(leader["Player"], leader["Season"])[["Game", "Cumulative 3PA"]]
        game_log["PlayerSeason"] = f"{leader['Player'].split(' ')[-1]}{leader['Season']}"
        data.append(game_log)
    leaders = get_leaders(top_n=NUM_ALL_TIME_LEADERS)
    leaders['Season'] = leaders['Season'].str[:4].astype(int) + 1
    for leader in leaders.to_dict(orient="records"):
        time.sleep(1.8)
        game_log = get_game_logs(leader["Player"], leader["Season"])[["Game", "Cumulative 3PA"]]
        game_log["PlayerSeason"] = f"{leader['Player'].split(' ')[-1]}{leader['Season']}"
        data.append(game_log)
    data = pandas.concat(data)
    data.to_csv("data/3pt_attempts.csv", index=False)
    plot_results(data)

if __name__ == "__main__":
    main()