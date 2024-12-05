import time
from matplotlib import pyplot
import pandas
import requests
import seaborn

NUM_CURRENT_LEADERS = 1
NUM_ALL_TIME_LEADERS = 1
SHOW_FIRST_N_GAMES = 30

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
    # Set 3PA and 3PA to 0 where 3P is not numeric
    data.loc[~data["3P"].astype(str).str.isnumeric(), "3P"] = 0
    data.loc[~data["3PA"].astype(str).str.isnumeric(), "3PA"] = 0
    # Cast to int
    data["Game"] = data["Game"].astype(int)
    data["3P"] = data["3P"].astype(int)
    data["3PA"] = data["3PA"].astype(int)
    # Sort
    data = data.sort_values("Game", ascending=True)
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
    data = data[0][
        [
            "Player",
            "Season",
            "3PA",
        ]
    ].sort_values("3PA", ascending=False).head(top_n)
    data['Rank'] = data.index + 1
    return data

def plot_results(data: pandas.DataFrame):
    seaborn.set_theme()
    fig, ax = pyplot.subplots(figsize=(10, 5))
    data_to_display = data[data["Game"] <= SHOW_FIRST_N_GAMES]
    seaborn.lineplot(
        x="Game",
        y="Cumulative 3PA",
        hue="PlayerSeason",
        data=data_to_display[data_to_display["Current"]],
        ax=ax,
        palette="flare",
        linewidth=3,
        alpha=0.8,
    )
    seaborn.lineplot(
        x="Game",
        y="Cumulative 3PA",
        hue="PlayerSeason",
        data=data_to_display[~data_to_display["Current"]],
        ax=ax,
        palette="crest",
        linewidth=2,
        linestyle="--",
    )
    ax.set_title(
        f"Current and all-time leaders in 3PA", 
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Games")
    ax.set_ylabel("3PA")
    # X-axis must be integers
    ax.xaxis.set_major_locator(pyplot.MaxNLocator(integer=True))
    # Set legend at bottom right
    ax.legend(loc="lower right")
    pyplot.text(
        data_to_display["Game"].max(),
        data_to_display["Cumulative 3PA"].max() / 1.80,
        "@wontcalltimeout",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        alpha=0.5,
    )
    pyplot.text(
        SHOW_FIRST_N_GAMES / 2,
        data_to_display["Cumulative 3PA"].max() + 5,
        f"First {SHOW_FIRST_N_GAMES} games",
        fontstyle="italic",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    #seaborn.despine()
    fig.savefig(f"data/3pt_attempts.png")

def format_rank(rank: int) -> str:
    if rank == 1:
        return "1st"
    elif rank == 2:
        return "2nd"
    elif rank == 3:
        return "3rd"
    else:
        return f"{rank}th"

def main():
    use_average_for_all_time = True
    data = []
    current_leaders = get_current_leaders(top_n=NUM_CURRENT_LEADERS, current_season=2025)
    for leader in current_leaders.to_dict(orient="records"):
        time.sleep(0.99)
        game_log = get_game_logs(leader["Player"], leader["Season"])[["Game", "3PA"]]
        game_log["Cumulative 3PA"] = game_log["3PA"].cumsum()
        game_log["PlayerSeason"] = f"{leader['Player'].split(' ')[-1]} {leader['Season']}"
        game_log['Current'] = True
        data.append(game_log)
    leaders = get_leaders(top_n=NUM_ALL_TIME_LEADERS)
    leaders['Season'] = leaders['Season'].str[:4].astype(int) + 1
    for leader in leaders.to_dict(orient="records"):
        time.sleep(1.8) 
        game_log = get_game_logs(leader["Player"], leader["Season"])[["Game", "3PA"]]
        if use_average_for_all_time:
            avg_3pa = leader["3PA"] / 82
            game_log["3PA"] = avg_3pa
        game_log["Cumulative 3PA"] = game_log["3PA"].cumsum()
        game_log["PlayerSeason"] = f"{leader['Player'].split(' ')[-1]} {leader['Season']} ({format_rank(leader['Rank'])} all-time)"
        game_log['Current'] = False
        data.append(game_log)
    data = pandas.concat(data)
    plot_results(data)

if __name__ == "__main__":
    main()