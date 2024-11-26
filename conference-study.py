import time
from matplotlib import pyplot
import pandas
import requests
from matplotlib.font_manager import fontManager, FontProperties
import seaborn

SEASONS = range(1975, 2026)

def get_wins_vs_other_conference(season: int):
    """ Get wins for each conference in a season.

    Returns a tuples (east_wins_vs_west, west_wins_vs_east).
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_standings.html#expanded_standings"
    print('Loading', url, '...')
    r = requests.get(url)
    if r.status_code == 200:
        data = pandas.read_html(
            r.text.replace("<!--", "").replace("-->", ""), # Remove comments
            attrs={"id": "expanded_standings"},
            header=1,
        )
    else:
        raise Exception(
            "Failed to fetch data"
            f"from {url}."
            f"Status code: {r.status_code}"
            f"Response: {r.text}"
        )
    data = data[0][['Team', 'E', 'W']]
    data['total_vs_east'] = data['E'].str.extract(r'(\d+)-(\d+)').astype(int).sum(axis=1)
    data['total_vs_west'] = data['W'].str.extract(r'(\d+)-(\d+)').astype(int).sum(axis=1)
    data['wins_vs_east'] = data['E'].str.extract(r'(\d+)-(\d+)').astype(int).iloc[:, 0]
    data['wins_vs_west'] = data['W'].str.extract(r'(\d+)-(\d+)').astype(int).iloc[:, 0]
    data['conference'] = (data['total_vs_east'] > data['total_vs_west']).map({True: 'East', False: 'West'})
    east_wins_vs_west = data[data['conference'] == 'East']['wins_vs_west'].sum()
    west_wins_vs_east = data[data['conference'] == 'West']['wins_vs_east'].sum()
    return east_wins_vs_west, west_wins_vs_east

def get_wins(season: int):
    """ Get wins for each conference in a season.

    Returns a tuples (east_wins, west_wins).
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
    print('Loading', url, '...')
    r = requests.get(url)
    if r.status_code == 200:
        east_wins = pandas.read_html(
            r.text,
            attrs={"id": "confs_standings_E"},
        )[0]["W"].sum()
        west_wins = pandas.read_html(
            r.text,
            attrs={"id": "confs_standings_W"},
        )[0]["W"].sum()
    else:
        raise Exception(
            "Failed to fetch data"
            f"from {url}."
            f"Status code: {r.status_code}"
            f"Response: {r.text}"
        )
    return east_wins, west_wins


def plot_results(data: pandas.DataFrame):
    seaborn.set_theme()
    fig, ax = pyplot.subplots(figsize=(10, 5))
    data_until_last_season = data[data.Season <= data.Season.max() - 1]
    data_last_two_seasons = data[data.Season >= data.Season.max() - 1]
    seaborn.lineplot(
        x="Season",
        y="West win %",
        data=data_until_last_season,
        ax=ax,
        color='white',
        linewidth=2,
        markers=True,
        marker='o',
    )
    seaborn.lineplot(
        x="Season",
        y="West win %",
        data=data_last_two_seasons,
        ax=ax,
        color='white',
        linewidth=2,
        markers=True,
        marker='o',
        linestyle='dashed',
    )
    # Fill under the line (from 0 to the line) in red
    ax.fill_between(data["Season"], 0, data["West win %"], alpha=0.7, color="#f9665e")
    # Fill above the line (from the line to 1) in blue
    ax.fill_between(data["Season"], data["West win %"], 1, alpha=0.7, color="#799fcb")
    ax.set_title(
        f"Win % in direct matchups",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # Show grid
    ax.grid(True, which='major', linewidth=.5, color='lightgrey')
    ax.xaxis.set_major_locator(pyplot.MaxNLocator(integer=True))
    ax.set_ylim(0.25, 0.75)
    ax.set_xlim(data["Season"].min(), data["Season"].max())
    ax.set_facecolor('white')
    # Change y axis to get 0.5 in the middle
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    # Change y axis to percentages
    ax.set_yticklabels(["30%", "40%", "50%", "60%", "70%"])
    # Draw a line at 50%
    pyplot.axhline(0.5, linestyle="dotted", linewidth=1, color="black")
    # Add a text in each area
    pyplot.text(
        data["Season"].median() + 5,
        0.35,
        "WEST WINS",
        fontsize=15,
        color='white',
        fontweight="bold",
    )
    pyplot.text(
        data["Season"].median() - 15,
        0.65,
        "EAST WINS",
        fontsize=15,
        color='white',
        fontweight="bold",

    )
    pyplot.text(
        data["Season"].max() - 1,
        0.3,
        "@wontcalltimeout",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="bottom",
        rotation=90,
        color="grey",
    )
    # Add maximal value
    max_west = data[data.Season < data.Season.max()]["West win %"].max()
    max_season_west = data[data["West win %"] == max_west]["Season"].values[0]
    pyplot.text(
        max_season_west,
        max_west + 0.02,
        f"{max_west:.1%}",
        fontsize=9,
        color="white",
        horizontalalignment="center",
        verticalalignment="center",
        fontstyle="italic",
        fontweight="bold",
    )
    seaborn.despine()
    fig.savefig(f"data/conference_wins.png")

def main():
    try:
        current_season = 2025
        data_old = pandas.read_csv("data/conference_wins.csv", index_col=False)
        data_old = data_old[data_old['Season'] < current_season]
        data = pandas.DataFrame(columns=["Season", "East win %", "West win %"])
        east_wins, west_wins = get_wins_vs_other_conference(current_season)
        pct_wins_east = east_wins / (east_wins + west_wins)
        pct_wins_west = west_wins / (east_wins + west_wins)
        data.loc[len(data)] = [current_season, pct_wins_east, pct_wins_west]
        data = pandas.concat(
            [data_old, data],
            ignore_index=True,
        )
    except FileNotFoundError:
        data = pandas.DataFrame(columns=["Season", "East win %", "West win %"])
        for season in SEASONS:
            try:
                east_wins, west_wins = get_wins_vs_other_conference(season)
                pct_wins_east = east_wins / (east_wins + west_wins)
                pct_wins_west = west_wins / (east_wins + west_wins)
                time.sleep(4.33)
                data.loc[len(data)] = [season, pct_wins_east, pct_wins_west]
            except Exception as e:
                print(f"Failed to fetch data for season {season}.", e)
    data.to_csv("data/conference_wins.csv", index=False)
    plot_results(data)

if __name__ == "__main__":
    main()