from bs4 import BeautifulSoup
from matplotlib import pyplot
import pandas
import seaborn

from nba_mvp_predictor.scrappers import BasketballReferenceScrapper

DRAFT_YEARS = range(1999, 2027)
MAX_PICK = 10
MIN_MINUTES_PER_GAME = 5
MIN_GAMES = 10
HIGHLIGHTED_PLAYER_ID = "acuffda01"
HIGHLIGHTED_PLAYER_NAME = "Darius Acuff Jr."
CACHE_PATH = "data/rookie_shot_attempts.csv"


def parse_table(html: bytes, table_id: str) -> pandas.DataFrame:
    table = BeautifulSoup(html, "html.parser").find("table", id=table_id)
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = {cell.get("data-stat"): cell for cell in tr.find_all(["th", "td"])}
        player = cells.get("player") or cells.get("name_display")
        if player is None or player.a is None:
            continue
        row = {stat: cell.get_text() for stat, cell in cells.items()}
        row["player_id"] = player.a["href"].split("/")[-1].removesuffix(".html")
        rows.append(row)
    return pandas.DataFrame(rows)


def get_top_picks(draft_year: int) -> pandas.DataFrame:
    r = BasketballReferenceScrapper.get_request(f"draft/NBA_{draft_year}.html")
    data = parse_table(r.content, "stats")
    data["pick"] = pandas.to_numeric(data["pick_overall"])
    data = data[data["pick"] <= MAX_PICK]
    data["season"] = draft_year + 1
    return data[["player_id", "player", "pick", "season"]]


def get_per_game_stats(season: int) -> pandas.DataFrame:
    r = BasketballReferenceScrapper.get_request(f"leagues/NBA_{season}_per_game.html")
    data = parse_table(r.content, "per_game_stats")
    # Traded players have their season total row first, then one row per team
    data = data.drop_duplicates(subset="player_id", keep="first")
    data = data.rename(
        columns={"games": "G", "mp_per_g": "MP", "fga_per_g": "FGA", "fta_per_g": "FTA"}
    )
    data[["G", "MP", "FGA", "FTA"]] = data[["G", "MP", "FGA", "FTA"]].apply(
        pandas.to_numeric
    )
    data["season"] = season
    return data[["player_id", "season", "G", "MP", "FGA", "FTA"]]


def get_data() -> pandas.DataFrame:
    try:
        return pandas.read_csv(CACHE_PATH)
    except FileNotFoundError:
        pass
    rookies = []
    for draft_year in DRAFT_YEARS:
        print("Loading draft", draft_year, "...")
        picks = get_top_picks(draft_year)
        season_stats = get_per_game_stats(draft_year + 1)
        rookies.append(picks.merge(season_stats, on=["player_id", "season"]))
    data = pandas.concat(rookies, ignore_index=True)
    data.to_csv(CACHE_PATH, index=False)
    return data


def plot_results(data: pandas.DataFrame, highlighted: pandas.Series):
    seaborn.set_theme()
    fig, ax = pyplot.subplots(figsize=(10, 5))
    counts = data["TSA"].floordiv(1).astype(int).value_counts().sort_index()
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 1), fill_value=0)
    highlighted_bin = int(highlighted["TSA"] // 1)
    colors = ["red" if tsa == highlighted_bin else "steelblue" for tsa in counts.index]
    ax.bar(counts.index, counts.values, width=0.8, color=colors)
    rank = (data["TSA"] > highlighted["TSA"]).sum() + 1
    ax.annotate(
        f"{HIGHLIGHTED_PLAYER_NAME}\n{highlighted['TSA']:.1f} per game\n"
        f"(#{rank} of {len(data)})",
        xy=(highlighted_bin, counts[highlighted_bin]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color="red",
        fontweight="bold",
        fontsize=10,
    )
    fig.suptitle("Shot attempts among rookies", fontsize=14, fontweight="bold", y=0.96)
    ax.set_title(
        f"Top {MAX_PICK} picks, rookie seasons {min(DRAFT_YEARS) + 1}-{max(DRAFT_YEARS) + 1}, "
        f"min {MIN_MINUTES_PER_GAME} min per game and {MIN_GAMES} games ({len(data)} rookies)",
        fontsize=10,
        fontstyle="italic",
        color="dimgrey",
    )
    ax.set_xlabel("True shot attempts per game")
    ax.set_ylabel("Rookies")
    ax.set_xticks(counts.index)
    ax.yaxis.set_major_locator(pyplot.MaxNLocator(integer=True))
    ax.set_ylim(0, counts.max() * 1.15)
    ax.text(
        0.5,
        -0.2,
        "True shot attempts = FGA + 0.44 x FTA",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        fontstyle="italic",
        color="grey",
    )
    ax.text(
        1.01,
        0.5,
        "@wontcalltimeout",
        transform=ax.transAxes,
        fontsize=8,
        rotation=90,
        va="center",
        color="grey",
        alpha=0.5,
    )
    seaborn.despine(left=True)
    pyplot.tight_layout()
    fig.savefig("data/rookie_shot_attempts.png")


def main():
    data = get_data()
    data["TSA"] = data["FGA"] + 0.44 * data["FTA"]
    data = data[(data["MP"] >= MIN_MINUTES_PER_GAME) & (data["G"] >= MIN_GAMES)]
    highlighted = data[data["player_id"] == HIGHLIGHTED_PLAYER_ID]
    if len(highlighted) != 1:
        raise ValueError(f"{HIGHLIGHTED_PLAYER_ID} not found after filters")
    print(data.sort_values("TSA", ascending=False).head(10).to_string(index=False))
    plot_results(data, highlighted.iloc[0])


if __name__ == "__main__":
    main()
