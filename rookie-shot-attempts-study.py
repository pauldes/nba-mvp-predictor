from bs4 import BeautifulSoup
from matplotlib import pyplot
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy
import pandas
import seaborn

from nba_mvp_predictor.scrappers import BasketballReferenceScrapper

DRAFT_YEARS = range(1997, 2027)
MAX_PICK = 30
MIN_MINUTES_PER_GAME = 5
MIN_GAMES = 10
HIGHLIGHTED_PLAYER_ID = "acuffda01"
NUM_RUNNERS_UP = 2
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


def get_picks(draft_year: int) -> pandas.DataFrame:
    r = BasketballReferenceScrapper.get_request(f"draft/NBA_{draft_year}.html")
    data = parse_table(r.content, "stats")
    data["pick"] = pandas.to_numeric(data["pick_overall"])
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
        picks = get_picks(draft_year)
        season_stats = get_per_game_stats(draft_year + 1)
        rookies.append(picks.merge(season_stats, on=["player_id", "season"]))
    data = pandas.concat(rookies, ignore_index=True)
    data.to_csv(CACHE_PATH, index=False)
    return data


def get_headshot(player_id: str):
    path = f"data/{player_id}.jpg"
    try:
        return pyplot.imread(path)
    except FileNotFoundError:
        pass
    r = BasketballReferenceScrapper.get_request(f"players/{player_id[0]}/{player_id}.html")
    meta = BeautifulSoup(r.content, "html.parser").find("div", id="meta")
    url = meta.find("img")["src"]
    r = BasketballReferenceScrapper.get_request(url.removeprefix(BasketballReferenceScrapper.BR_ORIGIN))
    with open(path, "wb") as f:
        f.write(r.content)
    return pyplot.imread(path)


def crop_to_ellipse(image, color: str, border_px: float):
    height, width = image.shape[:2]
    y, x = numpy.ogrid[:height, :width]
    radius = ((x - (width - 1) / 2) / (width / 2)) ** 2 + ((y - (height - 1) / 2) / (height / 2)) ** 2
    inner = (1 - border_px / (width / 2)) ** 2
    rgba = numpy.zeros((height, width, 4))
    rgba[..., :3] = image[..., :3] / 255
    rgba[..., 3] = radius <= 1
    rgba[(radius <= 1) & (radius > inner), :3] = to_rgb(color)
    return rgba


def add_headshot(ax, player_id: str, rank: int, bar: tuple, box: tuple, zoom: float, color: str):
    headshot = get_headshot(player_id)
    image = crop_to_ellipse(headshot, color, border_px=2 / zoom)
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(image, zoom=zoom),
            bar,
            xybox=box,
            boxcoords="data",
            frameon=False,
            pad=0,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.5},
        )
    )
    # OffsetImage zoom is in pixels at 72 dpi, so half the height in points is height * zoom / 2
    ax.annotate(
        f"#{rank}",
        xy=box,
        xytext=(0, headshot.shape[0] * zoom / 2 + 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=color,
        fontweight="bold",
        fontsize=10,
    )


def plot_results(data: pandas.DataFrame, runners_up: pandas.DataFrame):
    seaborn.set_theme()
    fig, ax = pyplot.subplots(figsize=(10, 5))
    counts = data["TSA"].floordiv(1).astype(int).value_counts().sort_index()
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 1), fill_value=0)
    highlighted_bin = int(data.loc[data["player_id"] == HIGHLIGHTED_PLAYER_ID, "TSA"].iloc[0] // 1)
    colors = ["red" if tsa == highlighted_bin else "steelblue" for tsa in counts.index]
    ax.bar(counts.index, counts.values, width=0.95, color=colors)
    # All headshots share one row, each right above its bar
    headshot_y = counts.max() * 0.3
    highlighted_rank = (data["TSA"] > data.loc[data["player_id"] == HIGHLIGHTED_PLAYER_ID, "TSA"].iloc[0]).sum() + 1
    add_headshot(
        ax,
        HIGHLIGHTED_PLAYER_ID,
        highlighted_rank,
        (highlighted_bin, counts[highlighted_bin]),
        (highlighted_bin, headshot_y),
        0.45,
        "red",
    )
    for _, player in runners_up.iterrows():
        tsa_bin = int(player["TSA"] // 1)
        add_headshot(
            ax,
            player["player_id"],
            (data["TSA"] > player["TSA"]).sum() + 1,
            (tsa_bin, counts[tsa_bin]),
            (tsa_bin, headshot_y),
            0.2,
            "steelblue",
        )
    fig.suptitle("Darius Acuff Jr. is outshooting every rookie of the last 30 years", fontsize=14, fontweight="bold", y=0.94)
    ax.set_title(
        f"First round picks, rookie seasons {min(DRAFT_YEARS) + 1}-{max(DRAFT_YEARS) + 1}, "
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
    data = data[
        (data["pick"] <= MAX_PICK)
        & (data["MP"] >= MIN_MINUTES_PER_GAME)
        & (data["G"] >= MIN_GAMES)
    ]
    highlighted = data[data["player_id"] == HIGHLIGHTED_PLAYER_ID]
    if len(highlighted) != 1:
        raise ValueError(f"{HIGHLIGHTED_PLAYER_ID} not found after filters")
    print(data.sort_values("TSA", ascending=False).head(10).to_string(index=False))
    others = data[data["player_id"] != HIGHLIGHTED_PLAYER_ID]
    plot_results(data, others.nlargest(NUM_RUNNERS_UP, "TSA"))


if __name__ == "__main__":
    main()
