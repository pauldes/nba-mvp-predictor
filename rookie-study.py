import argparse
import datetime
import math
import os

from bs4 import BeautifulSoup
from matplotlib import pyplot
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MultipleLocator
import numpy
import pandas
import seaborn

from nba_mvp_predictor.scrappers import BasketballReferenceScrapper

DRAFT_YEARS = range(1978, 2027)
MAX_PICK = 30
MIN_MINUTES_PER_GAME = 5
MIN_GAMES = 10
HIGHLIGHTED_PLAYER_ID = "acuffda01"
NUM_RUNNERS_UP = 2
PICKS_CACHE_PATH = "data/rookie_draft_picks.csv"
STATS_CACHE_PATH = "data/rookie_stats.csv"
BAR_COLOR = "#3478bd"
HIGHLIGHT_COLOR = "#f3bb4a"
MUTED_COLOR = "#999999"
GRID_COLOR = "#ececec"
TITLE_X = 0.02
RUNNER_UP_SPACING = 0.06
Y_TICK_STEP = 20
BOLD_FONT = FontProperties(fname="/System/Library/Fonts/Supplemental/Verdana Bold.ttf")
STATS = {
    "games": "G",
    "mp_per_g": "MP",
    "fga_per_g": "FGA",
    "fta_per_g": "FTA",
    "tov_per_g": "TOV",
    "ast_per_g": "AST",
}
METRICS = {
    "shot_attempts": {
        "formula": lambda data: data["FGA"] + 0.44 * data["FTA"],
        "title": "Darius Acuff Jr. takes more shots\nthan any rookie of the last 50 years",
        "xlabel": "True shot attempts per game",
    },
    "possessions": {
        "formula": lambda data: data["FGA"] + 0.44 * data["FTA"] + data["TOV"],
        "title": "Darius Acuff Jr. ends more possessions\nthan any rookie of the last 50 years",
        "xlabel": "Possessions used per game",
    },
}


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
    data = data.rename(columns=STATS)
    columns = list(STATS.values())
    data[columns] = data[columns].apply(pandas.to_numeric)
    data["season"] = season
    return data[["player_id", "season", *columns]]


def load_or_scrape(path: str, scrape) -> pandas.DataFrame:
    try:
        return pandas.read_csv(path)
    except FileNotFoundError:
        data = pandas.concat([scrape(draft_year) for draft_year in DRAFT_YEARS], ignore_index=True)
        data.to_csv(path, index=False)
        return data


def get_data() -> pandas.DataFrame:
    picks = load_or_scrape(PICKS_CACHE_PATH, get_picks)
    stats = load_or_scrape(STATS_CACHE_PATH, lambda draft_year: get_per_game_stats(draft_year + 1))
    return picks.merge(stats, on=["player_id", "season"])


def get_headshot(player_id: str):
    path = f"data/{player_id}.jpg"
    missing_path = f"data/{player_id}.nophoto"
    if os.path.exists(missing_path):
        return None
    try:
        return pyplot.imread(path)
    except FileNotFoundError:
        pass
    r = BasketballReferenceScrapper.get_request(f"players/{player_id[0]}/{player_id}.html")
    image = BeautifulSoup(r.content, "html.parser").find("div", id="meta").find("img")
    if image is None:
        open(missing_path, "w").close()
        return None
    r = BasketballReferenceScrapper.get_request(image["src"].removeprefix(BasketballReferenceScrapper.BR_ORIGIN))
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


def add_headshot(ax, headshot, rank: int, bar: tuple, box: tuple, zoom: float, color: str):
    image = crop_to_ellipse(headshot, color, border_px=2 / zoom)
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(image, zoom=zoom),
            bar,
            xybox=box,
            boxcoords="data",
            frameon=False,
            pad=0,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 2},
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
        fontproperties=BOLD_FONT,
        fontsize=60 * zoom,
    )


def plot_results(data: pandas.DataFrame, runners_up: list, metric: str):
    seaborn.set_theme(style="white", font="Verdana")
    fig, ax = pyplot.subplots(figsize=(10, 9), facecolor="white")
    ax.set_facecolor("white")
    counts = data["value"].floordiv(1).astype(int).value_counts().sort_index()
    counts = counts.reindex(range(counts.index.min(), counts.index.max() + 1), fill_value=0)
    highlighted_value = data.loc[data["player_id"] == HIGHLIGHTED_PLAYER_ID, "value"].iloc[0]
    highlighted_bin = int(highlighted_value // 1)
    colors = [HIGHLIGHT_COLOR if value == highlighted_bin else BAR_COLOR for value in counts.index]
    ax.bar(counts.index, counts.values, width=0.9, color=colors, zorder=2)
    # All headshots share one row: the highlighted one above its bar, runners-up spread to its left
    headshot_y = counts.max() * 0.35
    add_headshot(
        ax,
        get_headshot(HIGHLIGHTED_PLAYER_ID),
        (data["value"] > highlighted_value).sum() + 1,
        (highlighted_bin, counts[highlighted_bin]),
        (highlighted_bin, headshot_y),
        0.55,
        HIGHLIGHT_COLOR,
    )
    for i, (headshot, player) in enumerate(runners_up):
        value_bin = int(player["value"] // 1)
        add_headshot(
            ax,
            headshot,
            (data["value"] > player["value"]).sum() + 1,
            (value_bin, counts[value_bin]),
            (highlighted_bin - (i + 1.5) * RUNNER_UP_SPACING * len(counts), headshot_y),
            0.25,
            BAR_COLOR,
        )
    fig.text(
        TITLE_X,
        0.97,
        METRICS[metric]["title"],
        fontproperties=BOLD_FONT,
        fontsize=26,
        ha="left",
        va="top",
    )
    fig.text(
        TITLE_X,
        0.865,
        f"1st round picks, ≥{MIN_MINUTES_PER_GAME} MPG, ≥{MIN_GAMES} GP ({len(data)} rookies). "
        f"As of {datetime.date.today():%b %-d, %Y}.",
        fontsize=13,
        color=MUTED_COLOR,
        ha="left",
        va="top",
    )
    ax.set_xlabel(METRICS[metric]["xlabel"], fontsize=18, labelpad=10)
    ax.set_ylabel("Rookies", fontsize=18, labelpad=10)
    ax.set_xticks(counts.index)
    ax.tick_params(axis="both", labelsize=14, colors=MUTED_COLOR, length=0)
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
    ax.set_ylim(0, math.ceil(counts.max() / Y_TICK_STEP) * Y_TICK_STEP)
    ax.set_xlim(counts.index.min() - 0.7, counts.index.max() + 0.7)
    ax.grid(axis="both", color=GRID_COLOR, linewidth=1, zorder=0)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.text(
        0.99,
        0.98,
        "@wontcalltimeout",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="top",
        color="#666666",
    )
    pyplot.tight_layout(rect=(0, 0, 0.98, 0.835))
    fig.savefig(f"data/rookie_{metric}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", choices=METRICS.keys())
    metric = parser.parse_args().metric
    data = get_data()
    data["value"] = METRICS[metric]["formula"](data)
    data = data[
        (data["season"] - 1).isin(DRAFT_YEARS)
        & (data["pick"] <= MAX_PICK)
        & (data["MP"] >= MIN_MINUTES_PER_GAME)
        & (data["G"] >= MIN_GAMES)
    ]
    highlighted = data[data["player_id"] == HIGHLIGHTED_PLAYER_ID]
    if len(highlighted) != 1:
        raise ValueError(f"{HIGHLIGHTED_PLAYER_ID} not found after filters")
    print(data.sort_values("value", ascending=False).head(10).to_string(index=False))
    others = data[data["player_id"] != HIGHLIGHTED_PLAYER_ID].sort_values("value", ascending=False)
    # Players without a photo are skipped, the next best fills their slot
    runners_up = []
    for _, player in others.iterrows():
        headshot = get_headshot(player["player_id"])
        if headshot is not None:
            runners_up.append((headshot, player))
        if len(runners_up) == NUM_RUNNERS_UP:
            break
    plot_results(data, runners_up, metric)


if __name__ == "__main__":
    main()
