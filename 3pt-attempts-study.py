import datetime
import math

from bs4 import BeautifulSoup
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
import pandas
import seaborn

from nba_mvp_predictor.scrappers import BasketballReferenceScrapper

NUM_CURRENT_LEADERS = 1
NUM_ALL_TIME_LEADERS = 3
SHOW_FIRST_N_GAMES = 30
CURRENT_SEASON = 2027
BAR_COLOR = "#3478bd"
HIGHLIGHT_COLOR = "#f3bb4a"
MUTED_COLOR = "#999999"
GRID_COLOR = "#ececec"
WATERMARK_COLOR = "#666666"
TITLE_X = 0.02
Y_TICK_STEP = 50
BOLD_FONT = FontProperties(fname="/System/Library/Fonts/Supplemental/Verdana Bold.ttf")


def get_player_id(cell) -> str:
    return cell.a["href"].split("/")[-1].removesuffix(".html")


def get_current_leaders(top_n: int, current_season: int) -> pandas.DataFrame:
    r = BasketballReferenceScrapper.get_request(f"leagues/NBA_{current_season}_totals.html")
    table = BeautifulSoup(r.content, "html.parser").find("table", id="totals_stats")
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = {cell.get("data-stat"): cell for cell in tr.find_all(["th", "td"])}
        player = cells.get("name_display")
        if player is None or player.a is None:
            continue
        rows.append(
            {
                "Player": player.get_text(),
                "player_id": get_player_id(player),
                "3PA": int(cells["fg3a"].get_text()),
            }
        )
    # Traded players have their season total row first, then one row per team
    data = pandas.DataFrame(rows).drop_duplicates(subset="player_id", keep="first")
    data = data.sort_values("3PA", ascending=False).head(top_n)
    data["Season"] = current_season
    return data


def get_game_logs(player_id: str, season: int) -> pandas.DataFrame:
    r = BasketballReferenceScrapper.get_request(f"players/{player_id[0]}/{player_id}/gamelog/{season}")
    table = BeautifulSoup(r.content, "html.parser").find("table", id="player_game_log_reg")
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = {cell.get("data-stat"): cell.get_text() for cell in tr.find_all(["th", "td"])}
        # Header rows repeat inside the body, and games not played have no shooting stats
        if not cells.get("ranker", "").isnumeric() or "fg3a" not in cells:
            continue
        rows.append({"Game": int(cells["ranker"]), "3PA": int(cells["fg3a"])})
    return pandas.DataFrame(rows).sort_values("Game")


def get_leaders(top_n: int) -> pandas.DataFrame:
    r = BasketballReferenceScrapper.get_request("leaders/fg3a_season.html")
    table = BeautifulSoup(r.content, "html.parser").find("table", id="stats_NBA")
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 4 or cells[1].a is None:
            continue
        rows.append(
            {
                "Player": cells[1].get_text().removesuffix("*"),
                "player_id": get_player_id(cells[1]),
                "3PA": int(cells[2].get_text()),
                "Season": cells[3].get_text(),
            }
        )
    data = pandas.DataFrame(rows).sort_values("3PA", ascending=False).head(top_n).reset_index(drop=True)
    data["Rank"] = data.index + 1
    return data

def nudge_labels(values: list, min_gap: float) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    nudged = list(values)
    for previous, current in zip(order, order[1:]):
        nudged[current] = max(nudged[current], nudged[previous] + min_gap)
    return nudged


def plot_results(data: pandas.DataFrame, current_player: str):
    seaborn.set_theme(style="white", font="Verdana")
    fig, ax = pyplot.subplots(figsize=(10, 9), facecolor="white")
    ax.set_facecolor("white")
    data = data[data["Game"] <= SHOW_FIRST_N_GAMES]
    labels = []
    for player_season, line in data.groupby("PlayerSeason", sort=False):
        current = line["Current"].iloc[0]
        ax.plot(
            line["Game"],
            line["Cumulative 3PA"],
            color=HIGHLIGHT_COLOR if current else BAR_COLOR,
            linewidth=5 if current else 2.5,
            zorder=3 if current else 2,
        )
        last = line.iloc[-1]
        labels.append((player_season, last["Game"], last["Cumulative 3PA"], current))
    y_max = math.ceil(data["Cumulative 3PA"].max() / Y_TICK_STEP) * Y_TICK_STEP
    label_ys = nudge_labels([label[2] for label in labels], min_gap=y_max * 0.04)
    for (text, x, _, current), y in zip(labels, label_ys):
        ax.text(
            x + 0.5,
            y,
            text,
            fontproperties=BOLD_FONT if current else None,
            fontsize=16 if current else 13,
            color=HIGHLIGHT_COLOR if current else BAR_COLOR,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
            zorder=4,
        )
    fig.text(
        TITLE_X,
        0.97,
        f"{current_player} is shooting 3s\nat a record pace",
        fontproperties=BOLD_FONT,
        fontsize=26,
        ha="left",
        va="top",
    )
    fig.text(
        TITLE_X,
        0.85,
        f"Total 3PA by game played. All-time leaders at their season average. "
        f"As of {datetime.date.today():%b %-d, %Y}.",
        fontsize=13,
        color=MUTED_COLOR,
        ha="left",
        va="top",
    )
    ax.set_xlabel("Games played", fontsize=18, labelpad=10)
    ax.set_ylabel("3-point attempts", fontsize=18, labelpad=10)
    ax.tick_params(axis="both", labelsize=14, colors=MUTED_COLOR, length=0)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
    ax.set_xlim(0, SHOW_FIRST_N_GAMES + 7)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(0, SHOW_FIRST_N_GAMES + 1, 5))
    ax.grid(axis="both", color=GRID_COLOR, linewidth=1, zorder=0)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    # Lines rise to the top right, so the watermark goes to the empty top left
    ax.text(
        0.01,
        0.98,
        "@wontcalltimeout",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="top",
        color=WATERMARK_COLOR,
    )
    pyplot.tight_layout(rect=(0, 0, 0.98, 0.82))
    fig.savefig("data/3pt_attempts.png")

def format_season(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def main():
    use_average_for_all_time = True
    data = []
    current_leaders = get_current_leaders(top_n=NUM_CURRENT_LEADERS, current_season=CURRENT_SEASON)
    for leader in current_leaders.to_dict(orient="records"):
        game_log = get_game_logs(leader["player_id"], leader["Season"])
        game_log["Cumulative 3PA"] = game_log["3PA"].cumsum()
        game_log["PlayerSeason"] = f"{leader['Player'].split(' ')[-1]} {format_season(leader['Season'])}"
        game_log['Current'] = True
        data.append(game_log)
    leaders = get_leaders(top_n=NUM_ALL_TIME_LEADERS)
    leaders['Season'] = leaders['Season'].str[:4].astype(int) + 1
    for leader in leaders.to_dict(orient="records"):
        game_log = get_game_logs(leader["player_id"], leader["Season"])
        if use_average_for_all_time:
            avg_3pa = leader["3PA"] / 82
            game_log["3PA"] = avg_3pa
        game_log["Cumulative 3PA"] = game_log["3PA"].cumsum()
        game_log["PlayerSeason"] = (
            f"{leader['Player'].split(' ')[-1]} {format_season(leader['Season'])} (#{leader['Rank']} all-time)"
        )
        game_log['Current'] = False
        data.append(game_log)
    data = pandas.concat(data)
    plot_results(data, current_leaders["Player"].iloc[0])

if __name__ == "__main__":
    main()