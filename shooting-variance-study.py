from dataclasses import dataclass
from matplotlib import pyplot
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, PercentFormatter
from numpy import std, random, average
from matplotlib.font_manager import fontManager, FontProperties

import pandas
import seaborn

TRIALS = 20_000


@dataclass
class SimulationResult:
    average_outcome: float
    std_outcome: float


def print_measures(simulation_result: SimulationResult):
    std_outcome = round(simulation_result.std_outcome, 2)
    average_outcome = round(simulation_result.average_outcome, 2)
    return f"{average_outcome} ± {std_outcome}"


def simulate(
    twos_attempted,
    threes_attempted,
    twos_percentage,
    threes_percentage,
    per_100=False,
) -> list[int]:
    outcomes = []
    for _ in range(TRIALS):
        twos_made = random.binomial(twos_attempted, twos_percentage)
        threes_made = random.binomial(threes_attempted, threes_percentage)
        if per_100:
            outcome = (
                100
                * (twos_made * 2 + threes_made * 3)
                / (twos_attempted + threes_attempted)
            )
        else:
            outcome = twos_made * 2 + threes_made * 3
        outcomes.append(outcome)
    return outcomes


def analyze_outcomes(outcomes) -> SimulationResult:
    std_outcome = std(outcomes)
    average_outcome = average(outcomes)
    return SimulationResult(average_outcome, std_outcome)


def draw_distribution(outcomes):
    fig, ax = pyplot.subplots(figsize=(8, 6))
    seaborn.kdeplot(outcomes, ax=ax)
    fig.savefig(f"data/dist.png")


def load_yearly_data():
    data = pandas.read_csv(
        "data.csv",
        usecols=["3PA", "3P%", "FGA", "eFG%", "Season"],
        header=1,
        index_col="Season",
    ).fillna(0.0)
    data["2PA"] = data["FGA"] - data["3PA"]
    data["2P%"] = (data["eFG%"] * data["FGA"] - 1.5 * data["3P%"] * data["3PA"]) / data[
        "2PA"
    ]
    data['3PT_share'] = data['3PA'] / data['FGA']
    return data

def load_games_data():
    data = pandas.read_csv(
        "data/games_details.csv",
        usecols=["GAME_ID", "TEAM_ID", "FG3A", "FG3M", "FGA", "FGM"],
        header=0,
    ).fillna(0.0)
    data = data.groupby(["GAME_ID", "TEAM_ID"]).sum()
    data = data.rename(columns={"FG3A": "3PA", "FG3M": "3PM", "FGA": "FGA", "FGM": "FGM"})
    data["2PA"] = data["FGA"] - data["3PA"]
    data["2PM"] = data["FGM"] - data["3PM"]
    data["2P%"] = (data["2PM"] / data["2PA"]).round(3)
    data["3P%"] = (data["3PM"] / data["3PA"]).round(3)
    data["PTs"] = 2 * data["2PM"] + 3 * data["3PM"]
    data["PTs_per_FGA"] = (data["PTs"] / data["FGA"]).round(3)
    data['3PT_share'] = (data['3PA'] / data['FGA']).round(3)
    return data


def plot_shot_distributions(
    threes_share, averages, stds, threes_percentage, twos_percentage
):
    threes_attempted = [int(100 * share) for share in threes_share]
    seaborn.set_style("darkgrid")
    font_color = "#efecec"
    grid_background_color = "#525050"
    chart_background_color = "#2e2d2d"
    path = "data/Roboto-Regular.ttf"
    fontManager.addfont(path)
    prop = FontProperties(fname=path)
    seaborn.set(font=prop.get_name())
    seaborn.set(
        rc={
            "axes.facecolor": grid_background_color,
            "figure.facecolor": chart_background_color,
            "axes.labelcolor": font_color,
            "axes.edgecolor": chart_background_color,
            "text.color": font_color,
            "xtick.color": font_color,
            "ytick.color": font_color,
            "grid.color": chart_background_color,
            "grid.linewidth": 0.3,
            "axes.grid": True,
            "axes.linewidth": 0.1,
            "patch.linewidth": 0.1,
        }
    )
    fig, ax = pyplot.subplots(figsize=(8, 4))
    seaborn.lineplot(x=threes_attempted, y=averages, ax=ax, color="white")
    for i in [2, 1]:
        stds_lower = [average - i * std for average, std in zip(averages, stds)]
        stds_upper = [average + i * std for average, std in zip(averages, stds)]
        ax.fill_between(
            threes_attempted,
            stds_lower,
            stds_upper,
            alpha=(0.6 - 0.2 * i),
            # color="lightsteelblue",
            color="steelblue",
        )
    title = "Expected points against share of 3pt shots"
    ax.set_title(
        f"{title}\n",
        fontsize=12,
        fontweight="bold",
        color=font_color,
    )
    ax.text(
        50,
        130,
        f"{total_shots} field goal attempts • {round(100*twos_percentage,1)} 2PT% • {round(100*threes_percentage,1)} 3PT% (last 5 seasons avg.)",
        fontsize=9,
        horizontalalignment="center",
        verticalalignment="top",
        color=font_color,
    )
    x_68 = int(-(len(averages)*10/100))
    x_95 = int(-(len(averages)*30/100))
    ax.text(
        threes_attempted[x_68] + 1,
        averages[x_68] + 0.5 * stds[x_68],
        f"68% of outcomes\n(±1σ)",
        fontsize=9,
        horizontalalignment="left",
        verticalalignment="center",
        color=font_color,
        alpha=0.6,
    )
    ax.text(
        threes_attempted[x_95] + 1,
        averages[x_95] + 1.5 * stds[x_95],
        f"95% of outcomes\n(±2σ)",
        fontsize=9,
        horizontalalignment="left",
        verticalalignment="center",
        color=font_color,
        alpha=0.6,
    )
    ax.arrow(
        threes_attempted[x_68], 
        averages[x_68],
        0,
        1 * stds[x_68],
        ls="--",
        linewidth=1,
        color=font_color,
        alpha=0.6,
        head_width=1, head_length=1,
        length_includes_head=True
    )
    ax.arrow(
        threes_attempted[x_68], 
        averages[x_68],
        0,
        -1 * stds[x_68],
        ls="--",
        linewidth=1,
        color=font_color,
        alpha=0.6,
        head_width=1, head_length=1,
        length_includes_head=True
    )
    ax.arrow(
        threes_attempted[x_95], 
        averages[x_95],
        0,
        2 * stds[x_95],
        ls="--",
        linewidth=1,
        color=font_color,
        alpha=0.6,
        head_width=1, head_length=1,
        length_includes_head=True
    )
    ax.arrow(
        threes_attempted[x_95], 
        averages[x_95],
        0,
        -2 * stds[x_95],
        ls="--",
        linewidth=1,
        color=font_color,
        alpha=0.6,
        head_width=1, head_length=1,
        length_includes_head=True
    )
    pyplot.text(
        110,
        100,
        "@wontcalltimeout",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        color=font_color,
        alpha=0.5,
    )
    ax.set_xlabel("Share of 3PT shots", color=font_color, fontsize=10)
    ax.set_ylabel("Expected points", color=font_color, fontsize=10)
    ax.xaxis.label.set_color(font_color)
    ax.yaxis.label.set_color(font_color)
    ax.tick_params(axis="x", colors="#d6d4d4", labelsize=9)
    ax.tick_params(axis="y", colors="#d6d4d4", labelsize=9)
    ax.set_facecolor(grid_background_color)
    ax.spines["bottom"].set_color(font_color)
    ax.spines["top"].set_color(font_color)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.xaxis.set_major_locator(MultipleLocator(10))  # show every 5th tick
    ax.yaxis.set_major_locator(MultipleLocator(5))  # show every 5th tick
    fig.savefig(f"data/shot_distributions.png")


# Get simulation results by year
yearly_data = load_yearly_data()
yearly_data["result"] = yearly_data.apply(
    lambda row: print_measures(
        analyze_outcomes(simulate(row["2PA"], row["3PA"], row["2P%"], row["3P%"]))
    ),
    axis=1,
)
yearly_data["result_per100"] = yearly_data.apply(
    lambda row: print_measures(
        analyze_outcomes(
            simulate(row["2PA"], row["3PA"], row["2P%"], row["3P%"], per_100=True)
        )
    ),
    axis=1,
)
print(yearly_data.sort_index(ascending=False).head(30))

# Get 2pt and 3pt percentages for the last 5 seasons
twos_percentage_last_5_seasons = (
    yearly_data.sort_index(ascending=False).head(5)["2P%"].mean()
)
threes_percentage_last_5_seasons = (
    yearly_data.sort_index(ascending=False).head(5)["3P%"].mean()
)
fga_last_5_seasons = yearly_data.sort_index(ascending=False).head(5)["FGA"].mean()
print(
    "Last 5 seasons:",
    twos_percentage_last_5_seasons,
    "2PT%",
    threes_percentage_last_5_seasons,
    "3PT%",
)


# Simulate for different shot distributions
total_shots = round(fga_last_5_seasons)
averages = []
stds = []
threes_share = []
for i in range(0, total_shots + 1):
    twos_attempted = i
    threes_attempted = total_shots - i
    result = analyze_outcomes(
        simulate(
            twos_attempted,
            threes_attempted,
            twos_percentage_last_5_seasons,
            threes_percentage_last_5_seasons,
            per_100=True,
        )
    )
    print(f"{twos_attempted} 2s, {threes_attempted} 3s: {print_measures(result)}")
    averages.append(result.average_outcome)
    stds.append(result.std_outcome)
    threes_share.append(threes_attempted / total_shots)
plot_shot_distributions(
    threes_share,
    averages,
    stds,
    threes_percentage_last_5_seasons,
    twos_percentage_last_5_seasons,
)


# Get real-world data
real_world_data = load_games_data()
print(real_world_data.sample(10))
real_world_data['3PT_share_rounded'] = real_world_data['3PT_share'].round(1)
aggregated_real_world_data = real_world_data.groupby('3PT_share_rounded').agg(
    {'PTs_per_FGA': ['mean', 'std', 'count']}
)
aggregated_real_world_data = aggregated_real_world_data[aggregated_real_world_data[('PTs_per_FGA', 'count')] > 30]
print(aggregated_real_world_data.sort_values(by=('3PT_share_rounded'), ascending=False))

threes_share = aggregated_real_world_data.index.values
averages = aggregated_real_world_data[('PTs_per_FGA', 'mean')].values
stds = aggregated_real_world_data[('PTs_per_FGA', 'std')].values
threes_percentage = 0.50
twos_percentage = 0.50
#plot_shot_distributions(threes_share, averages, stds, threes_percentage, twos_percentage)