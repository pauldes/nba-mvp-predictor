from dataclasses import dataclass
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
from numpy import std, random, average
from matplotlib.font_manager import fontManager, FontProperties

import seaborn

TRIALS = 100_000
THREE_PERCENTAGE = 0.36
USE_PERCENTAGES = True


@dataclass
class SimulationResult:
    average_outcome: float
    std_outcome: float


def print_measures(simulation_result: SimulationResult):
    std_outcome = round(simulation_result.std_outcome, 2)
    average_outcome = round(simulation_result.average_outcome, 2)
    return f"{average_outcome} ± {std_outcome}"


def simulate(
    threes_attempted,
    threes_percentage,
    per_100=False,
) -> list[int]:
    outcomes = []
    for _ in range(TRIALS):
        threes_made = random.binomial(threes_attempted, threes_percentage)
        if per_100:
            outcome = round(100 * threes_made / threes_attempted)
        else:
            outcome = threes_made
        outcomes.append(outcome)
    return outcomes


def analyze_outcomes(outcomes) -> SimulationResult:
    std_outcome = std(outcomes)
    average_outcome = average(outcomes)
    return SimulationResult(average_outcome, std_outcome)


def plot_shot_distributions(
    threes_attempted, averages, stds, threes_percentage
):
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
    title = "Expected 3PT% against 3FGA"
    ax.set_title(
        f"{title}\n",
        fontsize=12,
        fontweight="bold",
        color=font_color,
    )
    ax.text(
        threes_attempted[round((len(averages) - 1)/2)],
        73,
        f"For a real {round(100*threes_percentage,1)}% 3PT shooter",
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
        threes_attempted[-1] + 80,
        averages[-1],
        "@wontcalltimeout",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        color=font_color,
        alpha=0.5,
    )
    ax.set_xlabel("3FGA", color=font_color, fontsize=10)
    ax.set_ylabel("3PT%", color=font_color, fontsize=10)
    ax.xaxis.label.set_color(font_color)
    ax.yaxis.label.set_color(font_color)
    ax.tick_params(axis="x", colors="#d6d4d4", labelsize=9)
    ax.tick_params(axis="y", colors="#d6d4d4", labelsize=9)
    ax.set_facecolor(grid_background_color)
    ax.spines["bottom"].set_color(font_color)
    ax.spines["top"].set_color(font_color)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(0, max(threes_attempted))
    fig.savefig(f"data/3pt_shooting_variance.png")


# Simulate for different shot distributions
min_shots = 10
max_shots = 1000
averages = []
stds = []
threes_attempts = []
for threes_attempted in range(min_shots, max_shots + 1):
    result = analyze_outcomes(
        simulate(
            threes_attempted=threes_attempted,
            threes_percentage=THREE_PERCENTAGE,
            per_100=USE_PERCENTAGES,
        )
    )
    print(f"{threes_attempted} 3s: {print_measures(result)}")
    averages.append(result.average_outcome)
    stds.append(result.std_outcome)
    threes_attempts.append(threes_attempted)
plot_shot_distributions(
    threes_attempts,
    averages,
    stds,
    THREE_PERCENTAGE,
)