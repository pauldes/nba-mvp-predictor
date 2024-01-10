import random
from time import sleep
from urllib.error import HTTPError
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import PercentFormatter
import pandas
import scipy
import seaborn
from scipy import stats

from nba_mvp_predictor import utils

def get_data():
    data = pandas.read_csv(
        "data/defense-2024-01-10.csv",
        sep=',',
        index_col='team',
        usecols=['team','luck_adjusted_defensive_rating','defensive_3ar','defensive_close_ar'],
        dtype={'team': str, 'luck_adjusted_defensive_rating': float, 'defensive_3ar': float, 'defensive_close_ar': float},
    )
    team_names = utils.get_dict_from_yaml(
        "./nba_mvp_predictor/team_names.yaml"
    )
    data.index = data.index.str.upper().map(team_names, na_action=None)
    data['luck_adjusted_defensive_rating_rank'] = data['luck_adjusted_defensive_rating'].rank(ascending=True)
    return data

def r2(x, y):
    regression = scipy.stats.linregress(x, y)
    return regression.slope, regression.intercept, regression.rvalue ** 2

def plot_data(data):
    cmap = seaborn.diverging_palette(h_neg=266, h_pos=10, s=99, l=50, sep=10, as_cmap=True)
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
    fig, ax = pyplot.subplots(figsize=(8, 6))
    a,b,r_squared = r2(data['defensive_close_ar'], data['defensive_3ar'])

    seaborn.scatterplot(
        data=data,
        x='defensive_close_ar',
        y='defensive_3ar',
        hue='luck_adjusted_defensive_rating', 
        palette=cmap,
        ax=ax,
        alpha=0.8,
        s=100,
        legend=None,
    )
    title = "Opponent 3PT shot attempt rate against close shot attempt rate"
    ax.set_title(
        f"{title}\n",
        fontsize=12,
        fontweight="bold",
        color=font_color,
    )
    pyplot.text(
        data.defensive_close_ar.max() + 1,
        (data.defensive_3ar.max() + data.defensive_3ar.min()) / 2,
        "@wontcalltimeout",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        color=font_color,
        alpha=0.5,
    )
    ax.axvline(
        x=data.defensive_close_ar.median(), 
        color=font_color, linestyle="--", linewidth=0.5,
        alpha=0.5,
    )
    ax.axhline(
        y=data.defensive_3ar.median(), 
        color=font_color, linestyle="--", linewidth=0.5,
        alpha=0.5,
    )

    ax.text(
        (data.defensive_close_ar.max() + data.defensive_close_ar.min()) / 2,
        data.defensive_3ar.max() + 1.2,
        f"Colors and ranks based on adjusted defensive ratings accounting for 3PT luck (©CraftedNBA)",
        fontsize=9,
        horizontalalignment="center",
        verticalalignment="top",
        color=font_color,
    )
    ax.set_xlabel("Opponent close shot attempt rate (<10 feet)", color=font_color, fontsize=10)
    ax.set_ylabel("Opponent 3PT shot attempt rate", color=font_color, fontsize=10)
    ax.xaxis.label.set_color(font_color)
    ax.yaxis.label.set_color(font_color)
    ax.tick_params(axis="x", colors="#d6d4d4", labelsize=9)
    ax.tick_params(axis="y", colors="#d6d4d4", labelsize=9)
    ax.set_facecolor(grid_background_color)
    ax.spines["bottom"].set_color(font_color)
    ax.spines["top"].set_color(font_color)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    def label_point(x, y, val, ax):
        a = pandas.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for _, point in a.iterrows():
            ax.text(
                point['x']+0.2, point['y'], str(point['val']),
                fontsize=8,
                horizontalalignment='left',
                verticalalignment='center',
                #fontweight="bold",
            )
    data['formatted_team_and_rank'] = data.apply(
        lambda row: f"{row.name} ({int(row.luck_adjusted_defensive_rating_rank)})",
        axis=1,
    )
    label_point(data.defensive_close_ar, data.defensive_3ar, data.formatted_team_and_rank, pyplot.gca())

    pyplot.tight_layout()
    fig.savefig("data/closear_against_3ar.png")

data = get_data()
print(data)
plot_data(data)

