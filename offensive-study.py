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

YEARS = range(2011, 2024)
TEAMS = ['BOS', 'BRK', 'NYK', 'PHI', 'TOR', 'CHI', 'CLE', 'DET', 'IND', 'MIL', 'ATL', 'CHO', 'MIA', 'ORL', 'WAS', 'DEN', 'MIN', 'OKC', 'POR', 'UTA', 'GSW', 'LAC', 'LAL', 'PHO', 'SAC', 'DAL', 'HOU', 'MEM', 'NOP', 'SAS']

def get_ratings_for_team_year(team, year):
    sleep(3 + random.random())
    url = f"https://www.basketball-reference.com/teams/{team}/{year}/gamelog-advanced/"
    print('Requesting', url, '...')
    data = pandas.read_html(
        url,
        attrs={'id':'tgl_advanced'},
        header=1,
    )[0][['G', 'Date', 'ORtg', 'DRtg', '3PAr']].set_index('G', drop=True)
    data = data[data.index.isin([str(i) for i in range(1, 83)])]
    data.index = data.index.astype(int)
    data['ORtg'] = data['ORtg'].astype(float)
    data['DRtg'] = data['DRtg'].astype(float)
    data['3PAr'] = data['3PAr'].astype(float)
    return data

def get_3par_for_team_year(team, year):
    sleep(3 + random.random())
    url = f"https://www.basketball-reference.com/teams/{team}/{year}.html"
    print('Requesting', url, '...')
    data = pandas.read_html(
        url,
        attrs={'id':'totals'},
        header=0,
    )[0].set_index('Player', drop=True)[['3PA', 'FGA']]
    data = data[data.index.isna()].iloc[0].to_dict()
    three_ar = data['3PA'] / data['FGA']
    return round(three_ar, 3)

def get_data():
    three_ars = []
    off_ratings = []
    for year in YEARS:
        print(three_ars)
        print(off_ratings)
        for team in TEAMS:
            print(f'Getting data for {team} in {year}...')
            try:
                three_ar = get_3par_for_team_year(team, year)
                ratings = get_ratings_for_team_year(team, year)
                off_rating_std = ratings['ORtg'].std().round(3)
                three_ars.append(three_ar)
                off_ratings.append(off_rating_std)
            except HTTPError as http_error:
                status_code = http_error.code
                if status_code == 404:
                    print('No data for', team, year)
                else:
                    raise http_error
    return three_ars, off_ratings


def r2(x, y):
    regression = scipy.stats.linregress(x, y)
    return regression.slope, regression.intercept, regression.rvalue ** 2

def plot_data(three_ars, off_ratings, years, start_at_year=None, groupby_year=False):
    cmap = seaborn.cubehelix_palette(rot=-.2, as_cmap=True)
    data = pandas.DataFrame({
        '3PAr': three_ars,
        'OffRtgStd': off_ratings,
        'Year': years,
    })
    data['3PAr'] = 100 * data['3PAr']
    if start_at_year:
        data = data[data['Year'] >= start_at_year]
    if groupby_year:
        data = data.groupby('Year').mean().reset_index()

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
    a,b,r_squared = r2(data['3PAr'], data['OffRtgStd'])

    seaborn.scatterplot(
        data=data,
        x='3PAr',
        y='OffRtgStd',
        hue='Year', 
        palette=cmap,
        ax=ax,
        alpha=0.8,
        s=100,
        legend='full',
    )
    seaborn.regplot(
        data=data,
        x='3PAr',
        y='OffRtgStd',
        ax=ax,
        robust=False,
        scatter=False,
        color='#d6d4d4',
        line_kws={'linewidth': 1, 'linestyle': '--'},
        ci=None,
        #alpha=0.5,
    )
    title = "Offensive rating standard deviation against 3PT attempt rate"
    ax.set_title(
        f"{title}\n",
        fontsize=12,
        fontweight="bold",
        color=font_color,
    )
    pyplot.text(
        55,
        11,
        "@wontcalltimeout",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        color=font_color,
        alpha=0.5,
    )
    ax.text(
        50,
        50*a + b,
        f"R² = {r_squared:.2f}",
        fontsize=9,
        horizontalalignment="center",
        verticalalignment="top",
        color=font_color,
        rotation=6,
    )
    ax.text(
        32,
        14.95,
        f"10 last seasons ({data['Year'].min()} to {data['Year'].max()})",
        fontsize=9,
        horizontalalignment="center",
        verticalalignment="top",
        color=font_color,
    )
    ax.set_xlabel("Share of 3PT shots", color=font_color, fontsize=10)
    ax.set_ylabel("Offensive rating standard deviation", color=font_color, fontsize=10)
    ax.xaxis.label.set_color(font_color)
    ax.yaxis.label.set_color(font_color)
    ax.tick_params(axis="x", colors="#d6d4d4", labelsize=9)
    ax.tick_params(axis="y", colors="#d6d4d4", labelsize=9)
    ax.set_facecolor(grid_background_color)
    ax.spines["bottom"].set_color(font_color)
    ax.spines["top"].set_color(font_color)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    #pyplot.legend(loc='lower center')
    pyplot.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
    pyplot.setp(ax.get_legend().get_title(), fontsize='8') # for legend title
    pyplot.tight_layout()
    fig.savefig("data/offrtg_std_against_3ar.png")

#three_ars, off_ratings = get_data()

three_ars = [0.18, 0.303, 0.184, 0.162, 0.215, 0.224, 0.189, 0.244, 0.216, 0.222, 0.235, 0.328, 0.172, 0.258, 0.223, 0.212, 0.228, 0.191, 0.248, 0.23, 0.22, 0.271, 0.183, 0.274, 0.264, 0.136, 0.261, 0.194, 0.288, 0.175, 0.209, 0.205, 0.237, 0.175, 0.198, 0.224, 0.249, 0.198, 0.346, 0.196, 0.243, 0.262, 0.252, 0.255, 0.153, 0.248, 0.269, 0.209, 0.238, 0.228, 0.271, 0.24, 0.157, 0.257, 0.215, 0.269, 0.354, 0.209, 0.249, 0.189, 0.229, 0.217, 0.245, 0.232, 0.286, 0.285, 0.223, 0.223, 0.217, 0.22, 0.244, 0.284, 0.206, 0.239, 0.265, 0.303, 0.21, 0.243, 0.236, 0.349, 0.166, 0.264, 0.251, 0.301, 0.302, 0.258, 0.285, 0.222, 0.236, 0.222, 0.235, 0.231, 0.316, 0.292, 0.235, 0.246, 0.278, 0.245, 0.271, 0.29, 0.237, 0.291, 0.291, 0.291, 0.3, 0.218, 0.274, 0.33, 0.171, 0.193, 0.257, 0.28, 0.24, 0.24, 0.319, 0.302, 0.269, 0.334, 0.29, 0.255, 0.223, 0.321, 0.226, 0.262, 0.235, 0.203, 0.284, 0.179, 0.262, 0.316, 0.274, 0.311, 0.322, 0.22, 0.291, 0.204, 0.296, 0.392, 0.184, 0.233, 0.269, 0.293, 0.218, 0.256, 0.327, 0.287, 0.244, 0.352, 0.303, 0.27, 0.189, 0.336, 0.348, 0.221, 0.255, 0.282, 0.277, 0.202, 0.275, 0.332, 0.297, 0.362, 0.324, 0.29, 0.302, 0.26, 0.339, 0.37, 0.222, 0.277, 0.223, 0.393, 0.371, 0.279, 0.349, 0.289, 0.256, 0.399, 0.263, 0.272, 0.29, 0.309, 0.335, 0.314, 0.3, 0.284, 0.329, 0.249, 0.295, 0.322, 0.327, 0.359, 0.329, 0.295, 0.255, 0.291, 0.366, 0.462, 0.316, 0.308, 0.281, 0.357, 0.411, 0.266, 0.344, 0.377, 0.35, 0.379, 0.333, 0.284, 0.297, 0.363, 0.314, 0.358, 0.342, 0.31, 0.357, 0.261, 0.345, 0.324, 0.357, 0.339, 0.314, 0.329, 0.32, 0.278, 0.382, 0.502, 0.317, 0.319, 0.282, 0.381, 0.403, 0.334, 0.342, 0.379, 0.295, 0.332, 0.394, 0.292, 0.419, 0.403, 0.378, 0.368, 0.36, 0.37, 0.348, 0.315, 0.347, 0.339, 0.394, 0.384, 0.295, 0.342, 0.335, 0.321, 0.422, 0.519, 0.342, 0.324, 0.286, 0.386, 0.423, 0.318, 0.36, 0.421, 0.396, 0.362, 0.381, 0.317, 0.428, 0.398, 0.399, 0.419, 0.364, 0.358, 0.344, 0.433, 0.353, 0.374, 0.414, 0.355, 0.375, 0.358, 0.361, 0.395, 0.457, 0.501, 0.346, 0.403, 0.318, 0.409, 0.413, 0.347, 0.347, 0.444, 0.383, 0.347, 0.385, 0.372, 0.404, 0.382, 0.422, 0.432, 0.356, 0.319, 0.383, 0.413, 0.399, 0.448, 0.488, 0.439, 0.4, 0.363, 0.392, 0.376, 0.436, 0.459, 0.342, 0.342, 0.314, 0.425, 0.359, 0.428, 0.376, 0.375, 0.332, 0.387, 0.391, 0.395, 0.43, 0.39, 0.418, 0.422, 0.417, 0.356, 0.416, 0.454, 0.419, 0.422, 0.468, 0.456, 0.391, 0.388, 0.354, 0.377, 0.439, 0.448, 0.346, 0.365, 0.345, 0.48, 0.397, 0.4, 0.389, 0.351, 0.333, 0.371, 0.372, 0.413, 0.446, 0.331, 0.36, 0.408, 0.361, 0.365, 0.361, 0.381, 0.369, 0.413, 0.421, 0.479, 0.387, 0.351, 0.362, 0.423, 0.487, 0.359, 0.372, 0.344, 0.348]
off_ratings = [11.878, 12.556, 11.171, 10.836, 10.739, 10.981, 10.314, 12.893, 11.357, 11.247, 10.531, 11.601, 10.263, 11.345, 10.437, 10.856, 11.316, 11.132, 11.068, 10.272, 10.603, 9.775, 10.133, 9.9, 9.918, 9.657, 10.011, 10.405, 12.59, 11.023, 11.928, 11.217, 10.211, 13.452, 11.448, 12.475, 12.418, 12.827, 14.426, 11.732, 10.598, 10.867, 8.861, 11.773, 9.111, 10.407, 8.979, 8.925, 11.73, 11.162, 10.828, 9.206, 10.843, 11.841, 10.052, 12.251, 11.61, 9.9, 10.504, 11.057, 10.926, 11.56, 10.541, 9.335, 11.303, 10.165, 11.584, 11.54, 9.872, 9.726, 10.476, 11.687, 11.262, 10.846, 11.148, 9.869, 11.326, 12.31, 11.899, 13.284, 9.441, 11.25, 10.156, 10.025, 12.787, 10.308, 9.273, 11.42, 11.609, 9.513, 9.644, 11.118, 10.736, 9.79, 8.583, 10.568, 11.115, 10.57, 10.846, 10.046, 11.429, 10.753, 11.218, 11.221, 10.95, 9.898, 10.512, 10.288, 9.743, 10.058, 9.933, 9.324, 12.201, 10.858, 10.144, 11.958, 10.778, 12.581, 10.441, 10.532, 8.932, 12.155, 11.915, 10.5, 9.421, 10.313, 12.19, 9.709, 12.136, 9.826, 9.535, 10.198, 10.085, 9.415, 12.149, 10.164, 12.651, 9.658, 9.804, 12.133, 11.117, 8.448, 10.349, 10.988, 10.408, 10.373, 10.881, 12.715, 11.418, 9.823, 9.43, 10.562, 10.685, 10.71, 10.928, 10.883, 11.696, 11.166, 10.747, 10.683, 10.144, 10.901, 10.571, 11.513, 10.665, 10.066, 10.818, 10.907, 11.992, 11.747, 11.277, 8.687, 10.925, 9.03, 10.4, 11.34, 10.557, 13.896, 11.431, 10.146, 11.705, 11.0, 10.827, 11.133, 11.78, 9.92, 11.956, 9.565, 10.384, 11.785, 12.161, 11.606, 11.194, 11.37, 10.178, 10.403, 12.12, 10.356, 11.711, 10.65, 10.316, 10.004, 9.486, 10.744, 9.722, 10.918, 11.416, 11.939, 10.222, 11.178, 10.045, 10.29, 12.676, 10.656, 10.731, 10.7, 12.604, 9.586, 11.69, 10.475, 12.717, 12.269, 10.62, 10.295, 10.781, 9.028, 10.115, 10.301, 10.839, 10.085, 11.484, 9.99, 10.456, 9.898, 10.729, 9.991, 11.1, 10.722, 13.13, 10.946, 9.763, 11.698, 12.053, 11.213, 12.152, 9.406, 11.813, 9.404, 8.954, 12.261, 12.177, 12.834, 11.414, 10.532, 10.025, 10.279, 10.734, 12.002, 10.103, 11.035, 10.887, 10.481, 10.866, 11.272, 11.198, 11.328, 10.768, 9.055, 12.026, 11.411, 10.153, 12.855, 11.416, 10.691, 11.808, 11.849, 11.315, 10.212, 11.781, 10.677, 11.504, 10.646, 12.17, 11.097, 11.434, 10.7, 10.831, 12.778, 10.671, 9.781, 9.727, 11.214, 10.52, 12.172, 11.368, 10.98, 10.424, 10.843, 10.971, 10.701, 9.707, 10.839, 8.946, 11.363, 10.518, 11.152, 9.878, 10.722, 10.634, 11.197, 9.748, 12.671, 11.828, 10.428, 10.998, 10.997, 11.395, 12.297, 11.353, 11.29, 10.646, 14.085, 10.769, 10.024, 10.582, 11.698, 11.925, 9.996, 11.584, 10.925, 11.259, 11.574, 12.737, 11.83, 10.314, 11.167, 12.366, 11.99, 10.829, 13.115, 10.585, 11.452, 13.008, 10.959, 10.41, 12.153, 13.645, 11.02, 12.165, 10.42, 11.36, 12.348, 11.389, 10.332, 11.47, 10.226, 11.683, 9.496, 10.604, 11.514, 11.624, 10.304, 10.834, 10.664, 9.45, 11.239, 10.408, 11.495, 10.318, 12.63, 8.948, 11.016, 13.173, 10.481, 11.207, 11.009, 10.792, 9.741, 11.98, 9.673, 12.369]
print(len(three_ars))
years = [2011] * 27 + [2012] * 27 + [2013] * 28 + [2014] * 29 + [2015] * 30 + [2016] * 30 + [2017] * 30 + [2018] * 30 + [2019] * 30 + [2020] * 30 + [2021] * 30 + [2022] * 30 + [2023] * 30
print(len(years))
plot_data(three_ars, off_ratings, years, start_at_year=2022, groupby_year=False)

