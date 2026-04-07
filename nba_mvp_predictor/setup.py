from distutils.core import setup

setup(
    name="nba_mvp_predictor",
    version="0.1",
    description="Predicting the NBA Most Valuable Player",
    py_modules=[
        "analytics",
        "analyze",
        "artifacts",
        "cli",
        "download",
        "evaluate",
        "load",
        "predict",
        "preprocess",
        "scrappers",
        "train",
        "utils",
        "web",
        "basketball_reference_scrapper.seasons",
    ],
)
