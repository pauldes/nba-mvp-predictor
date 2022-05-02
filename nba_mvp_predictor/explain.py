import joblib

import shap
import pandas
import numpy
from matplotlib import pyplot

from nba_mvp_predictor import conf, logger
from nba_mvp_predictor import load, download

def explain_model():
    """ Explain model predictions.
    """
    model = load.load_model()
    model_input = load.load_model_input()
    predictions = load.load_predictions()
    players_list = predictions.index.to_list()
    # Analyze SHAP values on 10 players
    sample_size = 10
    sample = model_input[model_input.index.isin(players_list[:sample_size])]
    # Find SHAP values on 100 players
    population_size = 100
    population = model_input[model_input.index.isin(players_list[:population_size])]
    explainer = shap.Explainer(model.predict, population, algorithm="auto")
    shap_values = explainer(sample)
    top10_shap_values = {}
    sample["player"] = sample.index
    sample = sample.reset_index(drop=True)
    """
    fig, ax = pyplot.subplots()
    shap.summary_plot(shap_values, population, plot_type="bar", max_display=num_features_displayed)
    pyplot.title(
        f"{num_features_displayed} most impactful features for top-{population_size} players"
    )
    #st.pyplot(fig, transparent=True, width=None, height=100)
    pyplot.show()
    """
    feature_names = shap_values.feature_names
    shap_df = pandas.DataFrame(shap_values.values, columns=feature_names, index=sample.player)
    """
    vals = numpy.abs(shap_df.values).mean(0)
    shap_importance = pandas.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    """
    shap_df.to_csv(
        conf.data.shap_values.path,
        sep=conf.data.shap_values.sep,
        encoding=conf.data.shap_values.encoding,
        compression=conf.data.shap_values.compression,
        index=False,
    )