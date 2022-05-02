import shap

def explain_model():
    pass

"""

@st.cache(ttl=3600)  # 1h cache
def download_model():
    date, url = artifacts.get_last_artifact("model.joblib")
    logger.debug(f"Downloading model from {url}")
    download.download_data_from_url_to_file(
        url, conf.data.model.path, auth=artifacts.get_github_auth()
    )

#@st.cache
def explain(population, sample_to_explain):
    download_model()
    model = load.load_model()
    shap_values = explain.compute_shap_values(model.predict, population, sample_to_explain)
    return shap_values

def compute_shap_values(predict_function, population, sample_to_explain):
    explainer = shap.Explainer(predict_function, population, algorithm="auto")
    shap_values = explainer(sample_to_explain)
    return shap_values




    # predictions, model_input = predict(dataset, model)
        # model_input_top10 = model_input[model_input.index.isin(players_list[:10])]
        # population = model_input[model_input.index.isin(players_list[:100])]
        population = None
        model_input_top10 = None
        shap_values = explain(population, model_input_top10)

"""