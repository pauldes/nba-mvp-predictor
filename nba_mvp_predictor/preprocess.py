import pandas
import numpy
import sklearn


def standardize(dataframe, fit_on=None, fit_per_values_of=None, min_max_scaler=False):
    if min_max_scaler:
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
    if fit_on is not None and fit_per_values_of is not None:
        raise NotImplementedError
    if fit_on is None:
        fit_on = dataframe.copy()
    scaled = dataframe.copy()
    if fit_per_values_of is not None:
        series = fit_per_values_of.copy()
        for unique in series.unique():
            curr_index = series[series == unique].index
            df_subset = dataframe.loc[curr_index, :]
            scaler.fit(df_subset[df_subset.columns])
            scaled.loc[curr_index, scaled.columns] = scaler.transform(
                df_subset[df_subset.columns]
            )
    else:
        scaler.fit(fit_on[fit_on.columns])
        scaled[scaled.columns] = scaler.transform(dataframe[dataframe.columns])
    return scaled


def get_numerical_columns(dataframe):
    """Get columns of type number

    Args:
        dataframe (pandas.DataFrame): Data frame

    Returns:
        list[str]: Columns
    """
    return dataframe.select_dtypes(include="number").columns


def get_categorical_columns(dataframe):
    """Get columns of type non-number

    Args:
        dataframe (pandas.DataFrame): Data frame

    Returns:
        list[str]: Columns
    """
    return dataframe.select_dtypes(exclude="number").columns


def natural_log_transform(series):
    """Perform natural log transformation

    Args:
        series (pandas.Series): Series to transform

    Returns:
        pandas.Series: Transformed series
    """
    return numpy.log(series + 1)


def exp_transform(series):
    """Perform exponential transformation

    Args:
        series (pandas.Series): Series to transform

    Returns:
        pandas.Series: Transformed series
    """
    return numpy.exp(series)


def select_random_unique_values(series: pandas.Series, share: float):
    """Select a set of random unique values

    Args:
        series (pandas.Series): Series to select values from
        share (float): Share of unique values to select

    Returns:
        list[]: List of unique values
    """
    share_int = int(share * len(series.unique()))
    sample = series.sample(share_int).tolist()
    return sample


def scale_per_value_of(
    data,
    selected_cat_features,
    selected_num_features,
    fit_per_value_of,
    min_max_scaler=True,
):
    if selected_num_features is None or len(selected_num_features) == 0:
        raise NotImplementedError("Need at least 1 numerical feature")
    processed_num_data = standardize(
        data[selected_num_features],
        fit_on=None,
        fit_per_values_of=fit_per_value_of,
        min_max_scaler=min_max_scaler,
    )
    if selected_cat_features is not None and len(selected_cat_features) > 0:
        processed_cat_data = pandas.get_dummies(
            data[selected_cat_features]
        )  # , drop_first=True)
        processed_data = pandas.concat([processed_num_data, processed_cat_data], axis=1)
        raw_data = data[selected_num_features + selected_cat_features]
    else:
        processed_data = processed_num_data
        raw_data = data[selected_num_features]
    return processed_data, raw_data
