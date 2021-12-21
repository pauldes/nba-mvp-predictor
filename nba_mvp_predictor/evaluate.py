import pandas
import numpy


def softmax(series: pandas.Series):
    """Compute softmax values for each sets of scores in x.

    Args:
        series (pandas.Series): Series

    Returns:
        pandas.Series: Transformed series (total is 1)
    """
    e_x = numpy.exp(series - numpy.max(series))
    return e_x / e_x.sum()


def share(series: pandas.Series):
    """Compute the share of each value in the series by its share of the series total

    Args:
        series (pandas.Series): Series

    Returns:
        pandas.Series: Series of shares (total is 1)
    """
    return series / series.sum()
