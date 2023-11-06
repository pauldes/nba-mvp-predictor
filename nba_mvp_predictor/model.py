import numpy
from sklearn import compose, neural_network


def get_model():
    return _get_regressor()


def _get_regressor():
    return neural_network.MLPRegressor(
        hidden_layer_sizes=9,
        learning_rate="adaptive",
        learning_rate_init=0.065,
        random_state=0,
    )


def _get_relu_regressor():
    return compose.TransformedTargetRegressor(
        regressor=_get_regressor(), func=None, inverse_func=_relu
    )


def _relu(x):
    return numpy.maximum(x, 0)
