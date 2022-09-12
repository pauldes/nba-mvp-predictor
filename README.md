[![tests](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/tests.yaml/badge.svg)](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/tests.yaml)
[![train](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/train.yaml/badge.svg)](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/train.yaml)
[![predict](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/predict.yaml/badge.svg)](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/predict.yaml)

# **The MVP Predict🏀r** : Predicting the NBA Most Valuable Player

This project aims at predicting the player who will win the NBA MVP award, by modelling the voting panel. The NBA MVP is given since the 1955–56 season to the best performing player of the regular season. Until the 1979–80 season, the MVP was selected by a vote of NBA players. Since the 1980–81 season, the award is decided by a panel of sportswriters and broadcasters - more info [here](https://en.wikipedia.org/wiki/NBA_Most_Valuable_Player_Award).

Have a look at the last model predictions at [streamlit.io/pauldes/nba-mvp-predictor/main](https://share.streamlit.io/pauldes/nba-mvp-predictor/main) !

## Development

Install the Python dependencies :
```pipenv install --python 3.8 --dev```

Use the app :
```pipenv run python . --help```

## Process and Methodology

The simplified pipeline overview for the prediction process is as follows:
- Extract player data from Basketball Reference
- Train regression models using MVP shares as response
  - Filter the data to narrow the player pool as much as possible (goes from bronze -> silver -> gold data)
  - Split the gold data into training and validation sets
  - Train with Multi Layer Perceptron and KFold Validation
- Predict the current MVP using the fitted model
- Display feature importance using Shapley values

#### Data Retrieval

Retrieving the player and team data is the first step. The abstract class is defined and implemented in ```scrappers.py``` and it makes use of the Beatiful Soup API to get player stats, team standings, and a dataframe of all the MVP winners starting from the 1974 NBA season up until the latest season from Basketball Reference. Then, the ```download.py``` file uses the methods defined in ```scrappers.py``` to dump the needed dataframes to csv files.

#### Preprocessing and Training 

This is the primary step of the pipleline and it is implemented mainly in ```train.py```. The filteration of data starts from raw data and ends up as "gold" data, which is data that is ready to be used in modeling:
- First, the raw data is converted to "bronze" data which is just a formatting of the raw downloaded data
- Next, filters of >= 50% games played, >= 24 MPG, and team seed of at least 10 in the conference were applied to further reduce the dataset. This is the "silver" data
- Finally, the "gold" dataset not only reduces the player space, but also reduces the dimensionality (features/columns) of the dataset by removing categorical features as well as features that don't meet the 98% correlation threshold.

The "gold" dataset generation method is also where both the normalization of the data and the actual training occur. The normalization methods are either min-max scaling or standardization:
- Min-max scaling: Transforms all the numeric values per column to be within 0 and 1. That is, all values in a certain column are divided by the maximum value of that column.
- Standardization: Transforms all the numeric values per column to lie on a normal distribution. That is, all values are centered around a mean of 0 and a standard deviation of 1

The dataset is split into training and validation sets based on seasons. Then, the architecture for the MLP regressor is defined, with the following hyperparameters:
```python
hidden_layer_sizes=9,
learning_rate="adaptive",
learning_rate_init=0.065,
random_state=0
```

Each step of the model is printed with its training and validation MAE, MSE, and MaxAE - all different accuracy metrics - to display the progress of the MLP with each step until convergence. Furthermore, this is done with a KFold Validation of 3 splits and 2 repeats.

A final regressor is then selected to fit the data and this model is saved using ```joblib```.

#### Predictions and evaluation

```predict.py``` then loads the saved model from the training stage as well as the "silver" data, the data that the model will be used on to make predictions. Using the "silver" set rather than the "gold" data avoids extreme overfitting (since the model has already seen it). 

First, the dimensionality of the data is reduced the and the values are scaled the same way to match both the required dimensions of the predict method as well as to keep the scale of the values consistent with what the model was trained on. 

The predictions are then made using Sci-kit learn's predict method and appended to the original dataframe. From there, the dataframe is reduced based on a prediction value greater than 0.0 to identify the players that the model predicted as MVP for their year. 

As for evaluation, Shapley values are plotted in ```explain.py``` as a way to display the features that contributed the most to determining an MVP winner (sorted in descending order).

## Main challenges


#### Imbalanced data 

There is only 1 MVP per year, among hundreds of players.

Solutions :
- Use MVP share instead of MVP award as the target variable (regression model). A dozen of players receive votes each season.
- Use generally accepted tresholds to filter non-MVP players and reduce the imbalance : 
  - More than 50% of the season games played
  - More than 24 minutes played per game
  - Team ranked in the conference top-10 (play-in qualifier)

#### Label consistency

A player winning MVP one year may not have won MVP the year before, event with the same stats. It all depends on the other players competition.

Solutions :
- Normalize stats per season
  - Min-max scaling
  - Standardization

## Future work and model improvement ideas

- Rank stats (another solution for label consistency issue)
- Use previous years voting results (to model voters lassitude phenomena)
- Limit the players pool in each team to 2 or 3 players based on a treshold to define (or on another model)
- Add top performances or statement games as a feature
- The current model output may be a negative number. This is impossible in real life, since the prediction is an MVP share. Could we leverage on this information to force the model to output non-negative numbers ?
- Feature impacts (SHAP values) may not be reliable for categorical variables (as demonstrated [here](https://arxiv.org/pdf/2103.13342.pdf) and [here](https://arxiv.org/pdf/1909.08128.pdf)).
- Work on my pull-up jumper

## Tools and main libraries

This project relies on awesome libraries, check them out :
[mlflow](https://github.com/mlflow/mlflow)
[streamlit](https://github.com/streamlit/streamlit)
[scikit-learn](https://github.com/scikit-learn/scikit-learn)
[black](https://github.com/psf/black)
[pylint](https://github.com/PyCQA/pylint)
[pipenv](https://github.com/pypa/pipenv)
[beautifulsoup](https://github.com/wention/BeautifulSoup4)
and [more](./Pipfile)
