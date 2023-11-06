[![test](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/tests.yaml/badge.svg)](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/test.yaml)
[![train](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/train.yaml/badge.svg)](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/train.yaml)
[![predict](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/predict.yaml/badge.svg)](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/predict.yaml)
[![explain](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/explain.yaml/badge.svg)](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/explain.yaml)
[![renew](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/renew.yaml/badge.svg)](https://github.com/pauldes/nba-mvp-predictor/actions/workflows/renew.yaml)

# **The MVP Predict🏀r** : Predicting the NBA Most Valuable Player

This project aims at predicting the player who will win the NBA MVP award, by modelling the voting panel. The NBA MVP is given since the 1955–56 season to the best performing player of the regular season. Until the 1979–80 season, the MVP was selected by a vote of NBA players. Since the 1980–81 season, the award is decided by a panel of sportswriters and broadcasters - more info [here](https://en.wikipedia.org/wiki/NBA_Most_Valuable_Player_Award).

Have a look at the last model predictions at [nba-mvp-predictor.streamlit.app](https://nba-mvp-predictor.streamlit.app/ ) !

## Development

Install the Python dependencies :
```pipenv install --python 3.10 --dev```

Use the app :
```pipenv run python . --help```

## Main challenges


#### Imbalanced data 

There is only 1 MVP per year, among hundreds of players.

Solutions :
- Use MVP share instead of MVP award as the target variable (regression model). A dozen of players receive votes each season.
- Use generally accepted tresholds to filter non-MVP players and reduce the imbalance : 
    - 50% games played
    - 28 minutes per game
    - 2 FG attemptes
    - Team ranked 12th in conference at least

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
