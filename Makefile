format: isort black
format-check: isort-check black-check pylint
	
black:
	pipenv run black nba_mvp_predictor

black-check:
	pipenv run black nba_mvp_predictor --check

isort:
	pipenv run isort nba_mvp_predictor --profile black

isort-check:
	pipenv run isort nba_mvp_predictor --check-only  --profile black

pylint:
	pipenv run pylint nba_mvp_predictor --disable missing-module-docstring,import-error,fixme --fail-under=7.0

clean:
	rm ./data/*
	touch ./data/.keep