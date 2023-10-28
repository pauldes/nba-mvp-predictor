format: black isort
	
black:
	pipenv run black nba_mvp_predictor

isort:
	pipenv run isort nba_mvp_predictor

clean:
	rm ./data/ -v !(".keep")