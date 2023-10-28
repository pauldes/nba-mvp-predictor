import argparse
import sys

import streamlit.web.cli

from nba_mvp_predictor import download, explain, predict, train


def download_data(args=None):
    """Download data"""
    download.download_data(args.seasons)


def train_model(args=None):
    """Train a model on dowloaded data"""
    train.train_model()


def make_predictions(args=None):
    """Make predictions with the trained model"""
    predict.make_predictions()


def explain_model(args=None):
    """Explain model decisions"""
    explain.explain_model()


def run_webapp(args=None):
    """Run the web application"""
    sys.argv = ["0", "run", "./streamlit_app.py"]
    streamlit.web.cli.main()


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")
    subparser.add_parser("web", help="Run the web application showing predictions")
    download_parser = subparser.add_parser("download", help="Download data")
    download_parser.add_argument(
        "--seasons",
        required=False,
        help="Seasons to download data for",
        nargs="+",
        type=int,
    )
    subparser.add_parser("train", help="Train a model on dowloaded data")
    subparser.add_parser("predict", help="Make predictions with the trained model")
    subparser.add_parser("explain", help="Explain the predictions made by the model")
    return parser


def run(args=None):
    """CLI entry point.

    Args:
        args : List of args as input of the command line.
    """
    parser = get_parser()
    args = parser.parse_args(args)
    if args.command == "web":
        run_webapp(args)
    elif args.command == "download":
        download_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        make_predictions(args)
    elif args.command == "explain":
        explain_model(args)
