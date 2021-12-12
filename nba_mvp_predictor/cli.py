import argparse
import sys

import streamlit.cli

from nba_mvp_predictor import conf, logger


def download_data(args=None):
    """Download data"""
    pass


def train_model(args=None):
    """Train a model on dowloaded data"""
    pass


def make_predictions(args=None):
    """Make predictions with the trained model"""
    pass


def run_webapp(args=None):
    """Run the web application"""
    sys.argv = ["0", "run", "./streamlit_app.py"]
    streamlit.cli.main()


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")
    subparser.add_parser("web", help="Run the web application showing predictions")
    subparser.add_parser("download", help="Download data")
    subparser.add_parser("train", help="Train a model on dowloaded data")
    subparser.add_parser("predict", help="Make predictions with the trained model")
    return parser


def run(args=None):
    """CLI entry point.

    Args:
        args : List of args as input of the command line.
    """
    logger.debug(f"CLI run called with arguments : {args}")
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
