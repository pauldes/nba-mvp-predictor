import argparse
import sys

import streamlit.cli

from nba_mvp_predictor import conf


def command_one(args=None):
    pass


def command_two(args=None):
    pass


def command_web(args=None):
    """Run the web application"""
    sys.argv = ["0", "run", "./streamlit_app.py"]
    streamlit.cli.main()


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")
    web_command = subparser.add_parser("web", help="Run the web application")
    return parser


def run(args=None):
    """CLI entry point.

    Args:
        args : List of args as input of the command line.
    """
    parser = get_parser()
    args = parser.parse_args(args)
    if args.command == "web":
        command_web(args)
