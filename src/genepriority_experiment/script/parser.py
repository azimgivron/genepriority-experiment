import argparse

from genepriority.scripts.utils import output_dir, yaml_file


def parse_post(parser: argparse.ArgumentParser):
    """
    This command loads serialized Evaluation objects, a YAML configuration file
    containing alpha values, and then produces metric
    plots and tables. It ensures that the number of evaluation paths matches
    the number of provided model names.

    Args:
        subparsers (argparse.ArgumentParser): The argparser.
    """
    parser.add_argument(
        "--output-path",
        metavar="FILE",
        type=output_dir,
        required=True,
        help="Directory to save output results.",
    )
    parser.add_argument(
        "--evaluation-paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to serialized `Evaluation` objects.",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        required=True,
        help="One or more model names corresponding to the evaluation paths (in the same order).",
    )
    parser.add_argument(
        "--post-config-path",
        metavar="FILE",
        type=yaml_file,
        default=yaml_file(
            "/home/TheGreatestCoder/code/genepriority/configurations/post.yaml"
        ),
        help=(
            "Path to the post-processing configuration file containing alpha values."
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-sharey",
        action="store_false",
        help="Whether to share the y axis for BEDROC boxplots (default: %(default)s).",
    )
    parser.add_argument(
        "--full",
        action="store_false",
        help=(
            "If flagged, assessment is made on whole completed matrix instead of the"
            " test set only (default: %(default)s)."
        ),
    )
