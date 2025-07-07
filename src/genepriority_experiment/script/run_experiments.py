#!/usr/bin/env python3
"""
NEGTrainer Command Launcher
===========================

Generates and sequentially executes sets of `genepriority` commands
for different experimental comparisons (side-info, latent dimensions,
label flipping, zero sampling factor, with/without side-info).

Usage:
    Simply run this script; it will iterate over all experiment types
    and invoke each command via subprocess, logging progress and errors.
"""

import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Sequence

# --- Constants and default settings ----------------------------------------

BASE_OUTPUT_DIR = Path("experiment")
COMMON_ARGS = [
    "--num-folds",
    "5",
    "--validation-size",
    "0.1",
    "--seed",
    "42",
]

# Flags needed for commands that require iterations, patience, and max_dims
ADVANCED_ARGS = [
    "--iterations",
    "2000",
    "--patience",
    "200",
    "--max_dims",
    "1000",
]

GENE_SI_PATHS = [
    Path("/home/TheGreatestCoder/code/data/postprocessed/interpro.csv"),
    Path("/home/TheGreatestCoder/code/data/postprocessed/uniprot.csv"),
    Path("/home/TheGreatestCoder/code/data/postprocessed/go.csv"),
    Path("/home/TheGreatestCoder/code/data/postprocessed/gene-literature.csv"),
]

DISEASE_SI_PATH = Path("/home/TheGreatestCoder/code/data/postprocessed/phenotype.csv")
GENE_DISEASE_PATH = Path(
    "/home/TheGreatestCoder/code/data/postprocessed/gene-disease.csv"
)

# --- Helper functions ------------------------------------------------------


def run_command(cmd: Sequence[str]) -> None:
    """
    Execute a shell command and wait for it to finish.

    Args:
        cmd: Sequence of command and arguments to execute.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero status.
    """
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Completed: {' '.join(cmd)}")


def all_sublists(items: List[Path]) -> List[List[Path]]:
    """
    Generate all non-empty sublists (combinations) of the input list.

    Args:
        items: List of Path objects to generate combinations from.

    Returns:
        A list of lists, each containing a non-empty combination of the input items.
    """
    return [
        list(combo)
        for r in range(1, len(items) + 1)
        for combo in combinations(items, r)
    ]


def format_paths_for_arg(paths: List[Path]) -> List[str]:
    """
    Convert a list of Path objects into their string representations.

    Args:
        paths: List of Path objects to convert.

    Returns:
        List of path strings suitable for use as command-line arguments.
    """
    return [str(p) for p in paths]


def build_base_cmd(method: str, output_subdir: str, args: Sequence[str]) -> List[str]:
    """
    Construct a full `genepriority` command list.

    Args:
        method: The tool invocation and subcommand (e.g., 'nega cv' or 'genehound').
        output_subdir: Subdirectory under `BASE_OUTPUT_DIR` for storing outputs.
        args: Additional command-line arguments to include.

    Returns:
        A list of strings representing the complete command to execute.
    """
    return [
        "genepriority",
        *method.split(),
        "--output-path",
        str(BASE_OUTPUT_DIR / output_subdir),
        *args,
    ]


# --- Comparison generators -------------------------------------------------


def comparison_si() -> List[List[str]]:
    """
    Generate commands for varying subsets of gene side information.

    Returns:
        A list of command argument lists, each representing a unique combination of side-information paths.
    """
    commands = []
    for sublist in all_sublists(GENE_SI_PATHS):
        name = "-".join(p.stem for p in sublist)
        args = [
            *COMMON_ARGS,
            *ADVANCED_ARGS,
            "--zero-sampling-factor",
            "5",
            "--side-info",
            "--gene-side-info-paths",
            *format_paths_for_arg(sublist),
            "--disease-side-info-paths",
            str(DISEASE_SI_PATH),
            "--gene-disease-path",
            str(GENE_DISEASE_PATH),
            "--latent-dimension",
            "40",
            "--results-filename",
            f"{name}-results.pickle",
            "--tensorboard-dir",
            "/home/TheGreatestCoder/code/logs/si"
        ]
        commands.append(build_base_cmd("nega cv", "si", args))
    return commands


def comparison_latent() -> List[List[str]]:
    """
    Generate commands for varying the latent-dimension parameter from 20 to 100.

    Returns:
        A list of command argument lists with different `--latent-dimension` values.
    """
    commands = []
    for latent in range(20, 101, 10):
        args = [
            *COMMON_ARGS,
            *ADVANCED_ARGS,
            "--zero-sampling-factor",
            "5",
            "--side-info",
            "--gene-side-info-paths",
            *format_paths_for_arg(GENE_SI_PATHS),
            "--disease-side-info-paths",
            str(DISEASE_SI_PATH),
            "--gene-disease-path",
            str(GENE_DISEASE_PATH),
            "--latent-dimension",
            str(latent),
            "--results-filename",
            f"latent{latent}-results.pickle",
            "--tensorboard-dir",
            "/home/TheGreatestCoder/code/logs/latent"
        ]
        commands.append(build_base_cmd("nega cv", "latent", args))
    return commands


def comparison_flip_label() -> List[List[str]]:
    """
    Generate commands for flip_fraction and zero-sampling factor combinations.

    Returns:
        A list of command argument lists varying `--flip_fraction` and `--zero-sampling-factor`.
    """
    commands = []
    for factor in (0, 5):
        for frac in [round(i * 0.1 + 0.05, 2) for i in range(6)]:
            args = [
                *COMMON_ARGS,
                *ADVANCED_ARGS,
                "--zero-sampling-factor",
                str(factor),
                "--flip_fraction",
                f"{frac:.2f}",
                "--flip_frequency",
                "50",
                "--side-info",
                "--gene-side-info-paths",
                *format_paths_for_arg(GENE_SI_PATHS),
                "--disease-side-info-paths",
                str(DISEASE_SI_PATH),
                "--gene-disease-path",
                str(GENE_DISEASE_PATH),
                "--latent-dimension",
                "40",
                "--results-filename",
                f"frac{frac:.2f}-results.pickle",
                "--tensorboard-dir",
                "/home/TheGreatestCoder/code/logs/flip-label"
            ]
            commands.append(
                build_base_cmd("nega cv", f"flip_label/factor{factor}", args)
            )
    return commands


def comparison_zero_sampling_factor() -> List[List[str]]:
    """
    Generate commands for zero-sampling-factor values from 0 to 10.

    Returns:
        A list of command argument lists with varying `--zero-sampling-factor`.
    """
    commands = []
    for factor in range(11):
        args = [
            *COMMON_ARGS,
            *ADVANCED_ARGS,
            "--zero-sampling-factor",
            str(factor),
            "--side-info",
            "--gene-side-info-paths",
            *format_paths_for_arg(GENE_SI_PATHS),
            "--disease-side-info-paths",
            str(DISEASE_SI_PATH),
            "--gene-disease-path",
            str(GENE_DISEASE_PATH),
            "--latent-dimension",
            "40",
            "--results-filename",
            f"factor{factor}-results.pickle",
            "--tensorboard-dir",
            "/home/TheGreatestCoder/code/logs/zero_sampling_factor"
        ]
        commands.append(build_base_cmd("nega cv", "zero_sampling_factor", args))
    return commands


def comparison_no_si() -> List[List[str]]:
    """
    Generate commands without any side information for both `nega cv` and `genehound`.

    Returns:
        A list containing two command argument lists: one for `nega cv` and one for `genehound`.
    """
    nega_args = [
        *COMMON_ARGS,
        "--zero-sampling-factor",
        "5",
        "--latent-dimension",
        "40",
        "--results-filename",
        "nega-results.pickle",
        "--tensorboard-dir",
        "/home/TheGreatestCoder/code/logs/no-si"
    ]
    genehound_args = [
        *COMMON_ARGS,
        "--zero-sampling-factor",
        "5",
        "--latent-dimension",
        "40",
        "--results-filename",
        "genehound-results.pickle",
        "--tensorboard-dir",
        "/home/TheGreatestCoder/code/logs/no-si"
    ]
    return [
        build_base_cmd("nega cv", "no-si", nega_args),
        build_base_cmd("genehound", "no-si", genehound_args),
    ]


def comparison_with_si() -> List[List[str]]:
    """
    Generate commands with full side information for both `nega cv` and `genehound`.

    Returns:
        A list containing two command argument lists with side-info enabled.
    """
    side_args = [
        "--side-info",
        "--gene-side-info-paths",
        *format_paths_for_arg(GENE_SI_PATHS),
        "--disease-sideinfo-paths",
        str(DISEASE_SI_PATH),
        "--gene-disease-path",
        str(GENE_DISEASE_PATH),
    ]
    nega_args = [
        *COMMON_ARGS,
        ADVANCED_ARGS[0],
        ADVANCED_ARGS[-1],
        "--zero-sampling-factor",
        "5",
        *side_args,
        "--latent-dimension",
        "40",
        "--results-filename",
        "nega-results.pickle",
        "--tensorboard-dir",
        "/home/TheGreatestCoder/code/logs/with-si"
    ]
    genehound_args = [
        *COMMON_ARGS,
        ADVANCED_ARGS[-1],
        "--zero-sampling-factor",
        "5",
        *side_args,
        "--latent-dimension",
        "40",
        "--results-filename",
        "genehound-results.pickle",
        "--tensorboard-dir",
        "/home/TheGreatestCoder/code/logs/with-si"
    ]
    return [
        build_base_cmd("nega cv", "with-si", nega_args),
        build_base_cmd("genehound", "with-si", genehound_args),
    ]


def comparison_with_si_no_max_dim() -> List[List[str]]:
    """
    Generate commands with full side information and omit the `--max_dims` argument.

    Returns:
        A list containing two command argument lists without the `--max_dims` flag.
    """
    side_args = [
        "--side-info",
        "--gene-side-info-paths",
        *format_paths_for_arg(GENE_SI_PATHS),
        "--disease-side-info-paths",
        str(DISEASE_SI_PATH),
        "--gene-disease-path",
        str(GENE_DISEASE_PATH),
    ]
    nega_args = [*COMMON_ARGS] + [
        ADVANCED_ARGS[0],
        "--zero-sampling-factor",
        "5",
        *side_args,
        "--latent-dimension",
        "40",
        "--results-filename",
        "nega-results.pickle",
        "--tensorboard-dir",
        "/home/TheGreatestCoder/code/logs/no-max_dim"
    ]
    genehound_args = [*COMMON_ARGS] + [
        *side_args,
        "--zero-sampling-factor",
        "5",
        "--latent-dimension",
        "40",
        "--results-filename",
        "genehound-results.pickle",
        "--tensorboard-dir",
        "/home/TheGreatestCoder/code/logs/no-max_dim"
    ]
    return [
        build_base_cmd("nega cv", "no-max-dim", nega_args),
        build_base_cmd("genehound", "no-max-dim", genehound_args),
    ]


# --- Main execution --------------------------------------------------------


def main() -> None:
    """
    Iterate through all comparison generators and execute each command in turn.

    Logs any errors encountered during execution to stderr without stopping the full batch.
    """
    generators = [
        comparison_si,
        comparison_latent,
        comparison_flip_label,
        comparison_zero_sampling_factor,
        comparison_no_si,
        comparison_with_si,
        comparison_with_si_no_max_dim,
    ]
    for gen in generators:
        for cmd in gen():
            try:
                run_command(cmd)
            except subprocess.CalledProcessError as e:
                print(
                    f"Error: Command in {gen.__name__}:\n`{' '.join(cmd)}`\n-> failed with exit code {e.returncode}.",
                    file=sys.stderr,
                    end="\n\n",
                )


if __name__ == "__main__":
    main()
