"""
Misc. utility functions
"""

import logging
import os

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from scipy import stats


def file_logger(name, level=logging.INFO):
    """
    returns a custom logger that logs to a file
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # don't print to console (don't propagate to root logger)
    logger.propagate = False

    # create a file handler
    handler = logging.FileHandler(f"{name}.csv", mode="w")
    handler.setLevel(level)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def read_all_yaml_cfgs(yaml_cfg_dir):
    """
    Read all yaml config files in a directory
    Returns a dictionary of configs keyed by the yaml filename
    """
    yaml_cfgs = {}
    yaml_cfg_files = os.listdir(yaml_cfg_dir)
    for yaml_cfg_file in yaml_cfg_files:
        if not yaml_cfg_file.endswith((".yaml", ".yml")):
            continue
        yaml_cfg_path = os.path.join(yaml_cfg_dir, yaml_cfg_file)
        yaml_cfg = OmegaConf.load(yaml_cfg_path)
        yaml_cfg_name = Path(yaml_cfg_path).stem
        yaml_cfgs[yaml_cfg_name] = yaml_cfg
    return yaml_cfgs


def get_statistics(values, statistics=None):
    """
    Compute statistics for a metric
    """
    if statistics is None:
        statistics = ["mean",
                      "std",
                      "min",
                      "max",
                      "median",
                      "p50",
                      "p90",
                      "p95",
                      "p99",
                      "p999",
                      "geomean"]
    results = {}
    if "mean" in statistics:
        if len(values) == 0:
            results["mean"] = -1
        else:
            results["mean"] = np.mean(values)
    if "std" in statistics:
        if len(values) == 0:
            results["std"] = -1
        else:
            results["std"] = np.std(values)
    if "min" in statistics:
        if len(values) == 0:
            results["min"] = -1
        else:
            results["min"] = np.min(values)
    if "max" in statistics:
        if len(values) == 0:
            results["max"] = -1
        else:
            results["max"] = np.max(values)
    if "median" in statistics:
        if len(values) == 0:
            results["median"] = -1
        else:
            results["median"] = np.median(values)
    if "p50" in statistics:
        if len(values) == 0:
            results["p50"] = -1
        else:
            results["p50"] = np.percentile(values, 50)
    if "p90" in statistics:
        if len(values) == 0:
            results["p90"] = -1
        else:
            results["p90"] = np.percentile(values, 90)
    if "p95" in statistics:
        if len(values) == 0:
            results["p95"] = -1
        else:
            results["p95"] = np.percentile(values, 95)
    if "p99" in statistics:
        if len(values) == 0:
            results["p99"] = -1
        else:
            results["p99"] = np.percentile(values, 99)
    if "p999" in statistics:
        if len(values) == 0:
            results["p999"] = -1
        else:
            results["p999"] = np.percentile(values, 99.9)
    if "geomean" in statistics:
        if len(values) == 0:
            results["geomean"] = -1
        else:
            results["geomean"] = stats.gmean(values)
    return results


def save_dict_as_csv(d, filename):
    dirname = os.path.dirname(filename)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    df = pd.DataFrame(d)
    df.to_csv(filename, index=False)

def save_dict_as_yaml(data, filename):
    data = OmegaConf.to_container(data, resolve=True)
    with open(get_original_cwd() + "/" + filename, 'w+') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def save_df_to_csv(df, filename):
    dirname = os.path.dirname(filename)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    df.to_csv(filename, index=False)

def convert_seconds_to_dd_hh_min_ss(seconds):
    seconds = int(seconds)
    days = seconds // 86400
    remaining_seconds = seconds % 86400
    hours = remaining_seconds // 3600
    remaining_seconds = remaining_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"