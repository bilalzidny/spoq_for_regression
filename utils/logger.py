import json
import glob
import logging
from datetime import datetime
import os
import numpy as np


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):  # np.int32, np.int64, ...
        return int(obj)
    elif isinstance(obj, (np.floating,)):  # np.float32, np.float64, ...
        return float(obj)
    else:
        return obj

def save_results(results_dict, output_dir="logs", file_prefix="experiment"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{file_prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    serializable_dict = make_json_serializable(results_dict)

    with open(filepath, "w") as f:
        json.dump(serializable_dict, f, indent=4)

    logging.info("Results saved to %s", filepath)
    print(f"Results saved to {filepath}")



def get_logged_results_by_noise():
    """
    Load all logged results and return a dictionary mapping noise levels to result dicts.
    """
    log_dir = "logs"
    results_by_noise = {}

    for file in os.listdir(log_dir):
        if file.startswith("run") and file.endswith(".json"):
            path = os.path.join(log_dir, file)
            with open(path, "r") as f:
                data = json.load(f)
                try:
                    noise = float(data["dataset_parameters"]["noise"])
                    results_by_noise[round(noise, 4)] = data
                except (KeyError, ValueError, TypeError):
                    continue  # skip files that don't match format
    return results_by_noise


def extract_metrics_from_logs(rounded_noise_values, logged_results):
    """
    Extracts relevant metrics from logs for given noise values.
    """
    results_by_noise = {
        "noise": [],
        "jaccard": {"spoq": [], "lasso": [], "mco": []},
        "hamming": {"spoq": [], "lasso": [], "mco": []},
        "euclidean distance to ref": {"spoq": [], "lasso": [], "mco": []},
        "rel_mse_test": {"spoq": [], "lasso": [], "mco": []},
        "rel_mse_train": {"spoq": [], "lasso": [], "mco": []},
        "sparsity": {"spoq": [], "lasso": [], "mco": []},
    }

    for noise in rounded_noise_values:
        log = logged_results[noise]
        results_by_noise["noise"].append(noise)
        for model in ["spoq", "lasso", "mco"]:
            results_by_noise["jaccard"][model].append(log["similarities"]["jaccard"][model])
            results_by_noise["hamming"][model].append(log["similarities"]["hamming"][model])
            results_by_noise["euclidean"][model].append(log["similarities"]["euclidian distance to ref"][model])
            results_by_noise["rel_mse_test"][model].append(log["relative_errors"]["test"][model])
            results_by_noise["rel_mse_train"][model].append(log["relative_errors"]["train"][model])
            results_by_noise["sparsity"][model].append(log["sparsities"][model])

    return results_by_noise
