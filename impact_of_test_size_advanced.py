import numpy as np
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import logging
from sklearn.model_selection import train_test_split

from utils.logger import save_results
from utils.algorithms import load_and_preprocess
from utils.metrics import jaccard_similarity, hamming_distance, euclidian_distance
from utils.plots import plot_train_size_curves_advanced
from run_results import run_results_optuna
from run_on_custom import run_on_custom
from create_dataset import create_dataset

# === SETUP LOGGING ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/experiment.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def evaluate_train_size_impact(file, target_name, train_sizes, random_state=42, n_runs=10, **base_kwargs):
    """
    Evaluate the effect of training set size on performance for a given dataset. 
    Test on 20% of the dataset and averaged over 10 runs.

    Parameters:
        file (str): Path to the dataset CSV file in data folder.
        target_name (str): Name of the target column in the dataset.
        train_sizes (list of float): Proportions of training data to use.
        random_state (int): Seed for reproducibility.
        n_runs (int): Number of repetitions per training size.
        force (bool): Whether to force rerun even if output figure already exists.
        **base_kwargs: Additional parameters for dataset creation (if custom).
    
    Saves:
        - Results in 'results/' directory
        - Plot in 'plots/[dataset_name]/' directory
    """
    
    output_path = f"plots/{file.split('.')[0]}/mse_sparsity_vs_train_size.png"
    is_custom = "custom_dataset" in file.lower()

    # Check if the figure already exists
    if os.path.exists(output_path):
        answer = input(f"The figure '{output_path}' already exists. Re-run? (y/n): ").strip().lower()
        if answer not in ["y", "yes"]:
            show = input("Show existing figure? (y/n): ").strip().lower()
            if show in ["y", "yes"]:
                img = mpimg.imread(output_path)
                plt.figure(figsize=(18, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            return

    logging.info("STARTED TRAIN SIZE IMPACT EXPERIMENT")

    methods = ["mco", "lasso", "spoq", "scad"]

    # Initialize results dictionary with base and new metrics
    results = {
        "train_size": train_sizes,
        "relative_mse_test": {m: [[] for _ in train_sizes] for m in methods},
        "relative_mse_train": {m: [[] for _ in train_sizes] for m in methods},
        "mae_test": {m: [[] for _ in train_sizes] for m in methods},
        "mae_train": {m: [[] for _ in train_sizes] for m in methods},
        "mape_test": {m: [[] for _ in train_sizes] for m in methods},
        "mape_train": {m: [[] for _ in train_sizes] for m in methods},
        "absolute_sparsity": {m: [[] for _ in train_sizes] for m in methods},
        "relative_sparsity": {m: [[] for _ in train_sizes] for m in methods},
        "lambda_pen_lasso": [[] for _ in train_sizes],
        "lambda_pen_spoq": [[] for _ in train_sizes],
        "lambda_pen_scad": [[] for _ in train_sizes],
    }

    # === CREATE THE DATASET ======
    if is_custom: 
        _, w_ref = create_dataset(save=True, noise_design="median", **base_kwargs)

    # Add custom dataset-specific metrics
    if is_custom:
        for metric in ["jaccard", "hamming", "euclidean", "confusion_matrix"]:
            results[metric] = {m: [[] for _ in train_sizes] for m in methods}

    # Load and split dataset
    X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=random_state
    )

    # Load true weights if using custom dataset
    if is_custom:
        w_ref = np.load("data/weights.npy")

    # Loop over each train size
    for idx, ts in enumerate(tqdm(train_sizes, desc="Train sizes")):
        for run in range(n_runs):
            seed = random_state + run
            rng = np.random.RandomState(seed)

            n = int(ts * len(X_train_full))
            sel = rng.choice(len(X_train_full), size=n, replace=False)
            Xt, yt = X_train_full[sel], y_train_full[sel]

            # Run appropriate evaluation depending on dataset type
            if is_custom:
                res = run_on_custom(
                    plot=False, log_results=True, train_size=ts, test_size=0.2,
                    X_train=Xt, y_train=yt, X_test=X_test, y_test=y_test,
                    lambda_range=np.logspace(-1, 7, 50), w_ref=w_ref, tuning="optuna", n_trials=200, **base_kwargs
                )
            else:
                res = run_results_optuna(
                    file=file, target_name=target_name, train_size=ts, test_size=0.2,
                    random_state=seed, log_results=True, return_results=True, verbose=False,
                    X_train=Xt, y_train=yt, X_test=X_test, y_test=y_test
                )

            # Store metrics per method
            for m in methods:
                results["relative_mse_test"][m][idx].append(res["metrics"]["test"]["relative_error"][m])
                results["relative_mse_train"][m][idx].append(res["metrics"]["train"]["relative_error"][m])
                results["mae_test"][m][idx].append(res["metrics"]["test"]["mae"][m])
                results["mae_train"][m][idx].append(res["metrics"]["train"]["mae"][m])
                results["mape_test"][m][idx].append(res["metrics"]["test"]["mape"][m])
                results["mape_train"][m][idx].append(res["metrics"]["train"]["mape"][m])
                results["absolute_sparsity"][m][idx].append(res["sparsity"]["absolute"][m])
                results["relative_sparsity"][m][idx].append(res["sparsity"]["relative"][m])

                if is_custom:
                    results["jaccard"][m][idx].append(res["similarities"]["jaccard"][m])
                    results["hamming"][m][idx].append(res["similarities"]["hamming"][m])
                    results["euclidean"][m][idx].append(res["similarities"]["relative euclidean distance to ref"][m])
                    results["confusion_matrix"][m][idx].append(res["confusion_matrices"][m])

            # Store optimized lambdas
            results["lambda_pen_lasso"][idx].append(res["params"]["fista"]["lambda_pen"])
            results["lambda_pen_spoq"][idx].append(res["params"]["trust_region"]["lambda_pen"])
            results["lambda_pen_scad"][idx].append(res["params"]["scad"]["lambda_pen"])

    # Save results and generate plots
    save_results(results, output_dir="results", file_prefix=f"train_size_{file.split('.')[0]}")
    plot_train_size_curves_advanced(results, output_path, save_plot=True)
    print("Done – results saved and plot generated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, default="custom_dataset.csv")
    p.add_argument("--target_name", type=str, default="target")
    args = p.parse_args()

    # base_kwargs = {
    #     "n_samples": 1000, "n_features": 100, "n_informative": 20,
    #     "noise": 0.1, "bias":10.0, "coef":True,
    #     "random_state":42, "effective_rank":None,
    #     "tail_strength":0.5, "output_path": "data/custom_dataset.csv"
    # }

    base_kwargs = {
        "n_samples": 100,
        "n_features": 50,
        "n_informative": 10,
        "bias": 0,
        "coef": True,
        "random_state": 42,
        "effective_rank": None,
        "tail_strength": 0.5,
    }
    
    train_sizes = np.linspace(0.15, 0.8, 14).tolist()
    evaluate_train_size_impact(
        file=args.file, target_name=args.target_name,
        train_sizes=train_sizes, n_runs=10, **base_kwargs
    )
