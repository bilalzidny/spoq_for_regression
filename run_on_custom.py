import numpy as np

from utils.logger import *
from utils.functions import *
from utils.algorithms import *
from utils.metrics import *
from utils.plots import plot_sparsity_comparisons

from run_results import run_results, run_results_optuna
from create_dataset import create_dataset


def run_on_custom(plot=True, log_results=True, train_size=0.8, test_size=0.2, 
                  X_train=None, X_test=None, y_train=None, y_test=None,
                  verbose=True, lambda_range=np.logspace(-1, 7), w_ref=None, 
                  tuning="default", n_trials=200, **kwargs):
    """
    Generates a custom dataset, trains LASSO, SPOQ, SCAD, MCO, and new nonconvex baselines 
    (Reweighted L1, MCP, IRLS), compares their sparsity recovery against a known reference.

    Parameters:
        plot (bool): Whether to display sparsity plots.
        log_results (bool): Whether to save results as JSON in logs/.
        tuning (str): "default" or "optuna".
        ...
    """

    # === 1. CREATE THE DATASET ===
    if w_ref is None:
        if verbose: print("Generating dataset...")
        _, w_ref = create_dataset(save=True, noise_design="median", **kwargs)
    
    # === 2. RUN ALGORITHMS (Existing + New Baselines) ===
    # Note: run_results must be updated to train and return w_reweighted, w_mcp, w_irls
    common_args = {
        "file": "custom_dataset.csv",
        "target_name": "target",
        "train_size": train_size,
        "test_size": test_size,
        "scoring": "aic",
        "lambda_range": lambda_range,
        "random_state": 42,
        "log_results": log_results,
        "return_results": True,
        "verbose": verbose,
        "plot": False,
        "X_train": X_train, "X_test": X_test, 
        "y_train": y_train, "y_test": y_test
    }

    if tuning == "default":
        results = run_results(**common_args)
    elif tuning == "optuna":
        results = run_results_optuna(**common_args, n_trials=n_trials)
    else:
        raise ValueError(f"Unknown tuning method: {tuning}")
    
    # === 3. DEFINE MODELS TO COMPARE ===
    # Maps internal key to Display Name
    models_map = {
        "mco": "MCO",
        "lasso": "LASSO",
        "spoq": "SPOQ",
        "scad": "SCAD",
        "reweighted": "Reweighted L1", # New
        "mcp": "MCP",                  # New
        "irls": "IRLS"                 # New
    }

    # Initialize result containers
    confusion_matrices = {}
    similarities = {
        "jaccard": {},
        "hamming": {},
        "euclidean distance to ref": {},
        "relative euclidean distance to ref": {}
    }

    # === 4. COMPUTE METRICS LOOP ===
    # We iterate dynamically to handle cases where some models might be missing from results
    
    for key, label in models_map.items():
        w_key = f"w_{key}" # Expected key in results e.g., 'w_spoq'
        
        # Check if the model results exist
        if w_key in results["weights"]:
            w_est = results["weights"][w_key]
            
            # Confusion Matrix (Sparsity Pattern)
            cm = compare_sparsity(w_ref, w_est, label_ref="Reference", label_test=label)
            confusion_matrices[key] = cm.tolist()

            # Similarity Metrics
            similarities["jaccard"][key] = jaccard_similarity(w_ref, w_est)
            similarities["hamming"][key] = hamming_distance(w_ref, w_est)
            similarities["euclidean distance to ref"][key] = euclidian_distance(w_ref, w_est)
            similarities["relative euclidean distance to ref"][key] = relative_euclidian_distance(w_est, w_ref)
        
        elif verbose and key not in ["reweighted", "mcp", "irls"]: 
            # Warn only for core models, ignore new ones if not yet implemented in run_results
            print(f"Warning: Weights for {label} ({w_key}) not found in results.")

    # === 5. AGGREGATE RESULTS ===
    extended_results = results.copy()
    extended_results.update({
        "dataset_parameters": kwargs,  
        "confusion_matrices": confusion_matrices,
        "similarities": similarities
    })

    # === 6. SAVE AND PLOT ===
    if log_results: 
        save_results(extended_results, output_dir="logs", file_prefix="run")

    if plot: 
        # Ensure your plot function can handle the dynamic keys or updated dictionary structure
        plot_sparsity_comparisons(extended_results, plot_table=True)

    return extended_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a custom regression dataset.")

    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=50, help="Total number of features")
    parser.add_argument("--n_informative", type=int, default=10, help="Number of informative features")
    parser.add_argument("--noise", type=float, default=0.1, help="Relative noise level")
    parser.add_argument("--bias", type=float, default=0, help="Bias term")
    parser.add_argument("--coef", default=True, action="store_true", help="Return coefficients")
    parser.add_argument("--effective_rank", type=int, default=None, help="Approximate rank")
    parser.add_argument("--tail_strength", type=float, default=0.5, help="Tail strength")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--output_path", type=str, default="data/custom_dataset.csv", help="Output path")
    parser.add_argument("--tuning", type=str, default="optuna", help="Tuning method")
    parser.add_argument("--n_trials", type=int, default=100, help="Optuna trials")

    args = parser.parse_args()
    
    # Run
    run_on_custom(**vars(args))