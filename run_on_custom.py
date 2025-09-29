import numpy as np

from utils.logger import *
from utils.functions import *
from utils.algorithms import *
from utils.metrics import *
from utils.plots import plot_sparsity_comparisons

from run_results import run_results, run_results_optuna
from create_dataset import create_dataset


def run_on_custom(plot = True, log_results = True, train_size = 0.8, test_size=0.2, X_train =None, X_test=None, y_train =None, y_test = None,
                   verbose=True, lambda_range =  np.logspace(-1, 7), w_ref=None, tuning="default", n_trials=200, **kwargs):
    """
    Generates a custom dataset, trains LASSO, SPOQ, SCAD, and MCO, compares their sparsity recovery
    against a known reference, and computes similarity metrics.

    Parameters:
        plot (bool): Whether to display sparsity plots.
        log_results (bool): Whether to save results as JSON in logs/.
        train_size (float): Fraction of data to use for training.
        test_size (float): Fraction of data to use for testing.
        X_train, X_test, y_train, y_test (np.ndarray): Optional data splits.
        lambda_range (np.ndarray): Range of lambdas to test.
        w_ref (np.ndarray): Ground-truth coefficients.
        tuning (str): "default" or "optuna" for hyperparameter tuning.
        n_trials (int): Number of Optuna trials.
        kwargs: Parameters for dataset generation.

    Returns:
        dict: Results including weights, confusion matrices, and similarity scores.
    """

    # === CREATE THE DATASET ======
    if w_ref is None:
        _, w_ref = create_dataset(save=True, noise_design="median", **kwargs)
    
    # === RUN RESULTS OF LASSO SPOQ AND MCO ===
    if tuning == "default":
        results = run_results(file = "custom_dataset.csv", target_name="target", train_size=train_size, test_size=test_size, 
                          scoring= "aic", lambda_range =  lambda_range, random_state=42, log_results = log_results, 
                          return_results = True, verbose = verbose, plot=False,
                          X_train =X_train, X_test=X_test, y_train =y_train, y_test = y_test)

    elif tuning == "optuna":
        results = run_results_optuna(file = "custom_dataset.csv", target_name="target", train_size=train_size, test_size=test_size, 
                          scoring= "aic", lambda_range =  lambda_range, random_state=42, log_results = log_results, 
                          return_results = True, verbose = verbose, plot=False,
                          X_train =X_train, X_test=X_test, y_train =y_train, y_test = y_test, n_trials=n_trials)
    else:
        raise ValueError(f"Unknown or unsupported tuning method: {tuning}")
    
    w_spoq = results["weights"]["w_spoq"]
    w_lasso = results["weights"]["w_lasso"]
    w_mco = results["weights"]["w_mco"]
    w_scad = results["weights"]["w_scad"]


    # Compare sparsity patterns
    cm_spoq = compare_sparsity(w_ref, w_spoq, label_ref="Reference", label_test="SPOQ")
    cm_lasso = compare_sparsity(w_ref, w_lasso, label_ref="Reference", label_test="LASSO")
    cm_mco = compare_sparsity(w_ref, w_mco, label_ref="Reference", label_test="MCO")
    cm_scad = compare_sparsity(w_ref, w_scad, label_ref="Reference", label_test="SCAD")


    # Compute similarity/distance metrics
    jac_sim_spoq = jaccard_similarity(w_ref, w_spoq)
    jac_sim_lasso = jaccard_similarity(w_ref, w_lasso)
    jac_sim_mco = jaccard_similarity(w_ref, w_mco)
    jac_sim_scad = jaccard_similarity(w_ref, w_scad)


    ham_dist_spoq = hamming_distance(w_ref, w_spoq)
    ham_dist_lasso = hamming_distance(w_ref, w_lasso)
    ham_dist_mco = hamming_distance(w_ref, w_mco)
    ham_dist_scad = hamming_distance(w_ref, w_scad)


    euc_dist_spoq = euclidian_distance(w_ref, w_spoq) 
    euc_dist_lasso = euclidian_distance(w_ref, w_lasso)
    euc_dist_mco = euclidian_distance(w_ref, w_mco) 
    euc_dist_scad = euclidian_distance(w_ref, w_scad)

    rel_euc_dist_spoq = relative_euclidian_distance(w_spoq, w_ref) 
    rel_euc_dist_lasso = relative_euclidian_distance(w_lasso, w_ref)
    rel_euc_dist_mco = relative_euclidian_distance(w_mco, w_ref) 
    rel_euc_dist_scad = relative_euclidian_distance(w_scad, w_ref)

    extended_results = results.copy()
    extended_results.update({
        "dataset_parameters": kwargs,  

        "confusion_matrices": {
            "spoq": cm_spoq.tolist(),
            "lasso": cm_lasso.tolist(),
            "mco": cm_mco.tolist(),
            "scad": cm_scad.tolist()
        },

        "similarities": {
            "jaccard": {
                "spoq": jac_sim_spoq,
                "lasso": jac_sim_lasso,
                "mco": jac_sim_mco,
                "scad": jac_sim_scad
            },
            "hamming": {
                "spoq": ham_dist_spoq,
                "lasso": ham_dist_lasso,
                "mco": ham_dist_mco,
                "scad": ham_dist_scad
            },
            "euclidean distance to ref": {
                "spoq": euc_dist_spoq,
                "lasso": euc_dist_lasso,
                "mco": euc_dist_mco,
                "scad": euc_dist_scad
            },

            "relative euclidean distance to ref": {
                "spoq": rel_euc_dist_spoq,
                "lasso": rel_euc_dist_lasso,
                "mco": rel_euc_dist_mco,
                "scad": rel_euc_dist_scad
            }
        }

    })

        
    if log_results : 
        save_results(extended_results, output_dir="logs", file_prefix="run")

    if plot : 
        plot_sparsity_comparisons(extended_results,plot_table=True)

    return extended_results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a custom regression dataset.")

    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=50, help="Total number of features")
    parser.add_argument("--n_informative", type=int, default=10, help="Number of informative features")
    parser.add_argument("--noise", type=float, default=0.1, help="Relative noise level: fraction of the target's std deviation (e.g., 0.1 = 10% of signal std)")
    parser.add_argument("--bias", type=float, default=0, help="Bias term in the underlying linear model")
    parser.add_argument("--coef", default=True, action="store_true", help="Return the coefficients of the underlying linear model")
    parser.add_argument("--effective_rank", type=int, default=None, help="Approximate number of singular vectors for input matrix.")
    parser.add_argument("--tail_strength", type=float, default=0.5, help="Relative importance of the fat noisy tail of the singular values profile.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_path", type=str, default="data/custom_dataset.csv", help="Output CSV path")
    parser.add_argument("--tuning", type=str, default="optuna", help="Choose the method for tuning lambda_pen")
    parser.add_argument("--n_trials", type=int, default= 200, help="The number of trials of optuna")


    args = parser.parse_args()
    run_on_custom(**vars(args))
