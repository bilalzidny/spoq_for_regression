import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from utils.logger import save_results
from utils.functions import *
from utils.algorithms import *
from utils.metrics import *
from utils.plots import plot_mse_sparsity_table

# === SETUP LOGGING ===
os.makedirs("logs", exist_ok=True)

def run_results(file, target_name=None, test_size=0.2, train_size = 0.8, scoring= "aic", lambda_range = np.logspace(-7, 7),
                random_state=42, log_results = True, return_results = False, verbose = True, plot = True, 
                X_train =None, X_test=None, y_train =None, y_test = None):
    """
    Run SPOQ, LASSO, SCAD and MCO on a dataset using grid search for lambda tuning.

    Parameters:
        file (str): CSV or dataset filename.
        target_name (str): Name of the target column in the dataset.
        test_size (float): Proportion of the dataset to use for testing.
        scoring (str): Metric used for model selection (e.g., 'aic', 'bic').
        lambda_range (np.ndarray): Range of lambda values to test.
        X_train, X_test, y_train, y_test (np.ndarray): Optional train/test split.
        ...
    
    Returns:
        dict: Dictionary of metrics, sparsities, weights and best parameters.
    """

    np.random.seed(random_state)

    # === LOAD AND PREPROCESS DATA ===
    if any(x is None for x in (X_train, X_test, y_train, y_test)):

        X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

    # === OLS ===
    w_mco , absolute_sparsity_mco, relative_sparsity_mco = MCO(X_train, y_train)

    # === GENERATE w_0 ONCE ===
    w_0_value = np.random.randn(X_train.shape[1])  # Same w_0 for both models

    # === CAN TAKE w_0 = w_mco ===
    w_0_value = w_mco

    # === TUNE TRUST REGION ===
    param_grid_trust = {
        "w_0": [w_0_value],
        "B": [15],
        "theta": [0.5],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [50000]
    }

    best_params_trust, best_score_trust, all_results_trust, best_output_trust, weights_trust = tune_model(
        model_fn=mm_algorithm_spoqreg,
        param_grid=param_grid_trust,
        X=X_train,
        y=y_train,
        scoring=scoring,
    )

    # === TUNE FISTA ===
    param_grid_fista = {
        "w_0": [w_0_value],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [5000]
    }

    best_params_fista, best_score_fista, results_fista, output_fista, weights_fista = tune_model(
        model_fn=fista_lasso,
        param_grid=param_grid_fista,
        X=X_train,
        y=y_train,
        scoring=scoring,
    )

    # === TUNE FISTA SCAD ===
    param_grid_scad = {
        "w_0": [w_0_value],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [5000],
        # tu peux ajouter d’autres paramètres SCAD si besoin, ex: "a": [3.7]
    }
    best_params_scad, best_score_scad, *_ = tune_model(
        model_fn=fista_scad,
        param_grid=param_grid_scad,
        X=X_train, y=y_train,
        scoring=scoring,
    )

    # === RUN MODELS ===
    w_spoq, _, _, _, abs_sparsities_spoq, rel_sparsities_spoq, _, _, _ = mm_algorithm_spoqreg(**best_params_trust,
        w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    w_lasso, _, _, _, abs_sparsities_lasso, rel_sparsities_lasso, _, _, _ = fista_lasso(**best_params_fista,
        w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    w_scad, _, _, _, abs_sparsities_scad, rel_sparsities_scad, _, _, _ = fista_scad(**best_params_scad,
        w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    # === Sparsity ===
    abs_spoq, abs_lasso, abs_scad = abs_sparsities_spoq[-1], abs_sparsities_lasso[-1], abs_sparsities_scad[-1]
    rel_spoq, rel_lasso, rel_scad = rel_sparsities_spoq[-1], rel_sparsities_lasso[-1], rel_sparsities_scad[-1]

    # === TEST ERRORS ===
    err_mco = compute_mse(w_mco, X_test, y_test)
    err_spoq = compute_mse(w_spoq, X_test, y_test)
    err_lasso = compute_mse(w_lasso, X_test, y_test)
    err_scad = compute_mse(w_scad, X_test, y_test)

    rel_err_mco = err_mco / np.linalg.norm(y_test)
    rel_err_spoq = err_spoq / np.linalg.norm(y_test)
    rel_err_lasso = err_lasso / np.linalg.norm(y_test)
    rel_err_scad = err_scad / np.linalg.norm(y_test)

    # === TRAIN ERRORS ===
    err_mco_train = compute_mse(w_mco, X_train, y_train)
    err_spoq_train = compute_mse(w_spoq, X_train, y_train)
    err_lasso_train = compute_mse(w_lasso, X_train, y_train)
    err_scad_train = compute_mse(w_scad, X_train, y_train)

    rel_err_mco_train = err_mco_train / np.linalg.norm(y_train)
    rel_err_spoq_train = err_spoq_train / np.linalg.norm(y_train)
    rel_err_lasso_train = err_lasso_train / np.linalg.norm(y_train)
    rel_err_scad_train = err_scad_train / np.linalg.norm(y_train)


    # === DICT RESULTS ===
    results = {
        "meta": {
            "file": file,
            "test_size": test_size,
            "scoring_metric": scoring,
            "random_state": random_state
        },
        "params": {
            "trust_region": best_params_trust,
            "fista": best_params_fista,
            "scad": best_params_scad
        },
        "errors": {
            "test": {
                "mco": err_mco,
                "spoq": err_spoq,
                "lasso": err_lasso,
                "scad": err_scad
            },
            "train": {
                "mco": err_mco_train,
                "spoq": err_spoq_train,
                "lasso": err_lasso_train,
                "scad": err_scad_train
            }
        },
        "relative_errors": {
            "test": {
                "mco": rel_err_mco,
                "spoq": rel_err_spoq,
                "lasso": rel_err_lasso,
                "scad": rel_err_scad
            },
            "train": {
                "mco": rel_err_mco_train,
                "spoq": rel_err_spoq_train,
                "lasso": rel_err_lasso_train,
                "scad": rel_err_scad_train
            }
        },
        "absolute_sparsities": {
            "mco": absolute_sparsity_mco,
            "spoq": abs_spoq,
            "lasso": abs_lasso,
            "scad": abs_scad
        },
        "relative_sparsities": {
            "mco": relative_sparsity_mco,
            "spoq": rel_spoq,
            "lasso": rel_lasso,
            "scad": rel_scad
        },
        "cv_scores": {
            "spoq": best_score_trust,
            "lasso": best_score_fista,
            "scad": best_score_scad
        },
        "weights": {
            "w_0": w_0_value.tolist(),
            "w_mco": w_mco.tolist(),
            "w_spoq": w_spoq.tolist(),
            "w_lasso": w_lasso.tolist(),
            "w_scad": w_scad.tolist()
        }
    }

    if log_results:
        save_results(results, output_dir="logs", file_prefix="run_scad")

    # === PRINT SUMMARY ===
    if verbose:
        print(f"\nTRAIN Relative MSE: MCO: {rel_err_mco_train:.4f} | SPOQ: {rel_err_spoq_train:.4f} | LASSO: {rel_err_lasso_train:.4f} | SCAD: {rel_err_scad_train:.4f}")
        print(f"TEST Relative MSE: MCO: {rel_err_mco:.4f} | SPOQ: {rel_err_spoq:.4f} | LASSO: {rel_err_lasso:.4f} | SCAD: {rel_err_scad:.4f}")
        print(f"Absolute Sparsities: MCO: {absolute_sparsity_mco:.2f}% | SPOQ: {abs_spoq:.2f}% | LASSO: {abs_lasso:.2f}% | SCAD: {abs_scad:.2f}%")

    if return_results:
        return results

    
    if plot : 
        plot_mse_sparsity_table(results)


def run_results_optuna(file, target_name=None, test_size=0.2, train_size=0.8, scoring="aic",
                     lambda_range=np.logspace(-1, 6), random_state=42,
                     log_results=True, return_results=False, verbose=True, plot=True,
                     X_train=None, X_test=None, y_train=None, y_test=None, n_trials = 200):
    
    """
    Run SPOQ, LASSO, SCAD and MCO on a dataset using Optuna for lambda tuning.

    Parameters:
        file (str): CSV or dataset filename.
        target_name (str): Name of the target column in the dataset.
        test_size (float): Proportion of the dataset to use for testing.
        train_size (float): Proportion of the dataset to use for training.
        scoring (str): Metric used for model selection (e.g., 'aic', 'bic').
        lambda_range (np.ndarray): Range of lambda values to explore.
        random_state (int): Seed for reproducibility.
        log_results (bool): Whether to save results to disk.
        return_results (bool): If True, returns the results dictionary.
        verbose (bool): If True, prints and logs summary.
        plot (bool): Whether to display the result table plot.
        X_train, X_test, y_train, y_test (np.ndarray): Optional pre-split data.
        n_trials (int): Number of Optuna trials for hyperparameter tuning.

    Returns:
        dict: Dictionary of metrics, sparsities, weights and best parameters.
    """

    np.random.seed(random_state)

    # === LOAD AND PREPROCESS DATA ===
    if any(x is None for x in (X_train, X_test, y_train, y_test)):
        X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

    # === OLS ===
    w_mco, absolute_sparsity_mco, relative_sparsity_mco = MCO(X_train, y_train)

    # === w_0 initial ===
    w_0_value = w_mco

    # w_0_value = np.random.randn(np.size(w_mco))

    # === TUNE TRUST REGION ===
    best_params_trust, best_score_trust, _, _, _ = tune_model_optuna(
        model_fn=mm_algorithm_spoqreg,
        lambda_bounds=(lambda_range.min(), lambda_range.max()),
        X=X_train,
        y=y_train,
        fixed_params={
            "w_0": w_0_value,
            "B": 15,
            "theta": 0.5,
            "epsilon": 1e-5,
            "max_iter": 50000
        },
        scoring=scoring,
        verbose=False,
        n_trials=n_trials
    )

    # === TUNE FISTA (LASSO) ===
    best_params_fista, best_score_fista, _, _, _ = tune_model_optuna(
        model_fn=fista_lasso,
        lambda_bounds=(lambda_range.min(), lambda_range.max()),
        X=X_train,
        y=y_train,
        fixed_params={
            "w_0": w_0_value,
            "epsilon": 1e-5,
            "max_iter": 5000
        },
        scoring=scoring,
        verbose=False,
        n_trials=n_trials
    )

    # === TUNE FISTA SCAD ===
    best_params_scad, best_score_scad, _, _, _ = tune_model_optuna(
        model_fn=fista_scad,
        lambda_bounds=(lambda_range.min(), lambda_range.max()),
        X=X_train,
        y=y_train,
        fixed_params={
            "w_0": w_0_value,
            "epsilon": 1e-5,
            "max_iter": 5000
        },
        scoring=scoring,
        verbose=False,
        n_trials=n_trials
    )

    # === RUN MODELS ===
    w_spoq, _, _, _, abs_sparsities_spoq, rel_sparsities_spoq, _, _, _ = mm_algorithm_spoqreg(**best_params_trust,
        w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    w_lasso, _, _, _, abs_sparsities_lasso, rel_sparsities_lasso, _, _, _ = fista_lasso(**best_params_fista,
        w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    w_scad, _, _, _, abs_sparsities_scad, rel_sparsities_scad, _, _, _ = fista_scad(**best_params_scad,
        w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)
    

    # === Sparsity ===
    abs_spoq, abs_lasso, abs_scad = abs_sparsities_spoq[-1], abs_sparsities_lasso[-1], abs_sparsities_scad[-1]
    rel_spoq, rel_lasso, rel_scad = rel_sparsities_spoq[-1], rel_sparsities_lasso[-1], rel_sparsities_scad[-1]

    # === METRICS DICTIONARY (by metric -> method) ===
    metrics = {
        "mse": {},
        "mae": {},
        "mape": {},
        "relative_error": {}
    }

    metrics_train = {
        "mse": {},
        "mae": {},
        "mape": {},
        "relative_error": {}
    }

    for name, w in zip(['mco', 'spoq', 'lasso', 'scad'], [w_mco, w_spoq, w_lasso, w_scad]):
        metrics["mse"][name] = compute_mse(w, X_test, y_test)
        metrics["mae"][name] = compute_mae(w, X_test, y_test)
        metrics["mape"][name] = compute_mape(w, X_test, y_test)
        metrics["relative_error"][name] = compute_relative_sse(w, X_test, y_test)

        metrics_train["mse"][name] = compute_mse(w, X_train, y_train)
        metrics_train["mae"][name] = compute_mae(w, X_train, y_train)
        metrics_train["mape"][name] = compute_mape(w, X_train, y_train)
        metrics_train["relative_error"][name] = compute_relative_sse(w, X_train, y_train)

    # === Sparsity (by metric -> method) ===
    sparsity = {
        "absolute": {
            "mco": absolute_sparsity_mco,
            "spoq": abs_spoq,
            "lasso": abs_lasso,
            "scad": abs_scad
        },
        "relative": {
            "mco": relative_sparsity_mco,
            "spoq": rel_spoq,
            "lasso": rel_lasso,
            "scad": rel_scad
        }
    }

    # === DICT RESULTS ===
    results = {
        "meta": {
            "file": file,
            "test_size": test_size,
            "scoring_metric": scoring,
            "random_state": random_state
        },
        "params": {
            "trust_region": best_params_trust,
            "fista": best_params_fista,
            "scad": best_params_scad
        },
        "metrics": {
            "test": metrics,
            "train": metrics_train
        },
        "sparsity": sparsity,
        "cv_scores": {
            "spoq": best_score_trust,
            "lasso": best_score_fista,
            "scad": best_score_scad
        },
        "weights": {
            "w_0": w_0_value.tolist(),
            "w_mco": w_mco.tolist(),
            "w_spoq": w_spoq.tolist(),
            "w_lasso": w_lasso.tolist(),
            "w_scad": w_scad.tolist()
        }
    }

    if log_results:
        save_results(results, output_dir="logs", file_prefix="run_scad")

    # === PRINT SUMMARY ===
    if verbose:
        print(
            f"\nTRAIN Relative SSE: "
            f"MCO: {metrics_train['relative_error']['mco']:.4f} | "
            f"SPOQ: {metrics_train['relative_error']['spoq']:.4f} | "
            f"LASSO: {metrics_train['relative_error']['lasso']:.4f} | "
            f"SCAD: {metrics_train['relative_error']['scad']:.4f}"
        )
        print(
            f"TEST Relative SSE: "
            f"MCO: {metrics['relative_error']['mco']:.4f} | "
            f"SPOQ: {metrics['relative_error']['spoq']:.4f} | "
            f"LASSO: {metrics['relative_error']['lasso']:.4f} | "
            f"SCAD: {metrics['relative_error']['scad']:.4f}"
        )
        print(
            f"Absolute Sparsities: "
            f"MCO: {absolute_sparsity_mco:.2f}% | "
            f"SPOQ: {abs_spoq:.2f}% | "
            f"LASSO: {abs_lasso:.2f}% | "
            f"SCAD: {abs_scad:.2f}%"
        )


    if return_results:
        return results

    
    if plot : 
        plot_mse_sparsity_table(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="the data file", default="bodyfat.csv")
    parser.add_argument("--target_name", type=str, help="the target of your linear model",default="BodyFat")
    parser.add_argument("--test_size", type=float, help="the data file", default=0.2)
    parser.add_argument("--scoring", type=str, help="the scoring metric for cross-validation", default="aic")
    parser.add_argument("--log_results", action="store_true", help="Log results (default: True)")
    parser.add_argument("--no-log_results", dest="log_results", action="store_false")
    parser.add_argument("--return_results", action="store_true", help="Return results (default: False)")
    parser.add_argument("--n_trials", type=int, help="number of optuna trials", default=200)


    args = parser.parse_args()

    run_results_optuna(**vars(args))

