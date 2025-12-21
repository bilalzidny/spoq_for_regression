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

def run_results(file, target_name=None, test_size=0.2, train_size=0.8, scoring="aic", lambda_range=np.logspace(-1, 6),
                random_state=42, log_results=True, return_results=False, verbose=True, plot=True, 
                X_train=None, X_test=None, y_train=None, y_test=None):
    """
    Run SPOQ, LASSO, SCAD, MCO + New Baselines (Reweighted, MCP, IRLS) using grid search.
    """

    np.random.seed(random_state)

    # === LOAD AND PREPROCESS DATA ===
    if any(x is None for x in (X_train, X_test, y_train, y_test)):
        X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

    # === OLS ===
    w_mco, absolute_sparsity_mco, relative_sparsity_mco = MCO(X_train, y_train)

    # === GENERATE w_0 ONCE ===
    # Using MCO as initialization is standard for nonconvex problems to avoid bad local minima
    w_0_value = w_mco.copy()

    # ==========================
    # === HYPERPARAM TUNING ===
    # ==========================

    # --- SPOQ ---
    param_grid_trust = {
        "w_0": [w_0_value],
        "B": [15],
        "theta": [0.5],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [5000]
    }
    best_params_trust, best_score_trust, *_ = tune_model(
        model_fn=mm_algorithm_spoqreg, param_grid=param_grid_trust, X=X_train, y=y_train, scoring=scoring
    )

    # --- LASSO (FISTA) ---
    param_grid_fista = {
        "w_0": [w_0_value],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [5000]
    }
    best_params_fista, best_score_fista, *_ = tune_model(
        model_fn=fista_lasso, param_grid=param_grid_fista, X=X_train, y=y_train, scoring=scoring
    )

    # --- SCAD ---
    param_grid_scad = {
        "w_0": [w_0_value],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [5000]
    }
    best_params_scad, best_score_scad, *_ = tune_model(
        model_fn=fista_scad, param_grid=param_grid_scad, X=X_train, y=y_train, scoring=scoring
    )

    # --- REWEIGHTED L1 ---
    param_grid_rw = {
        "w_0": [w_0_value],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [5000] # Total iters distributed inside
    }
    best_params_rw, best_score_rw, *_ = tune_model(
        model_fn=reweighted_l1, param_grid=param_grid_rw, X=X_train, y=y_train, scoring=scoring
    )

    # --- MCP ---
    param_grid_mcp = {
        "w_0": [w_0_value],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [5000]
    }
    best_params_mcp, best_score_mcp, *_ = tune_model(
        model_fn=fista_mcp, param_grid=param_grid_mcp, X=X_train, y=y_train, scoring=scoring
    )

    # --- IRLS (Lp p=0.5) ---
    param_grid_irls = {
        "w_0": [w_0_value],
        "epsilon": [1e-5],
        "lambda_pen": lambda_range,
        "max_iter": [5000], # IRLS converges faster usually (Newton-like)
        "p": [0.5]
    }
    best_params_irls, best_score_irls, *_ = tune_model(
        model_fn=irls_lp, param_grid=param_grid_irls, X=X_train, y=y_train, scoring=scoring
    )

    # ====================
    # === FINAL RUNS ===
    # ====================

    # SPOQ
    w_spoq, _, _, _, abs_sparsities_spoq, rel_sparsities_spoq, _, _, _ = mm_algorithm_spoqreg(
        **best_params_trust, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)
    
    # LASSO
    w_lasso, _, _, _, abs_sparsities_lasso, rel_sparsities_lasso, _, _, _ = fista_lasso(
        **best_params_fista, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    # SCAD
    w_scad, _, _, _, abs_sparsities_scad, rel_sparsities_scad, _, _, _ = fista_scad(
        **best_params_scad, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    # REWEIGHTED L1
    w_reweighted, _, _, _, abs_sparsities_rw, rel_sparsities_rw, _, _, _ = reweighted_l1(
        **best_params_rw, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    # MCP
    w_mcp, _, _, _, abs_sparsities_mcp, rel_sparsities_mcp, _, _, _ = fista_mcp(
        **best_params_mcp, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    # IRLS
    w_irls, _, _, _, abs_sparsities_irls, _, _, _, _ = irls_lp(
        **best_params_irls, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)


    # === COLLECT RESULTS ===
    models_list = ['mco', 'spoq', 'lasso', 'scad', 'reweighted', 'mcp', 'irls']
    weights_list = [w_mco, w_spoq, w_lasso, w_scad, w_reweighted, w_mcp, w_irls]
    
    # For sparsities, IRLS might not have "relative" computed in same list format, handle carefully
    # Assuming the functions return lists, take the last element
    abs_sparsities = {
        "mco": absolute_sparsity_mco,
        "spoq": abs_sparsities_spoq[-1], "lasso": abs_sparsities_lasso[-1], "scad": abs_sparsities_scad[-1],
        "reweighted": abs_sparsities_rw[-1], "mcp": abs_sparsities_mcp[-1], "irls": abs_sparsities_irls[-1]
    }

    # Relative sparsities (handle if IRLS returns empty list for relative)
    rel_sparsities_irls_val = 0 # Placeholder if not computed
    if isinstance(rel_sparsities_rw, list) and len(rel_sparsities_rw) > 0: rel_sparsities_rw_val = rel_sparsities_rw[-1]
    else: rel_sparsities_rw_val = 0
    
    rel_sparsities = {
        "mco": relative_sparsity_mco,
        "spoq": rel_sparsities_spoq[-1], "lasso": rel_sparsities_lasso[-1], "scad": rel_sparsities_scad[-1],
        "reweighted": rel_sparsities_rw_val, "mcp": rel_sparsities_mcp[-1], "irls": rel_sparsities_irls_val
    }

    # Errors calculation loop
    errors_test, errors_train = {}, {}
    rel_errors_test, rel_errors_train = {}, {}

    for name, w in zip(models_list, weights_list):
        errors_test[name] = compute_mse(w, X_test, y_test)
        errors_train[name] = compute_mse(w, X_train, y_train)
        rel_errors_test[name] = errors_test[name] / np.linalg.norm(y_test)
        rel_errors_train[name] = errors_train[name] / np.linalg.norm(y_train)

    # === DICT RESULTS ===
    results = {
        "meta": {"file": file, "test_size": test_size, "scoring_metric": scoring, "random_state": random_state},
        "params": {
            "trust_region": best_params_trust, "fista": best_params_fista, "scad": best_params_scad,
            "reweighted": best_params_rw, "mcp": best_params_mcp, "irls": best_params_irls
        },
        "errors": {"test": errors_test, "train": errors_train},
        "relative_errors": {"test": rel_errors_test, "train": rel_errors_train},
        "absolute_sparsities": abs_sparsities,
        "relative_sparsities": rel_sparsities,
        "cv_scores": {
            "spoq": best_score_trust, "lasso": best_score_fista, "scad": best_score_scad,
            "reweighted": best_score_rw, "mcp": best_score_mcp, "irls": best_score_irls
        },
        "weights": {
            "w_0": w_0_value.tolist(),
            "w_mco": w_mco.tolist(), "w_spoq": w_spoq.tolist(), "w_lasso": w_lasso.tolist(), "w_scad": w_scad.tolist(),
            "w_reweighted": w_reweighted.tolist(), "w_mcp": w_mcp.tolist(), "w_irls": w_irls.tolist()
        }
    }

    if log_results:
        save_results(results, output_dir="logs", file_prefix="run_grid")

    if verbose:
        print("\n=== FINAL RESULTS (Relative MSE Test) ===")
        for name in models_list:
            print(f"{name.upper():<12}: {rel_errors_test[name]:.4f} (Sparsity: {abs_sparsities[name]:.1f}%)")

    if return_results:
        return results
    
    if plot: 
        plot_mse_sparsity_table(results)


def run_results_optuna(file, target_name=None, test_size=0.2, train_size=0.8, scoring="aic",
                     lambda_range=np.logspace(-1, 6), random_state=42,
                     log_results=True, return_results=False, verbose=True, plot=True,
                     X_train=None, X_test=None, y_train=None, y_test=None, n_trials=100):
    """
    Run SPOQ, LASSO, SCAD, MCO + New Baselines (Reweighted, MCP, IRLS) using Optuna.
    """

    np.random.seed(random_state)

    if any(x is None for x in (X_train, X_test, y_train, y_test)):
        X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

    # === OLS ===
    w_mco, absolute_sparsity_mco, relative_sparsity_mco = MCO(X_train, y_train)

    # === w_0 initial ===
    w_0_value = w_mco.copy()

    # ==============================
    # === HYPERPARAM TUNING (OPTUNA)
    # ==============================
    
    lambda_bounds = (lambda_range.min(), lambda_range.max())

    # --- SPOQ ---
    best_params_trust, best_score_trust, _, _, _ = tune_model_optuna(
        model_fn=mm_algorithm_spoqreg, lambda_bounds=lambda_bounds, X=X_train, y=y_train,
        fixed_params={"w_0": w_0_value, "B": 15, "theta": 0.5, "epsilon": 1e-5, "max_iter": 50000},
        scoring=scoring, verbose=False, n_trials=n_trials
    )

    # --- LASSO ---
    best_params_fista, best_score_fista, _, _, _ = tune_model_optuna(
        model_fn=fista_lasso, lambda_bounds=lambda_bounds, X=X_train, y=y_train,
        fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000},
        scoring=scoring, verbose=False, n_trials=n_trials
    )

    # --- SCAD ---
    best_params_scad, best_score_scad, _, _, _ = tune_model_optuna(
        model_fn=fista_scad, lambda_bounds=lambda_bounds, X=X_train, y=y_train,
        fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000},
        scoring=scoring, verbose=False, n_trials=n_trials
    )

    # --- REWEIGHTED L1 ---
    best_params_rw, best_score_rw, _, _, _ = tune_model_optuna(
        model_fn=reweighted_l1, lambda_bounds=lambda_bounds, X=X_train, y=y_train,
        fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 1000},
        scoring=scoring, verbose=False, n_trials=n_trials
    )

    # --- MCP ---
    best_params_mcp, best_score_mcp, _, _, _ = tune_model_optuna(
        model_fn=fista_mcp, lambda_bounds=lambda_bounds, X=X_train, y=y_train,
        fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000},
        scoring=scoring, verbose=False, n_trials=n_trials
    )

    # --- IRLS ---
    # For IRLS, we also set p=0.5. If you want to tune p, you need to modify tune_model_optuna to handle it.
    best_params_irls, best_score_irls, _, _, _ = tune_model_optuna(
        model_fn=irls_lp, lambda_bounds=lambda_bounds, X=X_train, y=y_train,
        fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 500, "p": 0.5},
        scoring=scoring, verbose=False, n_trials=n_trials
    )

    # ====================
    # === FINAL RUNS ===
    # ====================

    w_spoq, _, _, _, abs_sparsities_spoq, rel_sparsities_spoq, _, _, _ = mm_algorithm_spoqreg(
        **best_params_trust, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    w_lasso, _, _, _, abs_sparsities_lasso, rel_sparsities_lasso, _, _, _ = fista_lasso(
        **best_params_fista, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    w_scad, _, _, _, abs_sparsities_scad, rel_sparsities_scad, _, _, _ = fista_scad(
        **best_params_scad, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)

    w_reweighted, _, _, _, abs_sparsities_rw, rel_sparsities_rw, _, _, _ = reweighted_l1(
        **best_params_rw, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)
    
    w_mcp, _, _, _, abs_sparsities_mcp, rel_sparsities_mcp, _, _, _ = fista_mcp(
        **best_params_mcp, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)
    
    w_irls, _, _, _, abs_sparsities_irls, _, _, _, _ = irls_lp(
        **best_params_irls, w_0=w_0_value, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, verbose=False)


    # === GATHER RESULTS ===
    models_list = ['mco', 'spoq', 'lasso', 'scad', 'reweighted', 'mcp', 'irls']
    weights_list = [w_mco, w_spoq, w_lasso, w_scad, w_reweighted, w_mcp, w_irls]

    # Metrics dictionary init
    metrics = {k: {} for k in ["mse", "mae", "mape", "relative_error"]}
    metrics_train = {k: {} for k in ["mse", "mae", "mape", "relative_error"]}

    for name, w in zip(models_list, weights_list):
        # Test Metrics
        metrics["mse"][name] = compute_mse(w, X_test, y_test)
        metrics["mae"][name] = compute_mae(w, X_test, y_test)
        metrics["mape"][name] = compute_mape(w, X_test, y_test)
        metrics["relative_error"][name] = compute_relative_sse(w, X_test, y_test)
        
        # Train Metrics
        metrics_train["mse"][name] = compute_mse(w, X_train, y_train)
        metrics_train["mae"][name] = compute_mae(w, X_train, y_train)
        metrics_train["mape"][name] = compute_mape(w, X_train, y_train)
        metrics_train["relative_error"][name] = compute_relative_sse(w, X_train, y_train)

    # Sparsity
    # Helper to safely get last element
    def get_last(lst, default=0):
        return lst[-1] if isinstance(lst, list) and len(lst) > 0 else default

    sparsity = {
        "absolute": {
            "mco": absolute_sparsity_mco,
            "spoq": get_last(abs_sparsities_spoq), "lasso": get_last(abs_sparsities_lasso), "scad": get_last(abs_sparsities_scad),
            "reweighted": get_last(abs_sparsities_rw), "mcp": get_last(abs_sparsities_mcp), "irls": get_last(abs_sparsities_irls)
        },
        "relative": {
            "mco": relative_sparsity_mco,
            "spoq": get_last(rel_sparsities_spoq), "lasso": get_last(rel_sparsities_lasso), "scad": get_last(rel_sparsities_scad),
            "reweighted": get_last(rel_sparsities_rw), "mcp": get_last(rel_sparsities_mcp), "irls": 0 # IRLS relative often not computed
        }
    }

    # === DICT RESULTS ===
    results = {
        "meta": {"file": file, "test_size": test_size, "scoring_metric": scoring, "random_state": random_state},
        "params": {
            "trust_region": best_params_trust, "fista": best_params_fista, "scad": best_params_scad,
            "reweighted": best_params_rw, "mcp": best_params_mcp, "irls": best_params_irls
        },
        "metrics": {"test": metrics, "train": metrics_train},
        "sparsity": sparsity,
        "cv_scores": {
            "spoq": best_score_trust, "lasso": best_score_fista, "scad": best_score_scad,
            "reweighted": best_score_rw, "mcp": best_score_mcp, "irls": best_score_irls
        },
        "weights": {
            "w_0": w_0_value.tolist(),
            "w_mco": w_mco.tolist(), "w_spoq": w_spoq.tolist(), "w_lasso": w_lasso.tolist(), "w_scad": w_scad.tolist(),
            "w_reweighted": w_reweighted.tolist(), "w_mcp": w_mcp.tolist(), "w_irls": w_irls.tolist()
        }
    }

    if log_results:
        save_results(results, output_dir="logs", file_prefix="run_optuna")

    if verbose:
        print("\n=== FINAL RESULTS (Relative SSE Test) ===")
        for name in models_list:
             print(f"{name.upper():<12}: {metrics['relative_error'][name]:.4f} | Sparsity: {sparsity['absolute'][name]:.2f}%")

    if return_results:
        return results
    
    if plot: 
        plot_mse_sparsity_table(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="the data file", default="bodyfat.csv")
    parser.add_argument("--target_name", type=str, help="the target of your linear model", default="BodyFat")
    parser.add_argument("--test_size", type=float, help="test split ratio", default=0.2)
    parser.add_argument("--scoring", type=str, help="scoring metric", default="aic")
    parser.add_argument("--log_results", action="store_true", help="Log results (default: True)")
    parser.add_argument("--no-log_results", dest="log_results", action="store_false")
    parser.add_argument("--return_results", action="store_true", help="Return results (default: False)")
    parser.add_argument("--n_trials", type=int, help="number of optuna trials", default=200)

    args = parser.parse_args()
    run_results_optuna(**vars(args))
