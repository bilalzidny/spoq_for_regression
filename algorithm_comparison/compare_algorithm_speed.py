import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import time
import sys
import os
from tqdm import tqdm
from joblib import Parallel, delayed

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.algorithms import *
from create_dataset import create_dataset

# === 1. PLOTTING FUNCTION UPDATED FOR 6 METHODS ===
def plot_comparison_extended(results, true_weights=None, plot_table=True):
    """
    Plots loss curves and distances for 6 methods: SPOQ, LASSO, SCAD, Reweighted L1, MCP, IRLS.
    Expects 'results' to be a dictionary containing lists of losses, weights, times, etc.
    """
    
    methods = ['SPOQ', 'LASSO', 'SCAD', 'Reweighted', 'MCP', 'IRLS']
    colors = ['green', 'orange', 'purple', 'blue', 'red', 'cyan']
    
    # Extract data for the last run (to plot curves)
    # Assuming results contains lists of runs, we take the last index [-1]
    losses = [results[f"loss_{m.lower().split()[0]}"][-1] for m in methods]
    
    show_distances = true_weights is not None
    rel_distances = []
    
    if show_distances:
        for m in methods:
            key = f"weights_{m.lower().split()[0]}"
            w_hist = results[key][-1] # History of weights for the last run
            # Calculate distance for each step in history
            dists = [relative_euclidian_distance(w, true_weights) for w in w_hist]
            rel_distances.append(dists)
    else:
        rel_distances = [None] * len(methods)

    # === Layout: 2 Rows of 3 Plots + 1 Row for Table
    fig = plt.figure(figsize=(20, 14 if plot_table else 10))
    
    if plot_table:
        gs = gridspec.GridSpec(3, 3, height_ratios=[3, 3, 1.5], hspace=0.4, wspace=0.3)
    else:
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Plot Curves
    for i, method in enumerate(methods):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Loss Curve
        ax.plot(losses[i], label=f"{method} - Loss", color=colors[i], linewidth=2)
        ax.set_title(method, fontsize=14, fontweight='bold')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle=':', alpha=0.6)

        # Distance Curve (Twin Axis)
        if rel_distances[i] is not None:
            ax2 = ax.twinx()
            ax2.plot(rel_distances[i], '--', label=f"Dist. to Truth", color='black', alpha=0.6, linewidth=1.5)
            ax2.set_ylabel("Rel. Distance")
            
            # Combined Legend
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, fontsize=10, loc='upper right')
        else:
            ax.legend(fontsize=10)

    # === TABLE ===
    if plot_table:
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')

        def fmt(mean_arr, std_arr=None, precision=3, unit=""):
            mean = np.mean(mean_arr)
            if std_arr is not None:
                std = np.std(mean_arr) # Calculate std from the array of means passed
                # Or if std_arr is actually the list of values, same thing.
                return f"{mean:.{precision}f} ± {std:.{precision}f}{unit}"
            return f"{mean:.{precision}f}{unit}"

        # Prepare Table Data
        # We need aggregated stats (mean/std) over all runs
        rows_labels = ['MCO'] + methods
        cols_labels = ['Avg Iterations', 'Avg Runtime (s)', 'Final Rel. Dist.', 'Final Loss']
        
        cell_text = []
        
        # MCO Row
        time_mco = results["time_mco"]
        cell_text.append(["-", fmt(time_mco, time_mco, 5, "s"), "-", "-"])
        
        # Methods Rows
        for i, m in enumerate(methods):
            key = m.lower().split()[0] # spoq, lasso, scad...
            
            iters = results[f"iters_{key}"]
            times = results[f"time_{key}"]
            final_losses = [l[-1] for l in results[f"loss_{key}"]]
            
            if show_distances:
                # Get the last distance of each run
                final_dists_all_runs = []
                for run_weights_hist in results[f"weights_{key}"]:
                    final_w = run_weights_hist[-1]
                    final_dists_all_runs.append(relative_euclidian_distance(final_w, true_weights))
                dist_str = fmt(final_dists_all_runs, final_dists_all_runs, 2, "%") # scientific notation usually better but % requested often
                dist_str = f"{np.mean(final_dists_all_runs):.2e}"
            else:
                dist_str = "-"

            cell_text.append([
                fmt(iters, iters, 0),
                fmt(times, times, 4, "s"),
                dist_str,
                f"{np.mean(final_losses):.2e}"
            ])

        table = ax_table.table(cellText=cell_text, rowLabels=rows_labels, colLabels=cols_labels,
                               loc='center', cellLoc='center', colLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)

    plt.tight_layout()
    plt.show()


# === 2. WORKER FUNCTION FOR PARALLEL EXECUTION ===
def _run_single_comparison(run_idx, params, synthetic, file, target_name, lambda_range):
    """
    Worker function that runs ONE complete experiment (Data generation -> Tuning -> Training 7 models).
    """
    
    # 1. Data Generation / Loading
    if synthetic:
        # Generate in memory (save=False)
        run_params = params.copy()
        run_params["random_state"] = 42 + run_idx
        df, w_ref = create_dataset(save=False, **run_params)
        y = df["target"].to_numpy()
        X = df.drop(columns=["target"]).to_numpy()
    else:
        # Load real data
        X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
        w_ref = None

    # 2. MCO Time (Baseline)
    start = time.time()
    w_mco, _, _ = MCO(X, y)
    time_mco = time.time() - start
    
    # Initial point for all nonconvex methods (Standard practice)
    w_0_value = w_mco.copy()

    # 3. Tuning & Running Models
    # Helper to clean up code
    def tune_and_run(model_fn, fixed_params, method_name):
        # Tune
        best_params, _, _, _, _ = tune_model_optuna(
            model_fn=model_fn,
            lambda_bounds=(lambda_range.min(), lambda_range.max()),
            X=X, y=y,
            fixed_params=fixed_params,
            scoring="aic", verbose=False, n_trials=50 # Reduced trials for speed inside parallel
        )
        
        if "w_0" not in best_params:
            best_params["w_0"] = w_0_value

        # Run
        start_t = time.time()
        # Note: All your algos return: w, loss, mse_val, mse_train, abs_sp, rel_sp, grads, k, weights
        # We need: time, k (iters), loss, weights_history
        _, losses, _, _, _, _, _, k, weights_hist = model_fn(
            **best_params,
            X_train=X, y_train=y,
            X_val=X, y_val=y, # Val same as train for convergence check purpose here
            verbose=False
        )
        duration = time.time() - start_t
        return duration, k, losses, weights_hist

    # --- SPOQ ---
    t_spoq, k_spoq, l_spoq, w_spoq = tune_and_run(
        mm_algorithm_spoqreg, 
        {"w_0": w_0_value, "B": 15, "theta": 0.5, "epsilon": 1e-5, "max_iter": 50000}, "SPOQ"
    )

    # --- LASSO ---
    t_lasso, k_lasso, l_lasso, w_lasso = tune_and_run(
        fista_lasso, 
        {"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000}, "LASSO"
    )

    # --- SCAD ---
    t_scad, k_scad, l_scad, w_scad = tune_and_run(
        fista_scad, 
        {"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000}, "SCAD"
    )

    # --- Reweighted L1 ---
    t_rw, k_rw, l_rw, w_rw = tune_and_run(
        reweighted_l1, 
        {"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 1000}, "Reweighted"
    )

    # --- MCP ---
    t_mcp, k_mcp, l_mcp, w_mcp = tune_and_run(
        fista_mcp, 
        {"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000}, "MCP"
    )

    # --- IRLS ---
    t_irls, k_irls, l_irls, w_irls = tune_and_run(
        irls_lp, 
        {"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 500, "p": 0.5}, "IRLS"
    )

    # 4. Pack results
    return {
        "true_weights": w_ref, # Pass back to verify
        "time_mco": time_mco,
        # SPOQ
        "time_spoq": t_spoq, "iters_spoq": k_spoq, "loss_spoq": l_spoq, "weights_spoq": w_spoq,
        # LASSO
        "time_lasso": t_lasso, "iters_lasso": k_lasso, "loss_lasso": l_lasso, "weights_lasso": w_lasso,
        # SCAD
        "time_scad": t_scad, "iters_scad": k_scad, "loss_scad": l_scad, "weights_scad": w_scad,
        # Reweighted
        "time_reweighted": t_rw, "iters_reweighted": k_rw, "loss_reweighted": l_rw, "weights_reweighted": w_rw,
        # MCP
        "time_mcp": t_mcp, "iters_mcp": k_mcp, "loss_mcp": l_mcp, "weights_mcp": w_mcp,
        # IRLS
        "time_irls": t_irls, "iters_irls": k_irls, "loss_irls": l_irls, "weights_irls": w_irls,
    }


# === 3. MAIN CONTROLLER ===
def compare_methods_parallel(file, target_name, lambda_range=np.logspace(-1, 7), n_runs=10, 
                             synthetic=True, plot=True, save_results=False, output_path="results_summary.csv", n_jobs=-1):
    
    # Parameters for synthetic generation
    params = {
        "n_samples": 1000, "n_features": 100, "n_informative": 20,
        "noise": 0.1, "bias": 10, "coef": True,
        "effective_rank": None, "tail_strength": 0.5,
        "random_state": 42
    }

    loop_range = range(n_runs) if synthetic else range(1)
    print(f"Running comparison on {len(loop_range)} dataset(s) using {n_jobs} cores...")

    # === PARALLEL EXECUTION ===
    # Using joblib to distribute the runs
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_comparison)(i, params, synthetic, file, target_name, lambda_range)
        for i in tqdm(loop_range, desc="Simulations")
    )

    # === AGGREGATION ===
    print("Aggregating results...")
    
    # Keys to aggregate
    keys = ["time_mco"]
    for m in ["spoq", "lasso", "scad", "reweighted", "mcp", "irls"]:
        keys.extend([f"time_{m}", f"iters_{m}", f"loss_{m}", f"weights_{m}"])

    aggregated = {k: [] for k in keys}
    
    # Collect true weights from the first run (assuming consistent if not random, 
    # but here random changes, so we take last run for plotting curves)
    true_weights_last = results_list[-1]["true_weights"]

    for res in results_list:
        for k in keys:
            aggregated[k].append(res[k])

    # === SUMMARY STATS ===
    # Create a clean DataFrame for CSV export (using only Time and Iters)
    stats_data = []
    for m in ["spoq", "lasso", "scad", "reweighted", "mcp", "irls"]:
        stats_data.append({
            "Method": m.upper(),
            "Avg Time (s)": np.mean(aggregated[f"time_{m}"]),
            "Std Time": np.std(aggregated[f"time_{m}"]),
            "Avg Iters": np.mean(aggregated[f"iters_{m}"]),
        })
    # Add MCO
    stats_data.insert(0, {
        "Method": "MCO", 
        "Avg Time (s)": np.mean(aggregated["time_mco"]), 
        "Std Time": np.std(aggregated["time_mco"]), 
        "Avg Iters": "-"
    })
    
    df_summary = pd.DataFrame(stats_data)
    print("\n=== Summary ===")
    print(df_summary)

    if save_results:
        df_summary.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    # === PLOTTING ===
    if plot:
        plot_comparison_extended(aggregated, true_weights=true_weights_last, plot_table=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="custom_dataset_spoq.csv")
    parser.add_argument("--target_name", type=str, default="target")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores")
    
    args = parser.parse_args()
    
    # Example call
    compare_methods_parallel(**vars(args))