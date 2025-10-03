import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import numpy as np
import time
import sys
import os

# Add the project root (spoq_for_reg) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from utils.algorithms import *
from create_dataset import create_dataset


def plot_comparison(
    loss_spoq, loss_lasso, loss_scad,
    time_spoq, time_lasso, time_scad,
    time_mco=None,
    iters_spoq=None, iters_lasso=None, iters_scad=None,
    weights_spoq=None, weights_lasso=None, weights_scad=None,
    true_weights=None,
    plot_table=True
):
    show_distances = true_weights is not None

    if show_distances:
        rel_distances_spoq = [relative_euclidian_distance(w, true_weights) for w in weights_spoq]
        rel_distances_lasso = [relative_euclidian_distance(w, true_weights) for w in weights_lasso]
        rel_distances_scad = [relative_euclidian_distance(w, true_weights) for w in weights_scad]

    # === Layout
    fig = plt.figure(figsize=(20, 10 if plot_table else 6))
    if plot_table:
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1], hspace=0.4, wspace=0.3)
    else:
        gs = gridspec.GridSpec(1, 3, wspace=0.3)

    methods = ['SPOQ', 'LASSO', 'SCAD']
    losses = [loss_spoq, loss_lasso, loss_scad]
    rel_distances = [rel_distances_spoq if show_distances else None,
                     rel_distances_lasso if show_distances else None,
                     rel_distances_scad if show_distances else None]
    colors = ['green', 'orange', 'red']

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(losses[i], label=f"{methods[i]} - Loss", color=colors[i], linewidth=2)
        ax.set_title(methods[i], fontsize=16)
        ax.set_xlabel("Iterations", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)

        if rel_distances[i] is not None:
            ax2 = ax.twinx()
            ax2.plot(rel_distances[i], '--', label=f"{methods[i]} - Dist.", color=colors[i], linewidth=2)
            ax2.set_ylabel("Distance", fontsize=14)
            ax2.tick_params(axis='y', labelsize=12)

            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, fontsize=12)
        else:
            ax.legend(fontsize=12)

    # === TABLE
    if plot_table:
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis('off')

        def fmt_mean_std(mean, std, precision=3, unit=""):
            return f"{mean:.{precision}f} ± {std:.{precision}f}{unit}"

        final_dists = [
            rel_distances_spoq[-1] if show_distances else "-",
            rel_distances_lasso[-1] if show_distances else "-",
            rel_distances_scad[-1] if show_distances else "-"
        ]

        cell_text = [
            ["-", fmt_mean_std(*time_mco, 6, "s"), "-", "-"],
            [fmt_mean_std(*iters_spoq, 0), fmt_mean_std(*time_spoq, 6, "s"),
             f"{final_dists[0]:.6e}" if show_distances else "-", f"{loss_spoq[-1]:.5e}"],
            [fmt_mean_std(*iters_lasso, 0), fmt_mean_std(*time_lasso, 6, "s"),
             f"{final_dists[1]:.6e}" if show_distances else "-", f"{loss_lasso[-1]:.5e}"],
            [fmt_mean_std(*iters_scad, 0), fmt_mean_std(*time_scad, 6, "s"),
             f"{final_dists[2]:.6e}" if show_distances else "-", f"{loss_scad[-1]:.5e}"]
        ]

        rows = ['MCO', 'SPOQ', 'LASSO', 'SCAD']
        columns = ['# Iterations', 'Runtime', 'Final Rel. Dist. to True Weights', 'Final Loss']

        table = ax_table.table(cellText=cell_text,
                               rowLabels=rows,
                               colLabels=columns,
                               loc='center',
                               cellLoc='center',
                               colLoc='center')

        for key, cell in table.get_celld().items():
            cell.set_fontsize(15)
        table.scale(1, 2.5)

    plt.tight_layout()
    plt.show()


    
def compare_methods(file, target_name, lambda_range=np.logspace(-1, 7), n_runs=10, 
                    synthetic=True, plot=True, save_results=False, output_path="results_summary.csv"):
    

    # === PARAMETERS FOR SYNTHETIC DATA GENERATION
    params = {
        "n_samples": 100,
        "n_features": 50,
        "n_informative": 10,
        "noise": 0.1,
        "bias": 0,
        "coef": True,
        "effective_rank": None,
        "tail_strength": 0.5,
        "random_state": 42,
        "output_path": "data/custom_dataset_spoq.csv",
    }

    # === DICT TO STORE RUN STATISTICS
    all_stats = {}

    # === SET LOOP RANGE BASED ON SYNTHETIC OR REAL DATASET
    loop_range = range(n_runs) if synthetic else range(1)
    desc = f"Running SPOQ on {n_runs} synthetic datasets" if synthetic else "Running SPOQ on real dataset"

    for i in tqdm(loop_range, desc=desc):
        # === GENERATE SYNTHETIC DATA (only if synthetic=True)
        if synthetic:
            params["random_state"] = 42 + i
            _, w_ref = create_dataset(save=True, **params)

        # === LOAD AND PREPROCESS DATASET
        X, y = load_and_preprocess(file, target_name=target_name, verbose=False)

        # === RUN MCO MULTIPLE TIMES TO AVERAGE EXECUTION TIME
        times_mco = []
        for _ in range(10):
            start = time.time()
            w_mco, _, _ = MCO(X, y)
            times_mco.append(time.time() - start)
        time_mco_mean = np.mean(times_mco)

        w_0_value = w_mco  # Initial weights for all algorithms

        # === HYPERPARAMETER TUNING USING OPTUNA

        best_params_trust, *_ = tune_model_optuna(
            model_fn=mm_algorithm_spoqreg,
            lambda_bounds=(lambda_range.min(), lambda_range.max()),
            X=X, y=y,
            fixed_params={"w_0": w_0_value, "B": 15, "theta": 0.5, "epsilon": 1e-5, "max_iter": 50000},
            scoring="aic", verbose=False, n_trials=200
        )

        best_params_fista, *_ = tune_model_optuna(
            model_fn=fista_lasso,
            lambda_bounds=(lambda_range.min(), lambda_range.max()),
            X=X, y=y,
            fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000},
            scoring="aic", verbose=False, n_trials=200
        )

        best_params_scad, *_ = tune_model_optuna(
            model_fn=fista_scad,
            lambda_bounds=(lambda_range.min(), lambda_range.max()),
            X=X, y=y,
            fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000},
            scoring="aic", verbose=False, n_trials=2
        )

        # === FUNCTION TO RUN EACH METHOD AND MEASURE TIME & CONVERGENCE
        def run_and_time(method_fn, params):
            start = time.time()
            _, losses, *_ , weights = method_fn(
                **params,
                w_0=w_0_value,
                X_train=X, y_train=y,
                X_val=X, y_val=y,
                verbose=False
            )
            return time.time() - start, len(losses), losses, weights

        # === RUN THE ALGORITHMS
        time_spoq, iters_spoq, loss_spoq, weights_spoq = run_and_time(mm_algorithm_spoqreg, best_params_trust)
        time_lasso, iters_lasso, loss_lasso, weights_lasso = run_and_time(fista_lasso, best_params_fista)
        time_scad, iters_scad, loss_scad, weights_scad = run_and_time(fista_scad, best_params_scad)

        # === STORE RESULTS FOR CURRENT RUN
        all_stats[f"run_{i}"] = {
            "time_mco": time_mco_mean,
            "time_spoq": time_spoq,
            "time_lasso": time_lasso,
            "time_scad": time_scad,
            "iters_spoq": iters_spoq,
            "iters_lasso": iters_lasso,
            "iters_scad": iters_scad,
            "loss_spoq": loss_spoq,
            "loss_lasso": loss_lasso,
            "loss_scad": loss_scad,
            "weights_spoq": weights_spoq, 
            "weights_lasso": weights_lasso,
            "weights_scad": weights_scad,
            "true_weights": w_ref,
        }


    # === FINAL AGGREGATED STATISTICS (mean and std)
    df_stats = pd.DataFrame([
        {
            "time_mco": stat["time_mco"],
            "time_spoq": stat["time_spoq"],
            "time_lasso": stat["time_lasso"],
            "time_scad": stat["time_scad"],
            "iters_spoq": stat["iters_spoq"],
            "iters_lasso": stat["iters_lasso"],
            "iters_scad": stat["iters_scad"],
        }
        for stat in all_stats.values()
    ])
    summary = df_stats.agg(["mean", "std"]).T
    summary.columns = ["Mean", "Std Dev"]

    # === PRINT RESULTS
    print("\n Summary of results:\n")
    print(summary.round(6))

    if len(loop_range) == 1:
        if plot:
            stats = list(all_stats.values())[0]  
            time_mco = (stats["time_mco"], 0)  

            plot_comparison(
                loss_spoq=stats["loss_spoq"],
                loss_lasso=stats["loss_lasso"],
                loss_scad=stats["loss_scad"],
                time_spoq=(stats["time_spoq"], 0),
                time_lasso=(stats["time_lasso"], 0),
                time_scad=(stats["time_scad"], 0),
                time_mco=(stats["time_mco"], 0),
                iters_spoq=(stats["iters_spoq"], 0),
                iters_lasso=(stats["iters_lasso"], 0),
                iters_scad=(stats["iters_scad"], 0),
                weights_spoq=stats["weights_spoq"],
                weights_lasso=stats["weights_lasso"],
                weights_scad=stats["weights_scad"],
                true_weights=stats["true_weights"]
            )


    # === SAVE TO CSV IF REQUESTED
    if save_results:
        summary.to_csv(output_path)
        print(f"\n Results saved to {output_path}")

    return summary



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to the data file", default="custom_dataset_spoq.csv")
    parser.add_argument("--target_name", type=str, help="Target column name", default="target")
    parser.add_argument("--n_runs", type=int, help="Number of runs (only used if synthetic)", default=10)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset (with multiple runs)")
    parser.add_argument("--save_results", action="store_true", help="Save summary results to CSV")
    parser.add_argument("--output_path", type=str, help="Path to save the CSV results", default="results_summary.csv")

    args = parser.parse_args()
    compare_methods(**vars(args))
