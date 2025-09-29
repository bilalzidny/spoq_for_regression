import json
import os
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from utils.metrics import compute_mse
from utils.algorithms import load_and_preprocess, fista_lasso, mm_algorithm_spoqreg, MCO, fista_scad 

def compute_regularization_paths(file, target_name, lambda_grid_lasso, lambda_grid_spoq, lambda_grid_scad,
                                 test_size=0.2, random_state=42, save_results=False):
    # === LOAD AND SPLIT DATA ===
    X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    w_mco, _, _ = MCO(X_train, y_train)
    w_0 = w_mco

    coefs_lasso, coefs_spoq, coefs_scad = [], [], []
    mses_lasso, mses_spoq, mses_scad = [], [], []

    # ==== LASSO ====
    for lam in lambda_grid_lasso:
        params_fista = {
            "epsilon": 1e-4,
            "lambda_pen": lam,
            "max_iter": 5000
        }
        w_lasso, *_ = fista_lasso(**params_fista, w_0=w_0, X_train=X_train, y_train=y_train,
                            X_val=X_train, y_val=y_train, verbose=False)
        coefs_lasso.append(w_lasso[1:])
        mses_lasso.append(compute_mse(w_lasso, X_test, y_test))

    # ==== SPOQ ====
    for lam in lambda_grid_spoq:
        params_spoq = {
            "B": 15,
            "theta": 0.5,
            "epsilon": 1e-4,
            "lambda_pen": lam,
            "max_iter": 50000
        }
        w_spoq, *_ = mm_algorithm_spoqreg(**params_spoq, w_0=w_0, X_train=X_train,
                                                   y_train=y_train, X_val=X_train, y_val=y_train, verbose=False)
        coefs_spoq.append(w_spoq[1:])
        mses_spoq.append(compute_mse(w_spoq, X_test, y_test))

    # ==== SCAD ====
    for lam in lambda_grid_scad:
        params_scad = {
            "lambda_pen": lam,
            "max_iter": 5000,
            "epsilon": 1e-4
        }
        w_scad, *_ = fista_scad(**params_scad, w_0=w_0, X_train=X_train, y_train=y_train,
                                X_val=X_train, y_val=y_train, verbose=False)
        coefs_scad.append(w_scad[1:])
        mses_scad.append(compute_mse(w_scad, X_test, y_test))

    coefs_lasso = np.array(coefs_lasso)
    coefs_spoq = np.array(coefs_spoq)
    coefs_scad = np.array(coefs_scad)

    # === Plot with adapted function ===
    plot_regularization_paths(lambda_grid_lasso, lambda_grid_spoq, lambda_grid_scad,
                                coefs_lasso, coefs_spoq, coefs_scad,
                                mses_lasso, mses_spoq, mses_scad)

    # === Save results ===
    if save_results: 
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/regularization_path_{timestamp}.json"

        results_dict = {
            "lambda_grid_lasso": lambda_grid_lasso.tolist(),
            "lambda_grid_spoq": lambda_grid_spoq.tolist(),
            "lambda_grid_scad": lambda_grid_scad.tolist(),
            "coefs_lasso": coefs_lasso.tolist(),
            "coefs_spoq": coefs_spoq.tolist(),
            "coefs_scad": coefs_scad.tolist(),
            "mse_lasso": mses_lasso,
            "mse_spoq": mses_spoq,
            "mse_scad": mses_scad,
            "meta": {
                "file": file,
                "target": target_name,
                "test_size": test_size,
                "random_state": random_state
            }
        }

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=4)

        print(f"Regularization paths saved to: {output_path}")


def plot_regularization_paths(lambda_grid_lasso, lambda_grid_spoq, lambda_grid_scad,
                                 coefs_lasso, coefs_spoq, coefs_scad,
                                 mses_lasso, mses_spoq, mses_scad):
    """
    Plot 3 regularization paths with their own lambda grids.
    """

    def _plot_single(ax, lambda_grid, coefs, mses, title, style):
        n_features = coefs.shape[1]
        coefs_norm = coefs / (np.abs(coefs).max(axis=0) + 1e-8)

        for i in range(n_features):
            ax.plot(lambda_grid, coefs[:, i], style, alpha=0.7, linewidth=1)

        ax.set_xscale('log')
        ax.set_xlabel("lambda_pen (log scale)")
        ax.set_ylabel("Standardized coefficient values")
        ax.set_title(title)
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(lambda_grid, mses, linestyle=':', color='gray', alpha=0.4, linewidth=4)
        ax2.set_ylabel("MSE (test)")
        ax2.tick_params(axis='y', labelcolor='gray')

    fig, axs = plt.subplots(1, 3, figsize=(21, 6), sharey=True)

    _plot_single(axs[0], lambda_grid_lasso, coefs_lasso, mses_lasso, "LASSO: Coefs + MSE", style='-')
    _plot_single(axs[1], lambda_grid_spoq, coefs_spoq, mses_spoq, "SPOQ: Coefs + MSE", style='-')
    _plot_single(axs[2], lambda_grid_scad, coefs_scad, mses_scad, "SCAD: Coefs + MSE", style='-')

    fig.tight_layout()
    plt.show()


# === For Bodyfat dataset we show the names of the features for interpretability ===
# === Replace plot_regularization_paths by plot_regularization_paths_bodyfat in compute_regularization_paths ===
def plot_regularization_paths_bodyfat(lambda_grid_lasso, lambda_grid_spoq, lambda_grid_scad,
                                     coefs_lasso, coefs_spoq, coefs_scad,
                                     mses_lasso, mses_spoq, mses_scad):

    variable_names = ["Density", "Age", "Weight", "Height", "Neck", "Chest", "Abdomen",
                      "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"]

    n_features = len(variable_names)
    colors = cm.get_cmap("tab20", n_features)

    def _plot_single(ax, lambda_grid, coefs, mses, title, style):
        coefs_norm = coefs / (np.abs(coefs).max(axis=0) + 1e-8)

        for i in range(n_features):
            color = colors(i)
            ax.plot(lambda_grid, coefs_norm[:, i], style, alpha=0.9, linewidth=1.5, color=color)

        ax.set_xscale('log')
        ax.set_xlabel("lambda_pen (log scale)")
        ax.set_ylabel("Coefficient value")
        ax.set_title(title)
        ax.grid(True)

        # MSE on twin axis
        ax2 = ax.twinx()
        ax2.plot(lambda_grid, mses, linestyle=':', color='gray', alpha=0.4, linewidth=4)
        ax2.set_ylabel("MSE (test)")
        ax2.tick_params(axis='y', labelcolor='gray')

    # === Create subplots ===
    fig, axs = plt.subplots(1, 3, figsize=(23, 7), sharey=True)

    _plot_single(axs[0], lambda_grid_lasso, coefs_lasso, mses_lasso, "LASSO: Coefs + MSE", style='-')
    _plot_single(axs[1], lambda_grid_spoq, coefs_spoq, mses_spoq, "SPOQ: Coefs + MSE", style='-')
    _plot_single(axs[2], lambda_grid_scad, coefs_scad, mses_scad, "SCAD: Coefs + MSE", style='-')

    # === Shared legend below ===
    legend_patches = [mpatches.Patch(color=colors(i), label=var) for i, var in enumerate(variable_names)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=7, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.show()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--target_name", type=str, required=True)
    parser.add_argument("--n_lambdas", type=int, default=50)

    args = parser.parse_args()

    # Adapt the scale of lambda for each method 
    lambda_grid_lasso = np.logspace(-2, 5, args.n_lambdas)
    lambda_grid_spoq = np.logspace(0, 7, args.n_lambdas)    # generally higher value for SPOQ
    lambda_grid_scad = np.logspace(-2, 5, args.n_lambdas)


    compute_regularization_paths(args.file, args.target_name, 
                                lambda_grid_lasso, lambda_grid_spoq, lambda_grid_scad)




