import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

def plot_mse_sparsity_table(results, precision=6):
    """
    Plot a summary table showing train & test metrics:
    - Relative MSE
    - MAE
    - MAPE
    - Absolute and Relative Sparsity (%)
    
    For the MCO, LASSO, SPOQ, and SCAD methods.

    Parameters:
        results : dict
            Dictionary with structure:
              - results["metrics"]["train"]["<metric_name>"][<method>]
              - results["metrics"]["test"]["<metric_name>"][<method>]
              - results["sparsity"]["absolute"][<method>]
              - results["sparsity"]["relative"][<method>]
        precision : int
            Number of decimal places to display for metrics.
    """

    methods = ["mco", "lasso", "spoq", "scad"]
    col_labels = [m.upper() for m in methods]

    # Extract metrics from nested structure
    train_metrics = results["metrics"]["train"]
    test_metrics = results["metrics"]["test"]

    abs_sparsity = results["sparsity"]["absolute"]
    rel_sparsity = results["sparsity"]["relative"]

    # Define table rows
    row_labels = [
        "MSE",
        # "Train Relative MSE",
        "Test Relative SSE",
        # "Train MAE",
        "Test MAE",
        # "Train MAPE",
        "Test MAPE",
        "Absolute Sparsity (%)",
        "Relative Sparsity (%)"
    ]

    # Build table values
    table_vals = [
        [f"{test_metrics['mse'][m]:.{precision}f}" for m in methods],
        # [f"{train_metrics['relative_error'][m]:.{precision}f}" for m in methods],
        [f"{test_metrics['relative_error'][m]:.{precision}f}" for m in methods],
        # [f"{train_metrics['mae'][m]:.{precision}f}" for m in methods],
        [f"{test_metrics['mae'][m]:.{precision}f}" for m in methods],
        # [f"{train_metrics['mape'][m]:.{precision}f}" for m in methods],
        [f"{test_metrics['mape'][m]:.{precision}f}" for m in methods],
        [f"{abs_sparsity[m]:.2f}" for m in methods],
        [f"{rel_sparsity[m]:.2f}" for m in methods],
    ]

    # Plot table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    table = ax.table(
        cellText=table_vals,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    plt.title("Comparison of MSE, MAE, MAPE and Sparsity", fontsize=16, pad=20)
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_sparsity_comparisons(results, plot_table=False, save_path=None):
    """
    Plot confusion matrices and optionally display a summary table
    including sparsity and error metrics for MCO, LASSO, SPOQ, and SCAD.

    Parameters:
        results : dict
            Dictionary must contain:
                - "confusion_matrices"
                - "similarities"
                - "metrics" -> ["test"] -> ["mse", "relative_error", "mae", "mape"]
                - "sparsity" -> ["absolute"], ["relative"]
        plot_table : bool
            If True, adds a summary table below the confusion matrices.
        save_path : str or None
            If provided, saves the figure to this path.
    """

    plt.close('all')

    methods = ["mco", "lasso", "spoq", "scad"]
    method_titles = {"mco": "MCO", "lasso": "LASSO", "spoq": "SPOQ", "scad": "SCAD"}
    cmaps = {"mco": "Blues", "lasso": "Oranges", "spoq": "Greens", "scad": "Purples"}

    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    plt.subplots_adjust(bottom=0.1)

    # === Plot confusion matrices ===
    for i, method in enumerate(methods):
        cm = np.array(results["confusion_matrices"][method])
        ax = axes[i]
        ax.set_aspect('equal')
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[method],
                    ax=ax, square=True, annot_kws={"size": 20})
        ax.set_title(f"{method_titles[method]} vs Reference", fontsize=18)
        ax.set_xlabel("Predicted Coef. (0: zero, 1: ≠ 0)", fontsize=12)
        ax.set_ylabel("Reference Coef.", fontsize=12)

    if plot_table:
        similarities = results["similarities"]
        test_metrics = results["metrics"]["test"]
        abs_sparsity = results["sparsity"]["absolute"]
        rel_sparsity = results["sparsity"]["relative"]

        col_labels = [method_titles[m] for m in methods]
        row_labels = [
            "Jaccard Similarity (↑)",
            "Hamming Distance (↓)",
            "Relative Euclidean Distance to True Weights (↓)",
            "Test MSE (↓)",
            "Relative Test MSE (↓)",
            "MAE (↓)",
            "MAPE (↓)",
            "Sparsity (Abs / Rel %)"
        ]

        table_vals = [
            [f"{similarities['jaccard'][m]:.6f}" for m in methods],
            [f"{similarities['hamming'][m]:.6f}" for m in methods],
            [f"{similarities['relative euclidean distance to ref'][m]:.6f}" for m in methods],
            [f"{test_metrics['mse'][m]:.6f}" for m in methods],
            [f"{test_metrics['relative_error'][m]:.6f}" for m in methods],
            [f"{test_metrics['mae'][m]:.6f}" for m in methods],
            [f"{test_metrics['mape'][m]:.6f}" for m in methods],
            [f"{abs_sparsity[m]:.2f} / {rel_sparsity[m]:.2f}" for m in methods]
        ]

        # Add table below plots
        ax_table = fig.add_axes([0.25, 0.05, 0.5, 0.3])  # X, Y, Width, Height
        ax_table.axis('off')

        table = ax_table.table(
            cellText=table_vals,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            rowLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

    else:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)  
    else:
        plt.show()



def plot_noise_curves(results_by_noise):
    noise_vals = results_by_noise["noise"]
    metrics = ["rel_mse_test", "rel_mse_train", "absolute_sparsity", "relative_sparsity", "jaccard", "hamming", "euclidean"]
    titles = {
        "jaccard": "Jaccard Similarity",
        "hamming": "Hamming Distance",
        "euclidean": "Euclidean Distance to True Weights",
        "rel_mse_test": "Relative MSE (Test)",
        "rel_mse_train": "Relative MSE (Train)",
        "absolute_sparsity": "Absolute Sparsity (%)",
        "relative_sparsity": "Relative Sparsity (%)"
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    methods = ["mco", "lasso", "spoq", "scad"]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for method in methods:
            if metric in results_by_noise and method in results_by_noise[metric]:
                ax.plot(noise_vals, results_by_noise[metric][method], label=method.upper())
        ax.set_title(titles.get(metric, metric))
        ax.set_xlabel("Noise level")
        ax.set_ylabel(titles.get(metric, metric))
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    os.makedirs("plots/custom_dataset", exist_ok=True)
    plt.savefig("plots/custom_dataset/impact_of_noise.png")
    plt.show()



def plot_noise_curves_advanced(results_by_noise, output_path="plots/noise_impact_plot.png", save_plot=False):
    noise_values = results_by_noise["noise"]

    # Define known metrics and their display names
    known_metrics = {
        "relative_mse_test": "Relative Test MSE",
        "relative_mse_train": "Relative Train MSE",
        "mae_test": "Mean Absolute Error (Test)",
        # "mae_train": "Mean Absolute Error (Train)",
        "mape_test": "Mean Absolute Percentage Error (Test)",
        # "mape_train": "Mean Absolute Percentage Error (Train)",
        "absolute_sparsity": "Absolute Sparsity (%)",
        "relative_sparsity": "Relative Sparsity (%)",
        "jaccard": "Jaccard Similarity",
        "hamming": "Hamming Distance",
        "euclidean": "Euclidean Distance to True Weights"
    }

    methods = ["mco", "lasso", "spoq", "scad"]
    available_metrics = [m for m in known_metrics if m in results_by_noise]

    # Set up the layout: include lambda plots as extra plots
    n_cols = 3
    n_extra_plots = 3  # For lambda_pen plots: lasso, spoq, scad
    n_rows = (len(available_metrics) + n_extra_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    # === Plot metrics with error bars ===
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        for method in methods:
            if method in results_by_noise[metric]:
                data = results_by_noise[metric][method]
                means = [np.mean(d) for d in data]
                stds = [np.std(d) for d in data]

                ax.errorbar(noise_values, means, yerr=stds, label=method.upper(),
                            fmt='-o', capsize=4, markersize=5, elinewidth=1)
        ax.set_title(known_metrics[metric])
        ax.set_xlabel("Noise Level")
        ax.set_ylabel(known_metrics[metric])
        ax.grid(True)
        ax.legend()

    # === λ LASSO plot ===
    ax = axes[len(available_metrics)]
    lasso_lambdas = [np.mean(l) for l in results_by_noise["lambda_pen_lasso"]]
    lasso_std = [np.std(l) for l in results_by_noise["lambda_pen_lasso"]]

    ax.errorbar(noise_values, lasso_lambdas, yerr=lasso_std, fmt='-o', capsize=4,
                label="LASSO λ", color='orange')
    ax.set_title("λ_lasso vs Noise Level")
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("λ (lambda_pen)")
    ax.grid(True)
    ax.legend()

    # === λ SPOQ plot ===
    ax = axes[len(available_metrics) + 1]
    spoq_lambdas = [np.mean(l) for l in results_by_noise["lambda_pen_spoq"]]
    spoq_std = [np.std(l) for l in results_by_noise["lambda_pen_spoq"]]

    ax.errorbar(noise_values, spoq_lambdas, yerr=spoq_std, fmt='-s', capsize=4,
                label="SPOQ λ", color='green')
    ax.set_title("λ_spoq vs Noise Level")
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("λ (lambda_pen)")
    ax.grid(True)
    ax.legend()

    # === λ SCAD plot ===
    ax = axes[len(available_metrics) + 2]
    scad_lambdas = [np.mean(l) for l in results_by_noise["lambda_pen_scad"]]
    scad_std = [np.std(l) for l in results_by_noise["lambda_pen_scad"]]

    ax.errorbar(noise_values, scad_lambdas, yerr=scad_std, fmt='-^', capsize=4,
                label="SCAD λ", color='purple')
    ax.set_title("λ_scad vs Noise Level")
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("λ (lambda_pen)")
    ax.grid(True)
    ax.legend()

    # === Remove unused axes if any ===
    for j in range(len(available_metrics) + n_extra_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save the plot to file if required
    if save_plot:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

    plt.show()



def plot_train_size_curves(results_by_train_size, output_path="plots/train_size_plot.png"):
    train_sizes = results_by_train_size["train_size"]
    known_metrics = {
        "mse_test": "Test MSE",
        "relative_mse_train": "Relative Train MSE",
        "sparsity": "Sparsity (%)",
        "jaccard": "Jaccard Similarity",
        "hamming": "Hamming Distance",
        "euclidean": "Euclidean Distance to true weights"
    }

    available_metrics = [m for m in known_metrics if m in results_by_train_size]

    n_cols = 3
    n_rows = (len(available_metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        for model in results_by_train_size[metric]:
            ax.plot(train_sizes, results_by_train_size[metric][model], label=model.upper())
        ax.set_title(known_metrics[metric])
        ax.set_xlabel("Train Size")
        ax.set_ylabel(known_metrics[metric])
        ax.grid(True)
        ax.legend()

    for j in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[j])  # remove unused subplots

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()



def plot_train_size_curves_advanced(results_by_train_size, output_path="plots/train_size_plot.png", save_plot=False):
    train_sizes = results_by_train_size["train_size"]

    # Define all known metrics to be plotted
    known_metrics = {
        "relative_mse_test": "Relative Test MSE",
        "relative_mse_train": "Relative Train MSE",
        "mae_test": "Mean Absolute Error (Test)",
        "mae_train": "Mean Absolute Error (Train)",
        "mape_test": "Mean Absolute Percentage Error (Test)",
        "mape_train": "Mean Absolute Percentage Error (Train)",
        "absolute_sparsity": "Absolute Sparsity (%)",
        "relative_sparsity": "Relative Sparsity (%)",
        "jaccard": "Jaccard Similarity",
        "hamming": "Hamming Distance",
        "euclidean": "Euclidean Distance to True Weights"
    }

    # Filter metrics present in results
    available_metrics = [m for m in known_metrics if m in results_by_train_size]

    # Determine subplot layout
    n_cols = 3
    n_extra_plots = 3  # For λ_lasso, λ_spoq, λ_scad
    n_rows = (len(available_metrics) + n_extra_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    methods = ["mco", "lasso", "spoq", "scad"]

    # === Plot each available metric ===
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        for model in methods:
            if model in results_by_train_size[metric]:
                data = results_by_train_size[metric][model]
                means = [np.mean(d) for d in data]
                stds = [np.std(d) for d in data]
                ax.errorbar(train_sizes, means, yerr=stds, label=model.upper(),
                            fmt='-o', capsize=4, markersize=5, elinewidth=1)
        ax.set_title(known_metrics[metric])
        ax.set_xlabel("Train Size")
        ax.set_ylabel(known_metrics[metric])
        ax.grid(True)
        ax.legend()

    # === Plot λ_lasso ===
    ax = axes[len(available_metrics)]
    lasso_lambdas = [np.mean(l) for l in results_by_train_size["lambda_pen_lasso"]]
    lasso_std = [np.std(l) for l in results_by_train_size["lambda_pen_lasso"]]
    ax.errorbar(train_sizes, lasso_lambdas, yerr=lasso_std, fmt='-o', capsize=4,
                label="LASSO λ", color='orange')
    ax.set_title("λ_lasso vs Train Size")
    ax.set_xlabel("Train Size")
    ax.set_ylabel("λ (lambda_pen)")
    ax.grid(True)
    ax.legend()

    # === Plot λ_spoq ===
    ax = axes[len(available_metrics) + 1]
    spoq_lambdas = [np.mean(l) for l in results_by_train_size["lambda_pen_spoq"]]
    spoq_std = [np.std(l) for l in results_by_train_size["lambda_pen_spoq"]]
    ax.errorbar(train_sizes, spoq_lambdas, yerr=spoq_std, fmt='-s', capsize=4,
                label="SPOQ λ", color='green')
    ax.set_title("λ_spoq vs Train Size")
    ax.set_xlabel("Train Size")
    ax.set_ylabel("λ (lambda_pen)")
    ax.grid(True)
    ax.legend()

    # === Plot λ_scad ===
    ax = axes[len(available_metrics) + 2]
    scad_lambdas = [np.mean(l) for l in results_by_train_size["lambda_pen_scad"]]
    scad_std = [np.std(l) for l in results_by_train_size["lambda_pen_scad"]]
    ax.errorbar(train_sizes, scad_lambdas, yerr=scad_std, fmt='-^', capsize=4,
                label="SCAD λ", color='blue')
    ax.set_title("λ_scad vs Train Size")
    ax.set_xlabel("Train Size")
    ax.set_ylabel("λ (lambda_pen)")
    ax.grid(True)
    ax.legend()

    # === Remove any unused subplots ===
    for j in range(len(available_metrics) + n_extra_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save the figure if requested
    if save_plot:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)

    plt.show()


def plot_lambda_sensitivity(
    mse_spoq, mse_lasso, mse_scad,
    mae_spoq, mae_lasso, mae_scad,
    lambda_range_spoq, lambda_range_lasso, lambda_range_scad,
    range_value_percent=None
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)

    def compute_var_percent(errors, lambda_range):
        mid_idx = len(lambda_range) // 2
        val_mid = errors[mid_idx]
        val_min = np.min(errors)
        val_max = np.max(errors)
        var_pct = 100 * (val_max - val_min) / val_mid if val_mid != 0 else 0
        return var_pct

    # === Row 1 : MSE ===
    axes[0, 0].plot(lambda_range_spoq, mse_spoq, marker='.', color='green', label='SPOQ')
    var_spoq_mse = compute_var_percent(mse_spoq, lambda_range_spoq)
    axes[0, 0].set_title('SPOQ – MSE')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_xlabel(r'$\lambda$')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    axes[0, 0].axvline(lambda_range_spoq[len(lambda_range_spoq) // 2], color='black', linestyle='--', alpha=0.7)
    axes[0, 0].text(0.95, 0.85, f'Var: {var_spoq_mse:.1f}%', transform=axes[0, 0].transAxes,
                   ha='right', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    axes[0, 1].plot(lambda_range_lasso, mse_lasso, marker='.', color='orange', label='LASSO')
    var_lasso_mse = compute_var_percent(mse_lasso, lambda_range_lasso)
    axes[0, 1].set_title('LASSO – MSE')
    axes[0, 1].set_xlabel(r'$\lambda$')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    axes[0, 1].axvline(lambda_range_lasso[len(lambda_range_lasso) // 2], color='black', linestyle='--', alpha=0.7)
    axes[0, 1].text(0.95, 0.85, f'Var: {var_lasso_mse:.1f}%', transform=axes[0, 1].transAxes,
                   ha='right', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    axes[0, 2].plot(lambda_range_scad, mse_scad, marker='.', color='red', label='SCAD')
    var_scad_mse = compute_var_percent(mse_scad, lambda_range_scad)
    axes[0, 2].set_title('SCAD – MSE')
    axes[0, 2].set_xlabel(r'$\lambda$')
    axes[0, 2].grid(True)
    axes[0, 2].legend()
    axes[0, 2].axvline(lambda_range_scad[len(lambda_range_scad) // 2], color='black', linestyle='--', alpha=0.7)
    axes[0, 2].text(0.95, 0.85, f'Var: {var_scad_mse:.1f}%', transform=axes[0, 2].transAxes,
                   ha='right', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    # === Row 2 : MAE ===
    axes[1, 0].plot(lambda_range_spoq, mae_spoq, marker='.', color='green', label='SPOQ')
    var_spoq_mae = compute_var_percent(mae_spoq, lambda_range_spoq)
    axes[1, 0].set_title('SPOQ – MAE')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_xlabel(r'$\lambda$')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    axes[1, 0].axvline(lambda_range_spoq[len(lambda_range_spoq) // 2], color='black', linestyle='--', alpha=0.7)
    axes[1, 0].text(0.95, 0.85, f'Var: {var_spoq_mae:.1f}%', transform=axes[1, 0].transAxes,
                   ha='right', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    axes[1, 1].plot(lambda_range_lasso, mae_lasso, marker='.', color='orange', label='LASSO')
    var_lasso_mae = compute_var_percent(mae_lasso, lambda_range_lasso)
    axes[1, 1].set_title('LASSO – MAE')
    axes[1, 1].set_xlabel(r'$\lambda$')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    axes[1, 1].axvline(lambda_range_lasso[len(lambda_range_lasso) // 2], color='black', linestyle='--', alpha=0.7)
    axes[1, 1].text(0.95, 0.85, f'Var: {var_lasso_mae:.1f}%', transform=axes[1, 1].transAxes,
                   ha='right', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    axes[1, 2].plot(lambda_range_scad, mae_scad, marker='.', color='red', label='SCAD')
    var_scad_mae = compute_var_percent(mae_scad, lambda_range_scad)
    axes[1, 2].set_title('SCAD – MAE')
    axes[1, 2].set_xlabel(r'$\lambda$')
    axes[1, 2].grid(True)
    axes[1, 2].legend()
    axes[1, 2].axvline(lambda_range_scad[len(lambda_range_scad) // 2], color='black', linestyle='--', alpha=0.7)
    axes[1, 2].text(0.95, 0.85, f'Var: {var_scad_mae:.1f}%', transform=axes[1, 2].transAxes,
                   ha='right', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    if range_value_percent is not None:
        fig.suptitle(f'Sensitivity to Lambda — ±{100*range_value_percent:.1f}% around best_lambda', fontsize=16)
        plt.subplots_adjust(top=0.92)

    plt.tight_layout()
    plt.show()
