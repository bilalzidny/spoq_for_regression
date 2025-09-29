import os
import logging
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt, image as mpimg

from utils.logger import save_results
from run_on_custom import run_on_custom
from utils.plots import plot_noise_curves_advanced

# === SETUP LOGGING ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/experiment_noise.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def evaluate_noise_impact_advanced(noise_values, n_runs=10, random_state=42, **base_kwargs):
    """
    Runs multiple regression models on synthetic data to evaluate the impact of varying noise levels.

    Metrics include relative MSE, MAE, MAPE, sparsity, and similarity to ground truth.
    Results are averaged over multiple runs and saved, along with a plot.

    Parameters:
        noise_values (list): List of noise ratio to evaluate.
        n_runs (int): Number of repetitions per noise level.
        random_state (int): Base seed for reproducibility.
        **base_kwargs: Parameters for synthetic dataset generation.
    """

    output_path = "plots/custom_dataset/mse_sparsity_vs_noise.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path): 
        answer = input(f"The figure '{output_path}' already exists. Do you want to rerun the experiment? (y/n): ").strip().lower()
        if answer not in ["y", "yes"]:
            show_plot = input("Do you want to display the existing figure? (y/n): ").strip().lower()
            if show_plot in ["y", "yes"]:
                img = mpimg.imread(output_path)
                plt.figure(figsize=(18, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title("Previously saved figure")
                plt.tight_layout()
                plt.show()
            else:
                print("Experiment skipped.")
            return
        
    logging.info("STARTED NOISE IMPACT EXPERIMENT")

    methods = ["mco", "lasso", "spoq", "scad"]
    
    results_by_noise = {
        "noise": noise_values,
        "relative_mse_test": {m: [[] for _ in noise_values] for m in methods},
        "relative_mse_train": {m: [[] for _ in noise_values] for m in methods},
        "mae_test": {m: [[] for _ in noise_values] for m in methods},
        "mae_train": {m: [[] for _ in noise_values] for m in methods},
        "mape_test": {m: [[] for _ in noise_values] for m in methods},
        "mape_train": {m: [[] for _ in noise_values] for m in methods},
        "absolute_sparsity": {m: [[] for _ in noise_values] for m in methods},
        "relative_sparsity": {m: [[] for _ in noise_values] for m in methods},
        "jaccard": {m: [[] for _ in noise_values] for m in methods},
        "hamming": {m: [[] for _ in noise_values] for m in methods},
        "euclidean": {m: [[] for _ in noise_values] for m in methods},
        "confusion_matrix": {m: [[] for _ in noise_values] for m in methods},
        "lambda_pen_lasso": [[] for _ in noise_values],
        "lambda_pen_spoq": [[] for _ in noise_values],
        "lambda_pen_scad": [[] for _ in noise_values],
    }

    for i, noise in enumerate(tqdm(noise_values, desc="Noise levels")):
        print(f"\nRunning experiments for noise = {noise:.2f}")
        logging.info(f"RUNNING EXPERIMENT FOR NOISE = {noise:.2f}")

        for run in range(n_runs):
            run_seed = random_state + run
            current_kwargs = base_kwargs.copy()
            current_kwargs.update({
                "noise": noise,
                "random_state": run_seed
            })

            logging.info(f"RUN n°{run} for noise: {noise:.2f}")

            result = run_on_custom(
                plot=False,
                log_results=True,
                tuning="optuna",
                w_ref=None,
                **current_kwargs
            )

            for model in methods:
                # Metrics
                results_by_noise["relative_mse_test"][model][i].append(result["metrics"]["test"]["relative_error"][model])
                results_by_noise["relative_mse_train"][model][i].append(result["metrics"]["train"]["relative_error"][model])
                results_by_noise["mae_test"][model][i].append(result["metrics"]["test"]["mae"][model])
                results_by_noise["mae_train"][model][i].append(result["metrics"]["train"]["mae"][model])
                results_by_noise["mape_test"][model][i].append(result["metrics"]["test"]["mape"][model])
                results_by_noise["mape_train"][model][i].append(result["metrics"]["train"]["mape"][model])

                # Sparsity & Similarities
                results_by_noise["absolute_sparsity"][model][i].append(result["sparsity"]["absolute"][model])
                results_by_noise["relative_sparsity"][model][i].append(result["sparsity"]["relative"][model])
                results_by_noise["jaccard"][model][i].append(result["similarities"]["jaccard"][model])
                results_by_noise["hamming"][model][i].append(result["similarities"]["hamming"][model])
                results_by_noise["euclidean"][model][i].append(result["similarities"]["relative euclidean distance to ref"][model])
                results_by_noise["confusion_matrix"][model][i].append(result["confusion_matrices"][model])

            # Hyperparams
            results_by_noise["lambda_pen_lasso"][i].append(result["params"]["fista"]["lambda_pen"])
            results_by_noise["lambda_pen_spoq"][i].append(result["params"]["trust_region"]["lambda_pen"])
            results_by_noise["lambda_pen_scad"][i].append(result["params"]["scad"]["lambda_pen"])

    save_results(results_by_noise, output_dir="results", file_prefix="noise_impact_advanced")
    plot_noise_curves_advanced(results_by_noise, output_path, save_plot=True)

    print("Experiment completed. Results saved and figure generated.")


if __name__ == "__main__":

    # base_kwargs = {
    #     "n_samples": 1000,
    #     "n_features": 100,
    #     "n_informative": 20,
    #     "bias": 10.0,
    #     "coef": True,
    #     "random_state": 42,
    #     "effective_rank": None,
    #     "tail_strength": 0.5,
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

    noise_values = np.linspace(0, 0.4, 17).tolist()

    evaluate_noise_impact_advanced(noise_values=noise_values, n_runs=10, **base_kwargs)