import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
import json

# Add the project root (spoq_for_reg) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from utils.algorithms import *
from utils.metrics import compute_mae, compute_mse




def plot_test_errors(mae_mse_dict):
    """
    Plots MAE and MSE over iterations for each method.
    """
    methods = ["SPOQ", "LASSO", "SCAD"]
    colors = {"SPOQ": "green", "LASSO": "orange", "SCAD": "red"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for method in methods:
        mae = mae_mse_dict[method]["mae"]
        mse = mae_mse_dict[method]["mse"]
        iterations = list(range(len(mae)))

        axes[0].plot(iterations, mae, label=method, color=colors[method])
        axes[1].plot(iterations, mse, label=method, color=colors[method])

    # MAE Plot
    axes[0].set_title("Test MAE over Iterations")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("MAE")
    axes[0].legend()
    axes[0].grid(True)

    # MSE Plot
    axes[1].set_title("Test MSE over Iterations")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()



def get_test_errors_vs_iterations(file, target_name, lambda_range=np.logspace(-1, 6), test_size=0.3):
    """
    Trains SPOQ, LASSO, and SCAD on the training set and returns MAE and MSE per iteration on the test set.
    """

    if file == "custom_dataset.csv":
        true_weights = np.load("./data/weights.npy")
    else:
        true_weights = None

    # === LOAD AND SPLIT DATA ===
    X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # === INITIALIZATION (OLS)
    w_mco, _, _ = MCO(X_train, y_train)
    w_0_value = w_mco

    # === TUNE HYPERPARAMETERS FOR EACH METHOD ===
    best_params_spoq, _, _, _, _ = tune_model_optuna(
        model_fn=mm_algorithm_spoqreg,
        lambda_bounds=(lambda_range.min(), lambda_range.max()),
        X=X_train,
        y=y_train,
        fixed_params={"w_0": w_0_value, "B": 15, "theta": 0.5, "epsilon": 1e-5, "max_iter": 50000},
        scoring="aic",
        verbose=False
    )

    best_params_lasso, _, _, _, _ = tune_model_optuna(
        model_fn=fista_lasso,
        lambda_bounds=(lambda_range.min(), lambda_range.max()),
        X=X_train,
        y=y_train,
        fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000},
        scoring="aic",
        verbose=False
    )

    best_params_scad, _, _, _, _ = tune_model_optuna(
        model_fn=fista_scad,
        lambda_bounds=(lambda_range.min(), lambda_range.max()),
        X=X_train,
        y=y_train,
        fixed_params={"w_0": w_0_value, "epsilon": 1e-5, "max_iter": 5000},
        scoring="aic",
        verbose=False
    )

    # === COMPUTE ERRORS ===

    def compute_errors(model_fn, best_params):
        _, _, _, _, _, _, _, _, all_weights = model_fn(
            **best_params,
            w_0=w_0_value,
            X_train=X_train,
            y_train=y_train,
            X_val=X_train,
            y_val=y_train,
            verbose=False
        )
        mae_list, mse_list = [], []
        for w in all_weights:
            mae_list.append(compute_mae(w, X_test, y_test))
            mse_list.append(compute_mse(w, X_test, y_test))
        return mae_list, mse_list

    mae_spoq, mse_spoq = compute_errors(mm_algorithm_spoqreg, best_params_spoq)
    mae_lasso, mse_lasso = compute_errors(fista_lasso, best_params_lasso)
    mae_scad, mse_scad = compute_errors(fista_scad, best_params_scad)

    return {
        "SPOQ": {"mae": mae_spoq, "mse": mse_spoq},
        "LASSO": {"mae": mae_lasso, "mse": mse_lasso},
        "SCAD": {"mae": mae_scad, "mse": mse_scad}
    }

    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="bodyfat.csv", help="Path to the dataset")
    parser.add_argument("--target_name", type=str, default="BodyFat", help="Target column name")
    args = parser.parse_args()

    results = get_test_errors_vs_iterations(file=args.file, target_name=args.target_name)
    os.makedirs("results", exist_ok=True)
    output_filename = f"results/mse_vs_iteration_{args.file.replace('.csv', '')}.json"
    with open(output_filename, "w") as F:
        json.dump(results, F, indent=4)
    plot_test_errors(results)
