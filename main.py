import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from utils.algorithms import load_and_preprocess, MCO, mm_algorithm_spoqreg, tune_model_optuna
from utils.metrics import compute_mae, compute_relative_sse, compute_mape

# === SETUP LOGGING ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/experiment.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def run_spoq_reg(file, target_name=None, train_size=0.8, test_size=0.2,
                 tuning=True, lambda_val=None,
                 lambda_range=np.logspace(-1, 7),
                 random_state=42, n_trials=200,
                 fit_full_data=False, save_weights=False):
    """
    Run SPOQ_reg on a dataset and return the estimated weights.
    Can either train/test split, or train on full data if fit_full_data=True.

    Args:
        file (str): Path to dataset.
        target_name (str): Target variable name.
        tuning (bool): If True, perform Optuna tuning to find lambda.
        lambda_val (float): If tuning=False, use this lambda value.
        lambda_range (np.array): Range of lambda values for tuning.
        random_state (int): Random seed.
        n_trials (int): Number of Optuna trials if tuning.
        fit_full_data (bool): If True, fit model on full dataset (no testing).
        save_weights (bool): If True, save the estimated weights to .npy file.

    Returns:
        np.ndarray: Estimated weights (w_spoq).
    """
    np.random.seed(random_state)

    # === Load data ===
    X, y = load_and_preprocess(file, target_name=target_name, verbose=False)

    if fit_full_data:
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, test_size=test_size, random_state=random_state
        )

    # === Initial solution (MCO) ===
    w_0 = MCO(X_train, y_train)[0]

    # === Lambda tuning ===
    if tuning:
        best_params, _, _, _, _ = tune_model_optuna(
            model_fn=mm_algorithm_spoqreg,
            lambda_bounds=(lambda_range.min(), lambda_range.max()),
            X=X_train,
            y=y_train,
            fixed_params={
                "w_0": w_0,
                "B": 15,
                "theta": 0.5,
                "epsilon": 1e-5,
                "max_iter": 50000
            },
            scoring="aic",
            verbose=False,
            n_trials=n_trials
        )
    else:
        if lambda_val is None:
            raise ValueError("If tuning is disabled, please provide --lambda_val.")
        best_params = {
            "lambda_pen": lambda_val,
            "B": 15,
            "theta": 0.5,
            "epsilon": 1e-5,
            "max_iter": 50000
        }

    # === Run SPOQ_reg ===
    w_spoq, _, _, _, abs_sparsity_list, rel_sparsity_list, _, _, _ = mm_algorithm_spoqreg(
        **best_params,
        w_0=w_0,
        X_train=X_train,
        y_train=y_train,
        X_val=X_train if fit_full_data else X_test,
        y_val=y_train if fit_full_data else y_test,
        verbose=False
    )

    # === Metrics (optional) ===
    abs_sparsity = abs_sparsity_list[-1]
    rel_sparsity = rel_sparsity_list[-1]

    if not fit_full_data:
        mae = compute_mae(w_spoq, X_test, y_test)
        mape = compute_mape(w_spoq, X_test, y_test)
        rte = compute_relative_sse(w_spoq, X_test, y_test)

        print("\nSPOQ_reg Results")
        print("-------------------")
        print(f"MAE (test)              : {mae:.4f}")
        print(f"MAPE (test)             : {mape:.4f}")
        print(f"Relative Test Error     : {rte:.4f}")
    else:
        print("\nSPOQ_reg trained on full dataset.")

    print(f"Absolute Sparsity       : {abs_sparsity:.2f}% non-zeros")
    print(f"Relative Sparsity       : {rel_sparsity:.2f}% non-zero coefficients\n")

    # === Save weights ===
    if save_weights:
        os.makedirs("weights", exist_ok=True)
        np.save("weights/weights_spoq.npy", w_spoq)
        print("Weights saved to logs/weights_spoq.npy")

    return w_spoq



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SPOQ_reg and display solution + metrics.")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target_name", type=str, required=True, help="Target column name")

    parser.add_argument("--tuning", type=lambda x: x.lower() == "true", default=True,
                        help="Use Optuna to tune lambda (True/False)")
    parser.add_argument("--lambda_val", type=float, default=None,
                        help="Lambda value to use if tuning is False")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--save_weights", type=lambda x: x.lower() == "true", default=False,
                    help="Save estimated weights to file (True/False)")
    parser.add_argument("--fit_full_data", action="store_true",
                    help="If set, trains the model on the full dataset without train/test split.")


    args = parser.parse_args()

    w_spoq = run_spoq_reg(**vars(args))

