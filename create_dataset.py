import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from utils.algorithms import check_and_add_bias_column, standardize_except_bias

def create_dataset(output_path="data/custom_dataset.csv", save = True, noise_design="std", **kwargs):

    # Extract specific parameters
    noise_ratio = kwargs.pop("noise", 0.0)  # relative noise level (fraction of target std)
    random_state = kwargs.get("random_state", None)

    # Generate X, y without noise or bias
    X, y, custom_weights = make_regression(noise=0.0, **kwargs)

    if noise_design == "std":
        # Compute signal amplitude after adding bias
        signal_std = np.std(y)

        # Add noise proportional to the signal amplitude
        rng = np.random.default_rng(random_state)
        noise = rng.normal(loc=0.0, scale=noise_ratio * signal_std, size=y.shape)
        y += noise

    if noise_design == "median":
        # Compute signal amplitude after adding bias
        median_of_abs = np.median(np.abs(y))

        # Add noise proportional to the signal amplitude
        rng = np.random.default_rng(random_state)
        noise = rng.normal(loc=0.0, scale=noise_ratio * median_of_abs, size=y.shape)
        y += noise

    # Add bias column (constant 1) to X and standardize (except bias)
    X = check_and_add_bias_column(X)
    X = standardize_except_bias(X)

    # Insert bias as the first weight
    custom_weights = np.insert(custom_weights, 0, kwargs["bias"])

    # Create DataFrame
    feature_names = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Save to CSV
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        # Save weights
        np.save(file="data/weights.npy", arr=custom_weights)

        print(f"Dataset saved to {output_path}")
    return df, custom_weights

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a custom regression dataset.")

    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=100, help="Total number of features")
    parser.add_argument("--n_informative", type=int, default=20, help="Number of informative features")
    parser.add_argument("--noise", type=float, default=0.1, help="Relative noise level: fraction of the target's std deviation (e.g., 0.1 = 10% of signal std)")
    parser.add_argument("--bias", type=float, default=10.0, help="Bias term in the underlying linear model")
    parser.add_argument("--coef", default=True, action="store_true", help="Return the coefficients of the underlying linear model")
    parser.add_argument("--effective_rank", type=int, default=None, help="Approximate number of singular vectors for input matrix.")
    parser.add_argument("--tail_strength", type=float, default=0.5, help="Relative importance of the fat noisy tail of the singular values profile.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_path", type=str, default="data/custom_dataset.csv", help="Output CSV path")

    args = parser.parse_args()
    create_dataset(**vars(args))
