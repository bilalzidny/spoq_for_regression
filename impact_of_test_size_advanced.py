import os
import logging
import numpy as np
import argparse
import json
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime  # <--- AJOUT ICI

from utils.logger import save_results
from utils.algorithms import load_and_preprocess
from run_results import run_results_optuna
from run_on_custom import run_on_custom
from create_dataset import create_dataset

# === UTILS JSON ===
class NumpyEncoder(json.JSONEncoder):
    """ Permet de sauvegarder les types Numpy (float32, ndarray) dans un JSON standard """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# === SETUP LOGGING ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/experiment_train_size.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# === FONCTION WORKER ===
def _run_single_train_size_experiment(ts_idx, ts_val, run_idx, random_state, 
                                      X_train_full, y_train_full, X_test, y_test, 
                                      w_ref, is_custom, base_kwargs):
    """
    Exécute une simulation pour une taille d'entraînement donnée.
    """
    seed = random_state + run_idx
    rng = np.random.RandomState(seed)

    # 1. Sous-échantillonnage
    n_samples_train = int(ts_val * len(X_train_full))
    if n_samples_train < 5: n_samples_train = 5

    sel_indices = rng.choice(len(X_train_full), size=n_samples_train, replace=False)
    Xt = X_train_full[sel_indices]
    yt = y_train_full[sel_indices]

    # 2. Benchmark
    if is_custom:
        result = run_on_custom(
            plot=False,
            log_results=False, 
            tuning="optuna",
            w_ref=w_ref,
            X_train=Xt, y_train=yt, 
            X_test=X_test, y_test=y_test,
            lambda_range=np.logspace(-1, 7, 50),
            n_trials=100, 
            verbose=False,
            **base_kwargs
        )
    else:
        result = run_results_optuna(
            file=None,
            target_name=None,
            train_size=None, test_size=None,
            random_state=seed,
            log_results=False, 
            return_results=True, 
            verbose=False,
            X_train=Xt, y_train=yt, 
            X_test=X_test, y_test=y_test,
            n_trials=100
        )

    return ts_idx, result


# === FONCTION PRINCIPALE AVEC SAUVEGARDE TEMPS RÉEL ===
def evaluate_train_size_impact_parallel(file, target_name, train_sizes, random_state=42, n_runs=10, n_jobs=-1, **base_kwargs):
    
    # --- MODIFICATION ICI : AJOUT DU TIMESTAMP ---
    # Format: YYYYMMDD_HHMMSS (ex: 20231027_143000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # On nettoie le nom du fichier pour enlever l'extension .csv si présente
    clean_filename = str(file).split(os.sep)[-1].split('.')[0]
    
    file_prefix = f"train_size_{clean_filename}_parallel_{timestamp}"
    json_path = os.path.join("results", f"{file_prefix}_partial.json")
    os.makedirs("results", exist_ok=True)
    
    is_custom = "custom_dataset" in str(file).lower()

    logging.info(f"STARTED TRAIN SIZE IMPACT EXPERIMENT (ID: {timestamp})")
    print(f"Starting parallel execution with n_jobs={n_jobs}...")
    print(f"Partial results will be saved to: {json_path}")

    methods = ["mco", "lasso", "spoq", "scad", "reweighted", "mcp", "irls"]

    # --- 1. DATASET SETUP ---
    print("Generating/Loading full dataset...")
    w_ref = None
    if is_custom:
        df, w_ref = create_dataset(save=False, **base_kwargs)
        y = df["target"].to_numpy()
        X = df.drop(columns=["target"]).to_numpy()
    else:
        X, y = load_and_preprocess(file, target_name=target_name, verbose=False)
        if os.path.exists("data/weights.npy"):
            w_ref = np.load("data/weights.npy")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=random_state
    )

    # --- 2. INITIALISATION STRUCTURE RÉSULTATS ---
    results = {
        "train_size": train_sizes,
        "meta": base_kwargs,
        "timestamp": timestamp, # On stocke aussi le timestamp dans le JSON
        "relative_mse_test": {m: [[] for _ in train_sizes] for m in methods},
        "relative_mse_train": {m: [[] for _ in train_sizes] for m in methods},
        "mae_test": {m: [[] for _ in train_sizes] for m in methods},
        "absolute_sparsity": {m: [[] for _ in train_sizes] for m in methods},
        "relative_sparsity": {m: [[] for _ in train_sizes] for m in methods},
        
        "lambda_pen_lasso": [[] for _ in train_sizes],
        "lambda_pen_spoq": [[] for _ in train_sizes],
        "lambda_pen_scad": [[] for _ in train_sizes],
        "lambda_pen_reweighted": [[] for _ in train_sizes],
        "lambda_pen_mcp": [[] for _ in train_sizes],
        "lambda_pen_irls": [[] for _ in train_sizes],
    }

    if is_custom:
        for metric in ["jaccard", "hamming", "euclidean", "confusion_matrix"]:
            results[metric] = {m: [[] for _ in train_sizes] for m in methods}

    # --- 3. PRÉPARATION TÂCHES ---
    tasks = []
    for i, ts in enumerate(train_sizes):
        for run in range(n_runs):
            tasks.append((i, ts, run))

    total_tasks = len(tasks)

    # --- 4. EXÉCUTION AVEC SAUVEGARDE TEMPS RÉEL ---
    try:
        with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
            
            results_generator = parallel(
                delayed(_run_single_train_size_experiment)(
                    i, ts, run, random_state, 
                    X_train_full, y_train_full, X_test, y_test, 
                    w_ref, is_custom, base_kwargs
                )
                for i, ts, run in tasks
            )

            # Compteur pour sauvegarde périodique (toutes les 5 tâches)
            save_counter = 0

            for idx, res in tqdm(results_generator, total=total_tasks, desc="Computing & Saving"):
                
                # --- AGREGATION ---
                for m in methods:
                    results["relative_mse_test"][m][idx].append(res["metrics"]["test"]["relative_error"][m])
                    results["relative_mse_train"][m][idx].append(res["metrics"]["train"]["relative_error"][m])
                    results["mae_test"][m][idx].append(res["metrics"]["test"]["mae"][m])
                    results["absolute_sparsity"][m][idx].append(res["sparsity"]["absolute"][m])
                    results["relative_sparsity"][m][idx].append(res["sparsity"]["relative"][m])

                    if is_custom:
                        results["jaccard"][m][idx].append(res["similarities"]["jaccard"][m])
                        results["hamming"][m][idx].append(res["similarities"]["hamming"][m])
                        results["euclidean"][m][idx].append(res["similarities"]["relative euclidean distance to ref"][m])
                        results["confusion_matrix"][m][idx].append(res["confusion_matrices"][m])

                results["lambda_pen_lasso"][idx].append(res["params"]["fista"].get("lambda_pen"))
                results["lambda_pen_spoq"][idx].append(res["params"]["trust_region"].get("lambda_pen"))
                results["lambda_pen_scad"][idx].append(res["params"]["scad"].get("lambda_pen"))
                results["lambda_pen_reweighted"][idx].append(res["params"]["reweighted"].get("lambda_pen"))
                results["lambda_pen_mcp"][idx].append(res["params"]["mcp"].get("lambda_pen"))
                results["lambda_pen_irls"][idx].append(res["params"]["irls"].get("lambda_pen"))

                save_counter += 1

                # --- SAUVEGARDE PÉRIODIQUE ---
                if save_counter % 5 == 0 or save_counter == total_tasks:
                    with open(json_path, "w") as f:
                        json.dump(results, f, indent=4, cls=NumpyEncoder)

    except KeyboardInterrupt:
        print("\nInterruption détectée ! Les résultats partiels sont sauvegardés dans :", json_path)
        raise
    except Exception as e:
        print(f"\nErreur critique : {e}")
        print("Les résultats partiels sont sauvegardés dans :", json_path)
        raise

    # --- 5. FINALISATION ---
    final_json_path = os.path.join("results", f"{file_prefix}_final.json")
    if os.path.exists(json_path):
        os.rename(json_path, final_json_path)
    
    print(f"Experiment completed. Final results saved to: {final_json_path}")


if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="custom_dataset.csv")
    parser.add_argument("--target_name", type=str, default="target")
    args = parser.parse_args()

    base_kwargs = {
        "n_samples": 1000,
        "n_features": 100,
        "n_informative": 20,
        "bias": 10,
        "coef": True,
        "random_state": 42,
        "effective_rank": None,
        "tail_strength": 0.5,
        "noise_design": "median",
        "noise": 0.1
    }
    
    # train_sizes = np.linspace(0.15, 0.8, 14)
    train_sizes = [0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09,
                   0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    evaluate_train_size_impact_parallel(
        file=args.file, target_name=args.target_name,
        train_sizes=train_sizes, n_runs=10, n_jobs=-1, **base_kwargs)