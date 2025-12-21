import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# Vos imports existants
from utils.logger import save_results
from run_on_custom import run_on_custom
from create_dataset import create_dataset  # <--- On utilise votre fonction !

# === SETUP LOGGING ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/experiment_noise.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# === FONCTION WORKER CORRIGÉE ===
def _run_single_experiment(noise_idx, noise_val, run_idx, random_state, base_kwargs):
    """
    Exécute une simulation EN MÉMOIRE (RAM).
    N'écrit aucun fichier CSV pour éviter les conflits et gagner du temps.
    """
    run_seed = random_state + run_idx
    
    # On prépare les arguments
    current_kwargs = base_kwargs.copy()
    current_kwargs.update({
        "noise": noise_val,        # Le niveau de bruit courant
        "random_state": run_seed,  # Seed unique pour ce run
        "output_path": None        # Pas de chemin de sortie nécessaire
    })

    # 1. GÉNÉRATION EN MÉMOIRE (save=False)
    # On utilise VOTRE fonction create_dataset qui gère le bruit "median/std", 
    # l'ajout de la colonne de biais et la standardisation.
    df, w_ref = create_dataset(save=False, **current_kwargs)

    # 2. CONVERSION DATAFRAME -> NUMPY
    # Votre create_dataset retourne un DataFrame, il faut extraire X et y
    y = df["target"].to_numpy()
    X = df.drop(columns=["target"]).to_numpy()

    # 3. SPLIT TRAIN/TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42
    )

    # 4. LANCEMENT DU BENCHMARK
    # On passe directement les données numpy (X_train, etc.) et w_ref
    result = run_on_custom(
        plot=False,
        log_results=False, 
        tuning="optuna", 
        w_ref=w_ref,      
        X_train=X_train,  
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_trials=100, 
        verbose=False,
        **current_kwargs
    )
    
    return noise_idx, result


def evaluate_noise_impact_advanced_parallel(noise_values, n_runs=10, random_state=42, n_jobs=-1, **base_kwargs):
    """
    Version parallélisée optimisée sans I/O disque.
    """
    
    # (Le début de la fonction reste identique : vérification fichier existant, etc.)
    output_path = "plots/custom_dataset/mse_sparsity_vs_noise.png"
    # ... [Code de vérification fichier existant coupé pour brièveté] ...

    logging.info("STARTED NOISE IMPACT EXPERIMENT (PARALLEL)")
    print(f"Starting parallel execution with n_jobs={n_jobs}...")

    methods = ["mco", "lasso", "spoq", "scad", "reweighted", "mcp", "irls"]
    
    # --- 1. Préparation des tâches ---
    tasks = []
    for i, noise in enumerate(noise_values):
        for run in range(n_runs):
            tasks.append((i, noise, run))

    # --- 2. Exécution Parallèle ---
    # Joblib va distribuer les appels à _run_single_experiment
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_experiment)(i, noise, run, random_state, base_kwargs)
        for i, noise, run in tqdm(tasks, desc="Parallel Simulations")
    )

    # --- 3. Initialisation du dictionnaire de résultats ---
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
        "lambda_pen_reweighted": [[] for _ in noise_values],
        "lambda_pen_mcp": [[] for _ in noise_values],
        "lambda_pen_irls": [[] for _ in noise_values],
    }

    # --- 4. Agrégation des résultats ---
    print("Aggregating results...")
    for noise_idx, result in parallel_results:
        
        for model in methods:
            # Metrics
            results_by_noise["relative_mse_test"][model][noise_idx].append(result["metrics"]["test"]["relative_error"][model])
            results_by_noise["relative_mse_train"][model][noise_idx].append(result["metrics"]["train"]["relative_error"][model])
            results_by_noise["mae_test"][model][noise_idx].append(result["metrics"]["test"]["mae"][model])
            results_by_noise["mae_train"][model][noise_idx].append(result["metrics"]["train"]["mae"][model])
            results_by_noise["mape_test"][model][noise_idx].append(result["metrics"]["test"]["mape"][model])
            results_by_noise["mape_train"][model][noise_idx].append(result["metrics"]["train"]["mape"][model])

            # Sparsity & Similarity
            results_by_noise["absolute_sparsity"][model][noise_idx].append(result["sparsity"]["absolute"][model])
            results_by_noise["relative_sparsity"][model][noise_idx].append(result["sparsity"]["relative"][model])
            results_by_noise["jaccard"][model][noise_idx].append(result["similarities"]["jaccard"][model])
            results_by_noise["hamming"][model][noise_idx].append(result["similarities"]["hamming"][model])
            results_by_noise["euclidean"][model][noise_idx].append(result["similarities"]["relative euclidean distance to ref"][model])
            results_by_noise["confusion_matrix"][model][noise_idx].append(result["confusion_matrices"][model])

        # Params (avec .get par sécurité)
        results_by_noise["lambda_pen_spoq"][noise_idx].append(result["params"]["trust_region"].get("lambda_pen"))
        results_by_noise["lambda_pen_lasso"][noise_idx].append(result["params"]["fista"].get("lambda_pen"))
        results_by_noise["lambda_pen_scad"][noise_idx].append(result["params"]["scad"].get("lambda_pen"))
        results_by_noise["lambda_pen_reweighted"][noise_idx].append(result["params"]["reweighted"].get("lambda_pen"))
        results_by_noise["lambda_pen_mcp"][noise_idx].append(result["params"]["mcp"].get("lambda_pen"))
        results_by_noise["lambda_pen_irls"][noise_idx].append(result["params"]["irls"].get("lambda_pen"))

    # Save
    save_results(results_by_noise, output_dir="results", file_prefix="noise_impact_advanced_parallel")
    print("Experiment completed. Results saved.")


if __name__ == "__main__":
    
    base_kwargs = {
        "n_samples": 100,
        "n_features": 50,
        "n_informative": 10,
        "bias": 10,
        "coef": True,   
        "random_state": 42,
        "effective_rank": None,
        "tail_strength": 0.5,
        "noise_design": "median" # or "std"
    }

    noise_values = np.linspace(0, 0.5, 21).tolist()
    # noise_values = np.linspace(0, 0.5, 21).tolist()

    # n_jobs=-1 pour utiliser tous les cœurs
    evaluate_noise_impact_advanced_parallel(noise_values=noise_values, n_runs=50, n_jobs=-1, **base_kwargs)