import glob
import json

from run_results import run_results

def get_best_lambdas_from_logs(file, scoring_metric="aic", target_name=None, test_size=0.2, relative_sparsity=True):
    best_spoq = None
    best_fista = None
    best_score = float("inf")

    log_files = glob.glob("logs/*.json")

    for log_path in log_files:
        try:
            with open(log_path, "r") as f:
                data = json.load(f)

            # Ensure log matches the dataset, metric, and sparsity mode
            if (
                data["meta"]["file"] != file
                or data["meta"].get("scoring_metric") != scoring_metric
                or data["meta"].get("sparsity") != ("relative" if relative_sparsity else "absolute")
            ):
                continue

            # Get CV scores
            spoq_score = data.get("cv_scores", {}).get("spoq")
            fista_score = data.get("cv_scores", {}).get("lasso")

            if spoq_score is None or fista_score is None:
                continue

            # Select best based on SPOQ CV score
            if spoq_score < best_score:
                best_score = spoq_score
                best_spoq = data["params"]["trust_region"]["lambda_pen"]
                best_fista = data["params"]["fista"]["lambda_pen"]

        except Exception as e:
            print(f"Could not parse log file {log_path}: {e}")
            continue

    # If nothing found, ask user
    if best_spoq is None or best_fista is None:
        print(f"\nNo logs found for dataset '{file}' with scoring metric '{scoring_metric}' and sparsity mode {'relative' if relative_sparsity else 'absolute'}.")
        user_input = input("Do you want to run `run_results()` to generate the optimal lambdas? (y/n) ").strip().lower()

        if user_input == 'y':
            if target_name is None:
                raise ValueError("Missing `target_name`, cannot run `run_results()`.")
            print("Running `run_results()`...")
            run_results(
                file=file,
                target_name=target_name,
                test_size=test_size,
                scoring=scoring_metric,
                relative_sparsity=relative_sparsity,
                verbose=True
            )
            return get_best_lambdas_from_logs(
                file, 
                scoring_metric=scoring_metric, 
                target_name=target_name, 
                test_size=test_size, 
                relative_sparsity=relative_sparsity
            )
        else:
            print("Please enter lambda values manually:")
            try:
                best_spoq = float(input("Enter lambda for SPOQ: ").strip())
                best_fista = float(input("Enter lambda for LASSO (FISTA): ").strip())
            except ValueError:
                raise ValueError("Invalid input. Lambda values must be numbers.")

    return best_spoq, best_fista
