import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold 
import time
from tqdm   import tqdm
import itertools
import optuna


from utils.functions import *
from utils.metrics import *

tolerance_absoulte_sparsity = 1e-6
relative_sparsity_ratio = 1e-3

def load_data(filename, data_dir="data", file_type=None, target_name=None, column_names=None, sep=','):
    """
    Loads data from a file in various formats (CSV, libsvm).

    Arguments:
        filename (str): name of the file to load (e.g. "data.txt")
        data_dir (str): directory where the data files are located
        file_type (str): 'csv', 'svmlight', or None to auto-detect
        target_column (str|int): name or index of the target column (for CSV)
        column_names (list): list of column names to use (for CSV)
        sep (str): separator for CSV files (default is auto-detected)

    Returns:
        (X, y): numpy arrays for features and target
    """
    
    data_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    # Auto-detect file type if not specified
    if file_type is None:
        if filename.endswith(".csv") or filename.endswith(".txt"):
            if filename in ["cpusmall_scale.txt", "YearPredictionMSD.txt"]:
                file_type = "svmlight"
            else:
                file_type = "csv"
        else:
            raise ValueError("Could not determine file type. Please specify 'file_type' explicitly.")

    # CSV or similar text file loading
    if file_type == "csv":
        df = pd.read_csv(data_path, header=None if column_names else "infer", names=column_names, sep=sep or r'\s+')

        if filename == 'bodyfat.csv':
            target_name = "BodyFat"

        if target_name is None:
            try:
                y = df['target']
                target_name = "target"
            except:
                raise ValueError("For CSV files, you must specify the target column (target_name).")

        y = df[target_name]
        X = df.drop(columns=[target_name])
        return X.to_numpy(), y.to_numpy()

    # LibSVM / SVMLight format loading
    elif file_type == "svmlight":
        X, y = load_svmlight_file(data_path)
        X, y = X.toarray(), np.asarray(y)
        return X, y

    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def check_and_add_bias_column(X, tol=1e-3, verbose = True):
    """
    Checks if there is already a bias column (i.e., a column close to all ones).
    If not, adds one as the first column.
    
    Parameters:
        X (np.ndarray): Input feature matrix.
        tol (float): Tolerance for checking if a column is constant and ≈ 1.
    
    Returns:
        np.ndarray: Modified feature matrix with bias column if needed.
    """
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.all(np.abs(col - 1) < tol):
            if verbose : 
                print(f"Bias column detected at index {i}")
            return X  # Bias column already 
    if verbose:
        print("No bias column found. Adding one as the first column.")
    bias = np.ones((X.shape[0], 1))
    return np.hstack([bias, X])  # Add bias column at the front


def standardize_except_bias(X):
    """
    Normalizes each column in X except the bias column (all ≈ 1).
    
    Parameters:
        X (np.ndarray): Input feature matrix.
    
    Returns:
        np.ndarray: Standardized feature matrix, excluding bias column.
    """
    X = X.copy()
    for i in range(X.shape[1]):
        col = X[:, i]
        if np.allclose(col, 1, atol=1e-3):
            continue  # Skip standardization for the bias column
        mean = np.mean(col)
        std = np.std(col)
        if std > 0:
            X[:, i] = (col - mean) / std
        else:
            X[:, i] = 0  # If the column is constant, set it to zero
    return X


def check_rank(X, verbose = True):
    if np.linalg.matrix_rank(X) == X.shape[1]:
        if verbose:
            print("The data matrix X is full rank")
    else : 
        print("The data matrix X is not full rank")
    
    return X


def load_and_preprocess(file, data_dir="data", file_type=None, target_name=None, column_names=None, sep=',', verbose = True):

    X,y = load_data(file,data_dir="data", file_type=file_type, target_name=target_name, column_names=column_names, sep=sep)
    # X,y = X.to_numpy(),y.to_numpy()
    X = check_rank(X, verbose=verbose)
    X = check_and_add_bias_column(X,verbose=verbose)
    X = standardize_except_bias(X)

    return X, y


# === MAIN ALGORITHM OF THE PAPER SPOQ_reg ===
def mm_algorithm_spoqreg(w_0, B, theta, epsilon, lambda_pen, max_iter, X_train, y_train, X_val, y_val, verbose=True):
    """
    Trust-region-MM-based algorithm with adaptive radius update.

    This algorithm solves a minimization problem using Majoration Minimization principle (MM), 
    with appropriate trust-region strategies.

    Can be interpreted as a constrained preconditioned gradient descent where the descent direction 
    is preconditioned by a matrix A(w), and constrained by a q-norm trust region. 

    Parameters:
        w_0        : Initial point in R^D (numpy array)
        B          : Maximum number of inner iterations to shrink the trust region
        theta      : Shrinking factor for the trust region radius (0 < theta < 1)
        epsilon    : Tolerance for the stopping criterion (based on gradient norm)
        max_iter   : Maximum number of outer iterations

    Returns:
        w                : Final solution (weight vector)
        loss             : List of loss values at each iteration
        sparsities       : List of sparsity percentages (based on threshold)
        gradients_norms  : List of gradient norms at each iteration
        n_iter           : Number of iterations performed
    """
    start_time = time.perf_counter()
    w = w_0
    train_loss = []
    mse_val = []
    mse_train = []
    gradients_norms = []
    relative_sparsities = []
    absolute_sparsities = []
    n_iter = 0
    weights = [w]
    
    X_t_X = X_train.T @ X_train
    # spec_norm = np.linalg.norm(X_t_X, ord=2)

    for k in range(max_iter):

        # we compute the loss
        current_train_loss = F(w,X_train,y_train,lambda_pen)
        y_pred_train = X_train @ w
        current_mse_train = mean_squared_error(y_pred_train,y_train)
        mse_train.append(current_mse_train)
        train_loss.append(current_train_loss)
        
        if verbose:
            print(f"Iteration {k}, Loss = {current_train_loss}") if k%1 == 0 else None

        # compute the validation loss
        y_pred = X_val @ w
        current_mse_val = mean_squared_error(y_pred,y_val)
        mse_val.append(current_mse_val)

        # we compute the gradient of the function
        grad = grad_F(w,X_train,y_train,lambda_pen)
        gradients_norms.append(np.linalg.norm(grad,2))

        # check stopping criterion on gradient norm
        if len(train_loss) > 2 and abs(current_train_loss - train_loss[-2]) <= epsilon:   # condition à faire évoluer (diviser par une valeur de la fonction max?)
            if verbose:
                print(f"Stopping criteria on function update reached at iter {k}/{max_iter}")
            break

        # we compute the radius for the trust region
        rho = rho_0(w)

        # trust region loop
        for i in range(1, B+1):
            
            if i >= B:
                rho = 0
            A = lambda_pen * A_tr(w,rho) +  X_t_X

            z = w - np.linalg.solve(A, grad)
            
            # we check if z belongs to the lq-ball complement
            if is_in_l_q_ball(z,rho):
                rho = theta * rho
            else : 
                break

        # when z belongs to the lq-ball complement we update the weights
        w = z
        
        weights.append(w)

        # === RELATIVE SPARSITY ===
        # we consider a weight negligable if it's less than 0.1% of the maximal weight
        max_weight = np.max(abs(w[1:]))
        current_relative_sparsity = 100 * np.sum((abs(w[1:]) <= relative_sparsity_ratio * max_weight)) / np.size(w[1:])
        relative_sparsities.append(current_relative_sparsity)

        # === ABSOLUTE SPARSITY ===
        current_absolute_sparsity = 100 * np.sum((abs(w[1:]) <= tolerance_absoulte_sparsity )) / np.size(w[1:])
        absolute_sparsities.append(current_absolute_sparsity)


    end_time = time.perf_counter()
    execution_time = end_time - start_time    

    if verbose:
        print(f"Temps d'exécution : {execution_time:.6f} secondes")
        if k == max_iter:
            print(f"Maximum number of iteration reached {k}")
        print(f"\nFinal Loss : {train_loss[-1]}")

    return w, train_loss, mse_val, mse_train, absolute_sparsities, relative_sparsities, gradients_norms, k, weights



# This is the VMFB algorithm of Cherni et Al. and we can show it is equivalent to the MM algorithm above
def trust_region_descent_implicit(w_0, B, theta, epsilon, lambda_pen, max_iter, X_train, y_train, X_val, y_val, verbose=True):
    """
    Trust-region-based gradient descent algorithm with adaptive radius update.

    This algorithm solves a minimization problem using a variant of gradient descent
    where the descent direction is preconditioned by a matrix A(w), and constrained 
    by a q-norm trust region. 

    Parameters:
        w_0        : Initial point in R^D (numpy array)
        B          : Maximum number of inner iterations to shrink the trust region
        theta      : Shrinking factor for the trust region radius (0 < theta < 1)
        epsilon    : Tolerance for the stopping criterion (based on gradient norm)
        max_iter   : Maximum number of outer iterations

    Returns:
        w                : Final solution (weight vector)
        loss             : List of loss values at each iteration
        sparsities       : List of sparsity percentages (based on threshold)
        gradients_norms  : List of gradient norms at each iteration
        n_iter           : Number of iterations performed
    """
    start_time = time.perf_counter()
    w = w_0
    train_loss = []
    mse_val = []
    mse_train = []
    gradients_norms = []
    relative_sparsities = []
    absolute_sparsities = []
    n_iter = 0
    weights=[w]

    X_t_X = X_train.T @ X_train
    L = np.linalg.norm(X_t_X, ord=2)

    for k in range(max_iter):

        # we compute the loss
        current_train_loss = F(w,X_train,y_train,lambda_pen)
        y_pred_train = X_train @ w
        y_pred_train = y_pred_train.reshape(-1)
        current_mse_train = mean_squared_error(y_pred_train,y_train)
        mse_train.append(current_mse_train)
        train_loss.append(current_train_loss)
        
        if verbose:
            print(f"Iteration {k}, Loss = {current_train_loss}") if k%1 == 0 else None

        # compute the validation loss
        y_pred = X_val @ w
        y_pred = y_pred.reshape(-1)
        current_mse_val = mean_squared_error(y_pred,y_val)
        mse_val.append(current_mse_val)

        # we compute the gradient of the penalty function
        grad = lambda_pen * grad_Psi(w)
        gradients_norms.append(np.linalg.norm(grad,2))

        # check stopping criterion on loss update 
        if len(train_loss) > 2 and abs(current_train_loss - train_loss[-2]) <= epsilon:   # condition à faire évoluer (diviser par une valeur de la fonction max?)
            if verbose:
                print(f"Stopping criteria on function update reached at iter {k}/{max_iter}")
            break

        # we compute the radius for the trust region
        rho = rho_0(w)
                             
        # trust region loop
        for i in range(1, B+1):

            
            if i >= B:
                rho = 0
            A = lambda_pen * A_tr(w,rho) 

            z = np.linalg.solve(A + X_t_X, A @ w - grad + X_train.T @ y_train)
            
            # we check if z belongs to the lq-ball complement
            if is_in_l_q_ball(z,rho):
                rho = theta * rho
            else : 
                break

        # when z belongs to the lq-ball complement we update the weights
        w = z
        
        weights.append(w)

        # === RELATIVE SPARSITY ===
        # we consider a weight negligable if it's less than 0.1% of the maximal weight
        max_weight = np.max(abs(w[1:]))
        current_relative_sparsity = 100 * np.sum((abs(w[1:]) <= relative_sparsity_ratio * max_weight)) / np.size(w[1:])
        relative_sparsities.append(current_relative_sparsity)

        # === ABSOLUTE SPARSITY ===
        current_absolute_sparsity = 100 * np.sum((abs(w[1:]) <= tolerance_absoulte_sparsity )) / np.size(w[1:])
        absolute_sparsities.append(current_absolute_sparsity)

    

    end_time = time.perf_counter()
    execution_time = end_time - start_time    

    if verbose:
        print(f"Temps d'exécution : {execution_time:.6f} secondes")
        if k == max_iter:
            print(f"Maximum number of iteration reached {k}")
        print(f"\nFinal Loss : {train_loss[-1]}")

    return w, train_loss, mse_val, mse_train, absolute_sparsities, relative_sparsities, gradients_norms, k, weights


def fista_lasso(w_0, epsilon, lambda_pen, max_iter, X_train, y_train, X_val, y_val, verbose=True):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for solving the Lasso problem:
        min_β (1/(2n)) * ||y - Xβ||²₂ + λ * ||β||₁

    Parameters:
        X         : (n, p) numpy array — design matrix
        y         : (n,) numpy array — response vector
        lambd     : regularization parameter λ (controls sparsity)
        epsilon   : stopping criterion for gradient norm
        max_iter  : maximum number of iterations

    Returns:
        beta         : final estimated coefficients after max_iter or early stop
        loss         : list of loss values per iteration
        sparsities   : list of sparsity percentages per iteration
        gradients_norms : list of ℓ² norms of the gradient
        n_iter       : number of iterations performed
    """
    # Precompute Lipschitz constant: L = σ_max(XᵀX)
    L = np.linalg.norm(X_train.T @ X_train, ord=2)  

    start_time = time.perf_counter()
    w = w_0              
    z = w              
    t = 1.0                          
    max_weight = np.max(abs(w[1:]))

    # History
    train_loss = []
    mse_val = []
    mse_train = []
    relative_sparsities = [100 * np.sum((abs(w[1:]) <= relative_sparsity_ratio * max_weight)) / np.size(w[1:])]
    absolute_sparsities = [100 * np.sum((abs(w[1:]) <= tolerance_absoulte_sparsity * max_weight)) / np.size(w[1:])]
    gradients_norms = []
    weights = [w]

    for k in range(1, max_iter + 1):

        # Compute loss
        current_train_loss = LASSO(w,X_train,y_train,lambda_pen)
        train_loss.append(current_train_loss)

        # Compute Train MSE
        y_pred_train = X_train @ w
        y_pred_train = y_pred_train.reshape(-1)
        current_mse_train = mean_squared_error(y_pred_train,y_train)
        mse_train.append(current_mse_train)
        if verbose:
            print(f"Iteration {k}, Loss = {current_train_loss}") if k%10 == 0 else None

        # compute the validation loss
        y_pred = X_val @ w
        current_mse_val = mean_squared_error(y_pred,y_val)
        mse_val.append(current_mse_val)
        

        # Keeping gradients of g
        grad = grad_Phi(z,X_train,y_train)
        grad_w = grad_Phi(w,X_train,y_train)
        gradients_norms.append(np.linalg.norm(grad_w, 2))


        # FISTA update
        w_next = soft_thresholding(z - (1/L) * grad, lambda_pen / L)

        # Checking stopping criterion on function update
        if abs(LASSO(w,X_train,y_train,lambda_pen) - LASSO(w_next,X_train,y_train,lambda_pen)) < epsilon:
            if verbose : 
                print(f"Stopping criteria on function update reached at iter {k}/{max_iter}")
            break

        # Momentum update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        z = w_next + ((t - 1) / t_next) * (w_next - w)
        
        
        # Update variables
        w = w_next
        t = t_next

        weights.append(w)

        # === RELATIVE SPARSITY ===
        # we consider a weight negligable if it's less than 0.1% of the maximal weight
        max_weight = np.max(abs(w[1:]))
        current_relative_sparsity = 100 * np.sum((abs(w[1:]) <= relative_sparsity_ratio * max_weight)) / np.size(w[1:])
        relative_sparsities.append(current_relative_sparsity)

        # === ABSOLUTE SPARSITY ===
        current_absolute_sparsity = 100 * np.sum((abs(w[1:]) <= tolerance_absoulte_sparsity )) / np.size(w[1:])
        absolute_sparsities.append(current_absolute_sparsity)

    end_time = time.perf_counter()
    execution_time = end_time - start_time 

    if verbose:   
        print(f"Temps d'exécution : {execution_time:.6f} secondes")
        print(f"\nMaximum number of iterations reached {k}" if k==max_iter-1 else "")
        print(f"Final Loss : {train_loss[-1]}")

    return w, train_loss, mse_val, mse_train, absolute_sparsities, relative_sparsities, gradients_norms, k, weights


def fista_scad(w_0, epsilon, lambda_pen, max_iter, X_train, y_train, X_val, y_val, verbose=True):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for solving the SCAD problem:
        min_β (1/(2n)) * ||y - Xβ||²₂ + λ * SCAD(β)

    Parameters:
        X         : (n, p) numpy array — design matrix
        y         : (n,) numpy array — response vector
        lambda_pen: regularization parameter λ (controls sparsity)
        epsilon   : stopping criterion for gradient norm
        max_iter  : maximum number of iterations

    Returns:
        beta         : final estimated coefficients after max_iter or early stop
        loss         : list of loss values per iteration
        sparsities   : list of sparsity percentages per iteration
        gradients_norms : list of ℓ² norms of the gradient
        n_iter       : number of iterations performed
    """
    # Precompute Lipschitz constant: L = σ_max(XᵀX)
    L = np.linalg.norm(X_train.T @ X_train, ord=2)  

    start_time = time.perf_counter()
    w = w_0              
    z = w              
    t = 1.0      
    max_weight = np.max(abs(w[1:]))
                    

    # History
    train_loss = []
    mse_val = []
    mse_train = []
    relative_sparsities = [100 * np.sum((abs(w[1:]) <= relative_sparsity_ratio * max_weight)) / np.size(w[1:])]
    absolute_sparsities = [100 * np.sum((abs(w[1:]) <= tolerance_absoulte_sparsity * max_weight)) / np.size(w[1:])]
    gradients_norms = []
    weights = [w]

    for k in range(1, max_iter + 1):

        # Compute loss
        current_train_loss = scad_loss(w,X_train,y_train,lambda_pen)
        train_loss.append(current_train_loss)

        # Compute Train MSE
        y_pred_train = X_train @ w
        y_pred_train = y_pred_train.reshape(-1)
        current_mse_train = mean_squared_error(y_pred_train,y_train)
        mse_train.append(current_mse_train)
        if verbose:
            print(f"Iteration {k}, Loss = {current_train_loss}") if k%10 == 0 else None

        # compute the validation loss
        y_pred = X_val @ w
        current_mse_val = mean_squared_error(y_pred,y_val)
        mse_val.append(current_mse_val)
        

        # Keeping gradients of g
        grad = grad_Phi(z,X_train,y_train)
        grad_w = grad_Phi(w,X_train,y_train)
        gradients_norms.append(np.linalg.norm(grad_w, 2))


        # FISTA update
        w_next = scad_prox(z - (1/L) * grad, lambda_pen / L, a=3.7)

        # Checking stopping criterion on function update
        if abs(scad_loss(w,X_train,y_train,lambda_pen) - scad_loss(w_next,X_train,y_train,lambda_pen)) < epsilon:
            if verbose : 
                print(f"Stopping criteria on function update reached at iter {k}/{max_iter}")
            break

        # Momentum update
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        z = w_next + ((t - 1) / t_next) * (w_next - w)
        
        
        # Update variables
        w = w_next
        t = t_next

        weights.append(w)

        # === RELATIVE SPARSITY ===
        # we consider a weight negligable if it's less than 0.1% of the maximal weight
        max_weight = np.max(abs(w[1:]))
        current_relative_sparsity = 100 * np.sum((abs(w[1:]) <= relative_sparsity_ratio * max_weight)) / np.size(w[1:])
        relative_sparsities.append(current_relative_sparsity)

        # === ABSOLUTE SPARSITY ===
        current_absolute_sparsity = 100 * np.sum((abs(w[1:]) <= tolerance_absoulte_sparsity )) / np.size(w[1:])
        absolute_sparsities.append(current_absolute_sparsity)

    end_time = time.perf_counter()
    execution_time = end_time - start_time 

    if verbose:   
        print(f"Temps d'exécution : {execution_time:.6f} secondes")
        print(f"\nMaximum number of iterations reached {k}" if k==max_iter-1 else "")
        print(f"Final Loss : {train_loss[-1]}")

    return w, train_loss, mse_val, mse_train, absolute_sparsities, relative_sparsities, gradients_norms, k, weights


def MCO(X,y):
    X_t = np.transpose(X)
    w_mco = np.linalg.solve(X_t @ X,  X_t @ y)

    # === RELATIVE SPARSITY ===
    max_weight = np.max(abs(w_mco[1:]))
    relative_sparsity = 100 * np.sum((abs(w_mco[1:]) <= relative_sparsity_ratio * max_weight)) / np.size(w_mco[1:])

    # === ABSOLUTE SPARSITY ===
    absolute_sparsity = 100 * np.sum((abs(w_mco[1:]) <= tolerance_absoulte_sparsity )) / np.size(w_mco[1:])
    
    return w_mco, absolute_sparsity, relative_sparsity


def tune_model(model_fn, param_grid, X, y, n_splits=5, scoring='mse', verbose=False):
    """
    Hyperparameter tuning with customizable scoring (MSE, AIC, BIC).
    """
    assert scoring in {'mse', 'aic', 'bic'}, "scoring must be 'mse', 'aic', or 'bic'"

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = float('inf')
    best_params = None
    best_output = None
    all_results = []
    outputs = []

    print(f"\nStarting tuning for model: {model_fn.__name__} using {scoring.upper()}")

    for i, params in enumerate(tqdm(combinations, desc=f"Grid Search ({model_fn.__name__})")):
        if verbose:
            print(f"\nCombo {i+1}/{len(combinations)}: {params}")
        
        fold_scores = []
        last_output = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            if verbose:
                print(f"  Fold {fold+1}/{n_splits}")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            try:
                output = model_fn(
                    **params,
                    X_train=X_train_fold,
                    y_train=y_train_fold,
                    X_val=X_val_fold,
                    y_val=y_val_fold,
                    verbose=False
                )

                w = output[0]
                y_pred = X_val_fold @ w

                # ICI ON NE CONSIDERE POUR L'AIC QUE LES PARAMETRES NON "NEGLIGEABLES"
                max_weight = np.max(w[1:])
                k =  np.sum((abs(w[1:]) >= relative_sparsity_ratio * max_weight))

                # OTHER OPTIONS :
                # k =  np.sum((abs(w[1:]) >=  0.001 * max_weight))
                # k = np.sum((abs(w[1:]) <= tolerance_absoulte_sparsity ))


                if scoring == 'mse':
                    score = mean_squared_error(y_val_fold, y_pred)
                elif scoring == 'aic':
                    score = compute_aic(y_val_fold, y_pred, k)
                elif scoring == 'bic':
                    score = compute_bic(y_val_fold, y_pred, k)

                fold_scores.append(score)
                last_output = output

            except Exception as e:
                print(f"Error in fold {fold+1}: {e}")
                fold_scores.append(np.inf)

        avg_score = np.mean(fold_scores)
        all_results.append((params, avg_score))
        outputs.append(w)

        if verbose:
            print(f"Avg {scoring.upper()} = {avg_score:.4f}")

        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            best_output = last_output

    if "w_0" in best_params:
       del best_params["w_0"]
       
    print(f"\nBest config for model: {model_fn.__name__}")
    print(f"Params: {best_params}")
    print(f"Avg CV {scoring.upper()}: {best_score:.4f}")

    return best_params, best_score, all_results, best_output, outputs


def tune_model_optuna(model_fn, lambda_bounds, X, y, fixed_params=None, n_splits=5, scoring='aic',
                       n_trials=200, verbose=True):
    """
    Tuning lambda_pen using Optuna, compatible with run_results() format.
    """

    optuna.logging.set_verbosity(optuna.logging.ERROR)


    if fixed_params is None:
        fixed_params = {}

    assert scoring in {'mse', 'aic', 'bic'}, "scoring must be 'mse', 'aic', or 'bic'"

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results = []
    weights_list = []

    if verbose:
       print(f"\nStarting Optuna tuning for model: {model_fn.__name__} using {scoring.upper()}")

    def eval_lambda(lambda_pen):
        fold_scores = []
        last_output = None

        for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(X)),desc="Fold", leave=False):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            try:
                output = model_fn(
                    lambda_pen=lambda_pen,
                    X_train=X_train_fold,
                    y_train=y_train_fold,
                    X_val=X_val_fold,
                    y_val=y_val_fold,
                    verbose=False,
                    **fixed_params
                )

                w = output[0]
                y_pred = X_val_fold @ w

                if len(w) > 1:
                    max_weight = np.max(np.abs(w[1:]))
                    k = np.sum(np.abs(w[1:]) >= relative_sparsity_ratio * max_weight) if max_weight > 0 else 0
                    # k = np.sum(np.abs(w[1:]) >= tolerance_absoulte_sparsity) 
                    # k = np.sum(w[1:] != 0)


                else:
                    k = 0

                if scoring == 'mse':
                    score = mean_squared_error(y_val_fold, y_pred)
                elif scoring == 'aic':
                    score = compute_aic(y_val_fold, y_pred, k)
                elif scoring == 'bic':
                    score = compute_bic(y_val_fold, y_pred, k)

                fold_scores.append(score)
                last_output = output

            except Exception as e:
                if verbose:
                    print(f"Error in fold {fold+1} with lambda_pen={lambda_pen:.4f}: {e}")
                fold_scores.append(np.inf)

        avg_score = np.mean(fold_scores)
        all_results.append(({'lambda_pen': lambda_pen}, avg_score))
        weights_list.append(last_output[0] if last_output else None)

        if verbose:
            print(f"lambda_pen={lambda_pen:.6f}, Avg {scoring.upper()}={avg_score:.4f}")
        return avg_score

    def objective(trial):
        if lambda_bounds[0]==0:
            lambda_pen = trial.suggest_float('lambda_pen', lambda_bounds[0], lambda_bounds[1])
        else:
            lambda_pen = trial.suggest_float('lambda_pen', lambda_bounds[0], lambda_bounds[1], log=True)
        return eval_lambda(lambda_pen)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

    study.optimize(objective, n_trials=n_trials)

    best_lambda = study.best_params['lambda_pen']
    best_score = study.best_value

    final_output = model_fn(
        lambda_pen=best_lambda,
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        verbose=False,
        **fixed_params
    )

    best_weights = final_output[0]

    best_params = dict(fixed_params)
    best_params['lambda_pen'] = best_lambda
    if "w_0" in best_params:
        del best_params["w_0"]

    print(f"\nBest config for algorithm: {model_fn.__name__}")
    # print(f"Params: {best_params}") # to see all params 
    print(f"Best lambda: {best_params['lambda_pen']:.2f}")
    print(f"Avg CV {scoring.upper()}: {best_score:.4f}")

    return best_params, best_score, all_results, final_output, weights_list