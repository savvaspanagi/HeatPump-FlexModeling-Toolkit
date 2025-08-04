import pandas as pd
import numpy as np
from .q_models import *
from .q_model_simulate import *

def train_greybox_q_model(train_df, model_type, bounds, deltaT, solver_name="ipopt", num_trials=20, show_plot=False, log=True):
    
    """
    General training framework. This training take only training df. No any validation process performed. 
    
    Args:
        train_df: pandas DataFrame with training data.
        model_type: str, e.g., '1R1C', '2R2C_A', '3R2C', etc.
        bounds: dict of parameter bounds.
        solver_name: optimization solver to use.
        num_trials: number of random initializations.
        deltaT: timestep in seconds.
        show_plot: bool, whether to plot each trial.

    Returns:
        best_model, best_parameters, best_initialization, all_trials DataFrame
    """

    ### Give Some initial values 
    best_rmse = np.inf
    best_params = None
    best_initialization = None
    best_model = None
    best_trial_index = -1
    all_trials = pd.DataFrame()
    solve_time = 0 
    param_names = list(bounds.keys())

    ### The training performed several times depend on the number of trials that are given
    for i in range(num_trials):
        init_vals = {param: np.random.uniform(*bounds[param]) for param in param_names} # Random initialization inside bounds
        try:
            # Train model with this initialization
            model, parameters = build_greybox_q_model(
                train_df=train_df,
                model_type=model_type,
                deltaT=deltaT,
                solver_name=solver_name,
                Tee=False,
                initialization=init_vals,
                bounds=bounds,
                show_plot=show_plot
            )
            rmse = sqrt(parameters.get('Objective', np.inf)) # In the simulation process MSE objective function is used. Therefore RMSE is obtained from this calculation
            solve_time = solve_time + parameters['Solve_time'] # The total solving time 

            # In each iteration the result parameters (tuning variables), initial values, RMSE, and the number of trials is save. 
            trial_data = parameters.copy()
            for param in param_names:
                trial_data[param + "_init"] = init_vals[param]
            trial_data['RMSE'] = rmse
            trial_data['Trial'] = i + 1

            all_trials = pd.concat([all_trials, pd.DataFrame([trial_data])], ignore_index=True) # Merge this trial result with the rest trials in a single df.

            if log==True: # When log is enable print in each iteration the RMSE, initial values and tuning parameters.
                print(f"[Trial {i+1}] RMSE = {rmse:.4f}, Init: {init_vals}")
                print(f" parameters: {parameters}")

            # Update the best values if a better solution is been found 
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = parameters
                best_initialization = init_vals
                best_model = model
                best_trial_index = i + 1

        except Exception as e: # If an error occur then just print out an exception error and continue with the next trial.
            print(f"[Trial {i+1}] Failed: {e}")

    ## After the whole simulation finish, print the best trial number, RMSE, initialization, parameters and the total solve time
    print("\n=== Best result ===")
    print(f"Best Trial #: {best_trial_index}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Best initialization: {best_initialization}")
    print(f"Best parameters: {best_params}")
    print(f"Solve Time: {solve_time}")

    return best_model, best_params, best_initialization, all_trials


# def train_greybox_model_with_validation_process(train_df, val_df, model_type, bounds, deltaT, solver_name="ipopt", num_trials=10, show_plot=False, log=True):
    
#     """
#     General training framework for any grey-box model.
    
#     Args:
#         train_df: pandas DataFrame with training data.
#         model_type: str, e.g., '1R1C', '2R2C_A', '3R2C', etc.
#         bounds: dict of parameter bounds.
#         solver_name: optimization solver to use.
#         num_trials: number of random initializations.
#         deltaT: timestep in seconds.
#         show_plot: bool, whether to plot each trial.

#     Returns:
#         best_model, best_parameters, best_initialization, all_trials DataFrame
#     """

#     ### Give Some initial values 
#     best_rmse_training = np.inf
#     best_rmse_validation = np.inf
#     best_params_training = None
#     best_params_validation = None
#     best_initializatio_training = None
#     best_initialization_validation = None
#     best_model_validation = None
#     best_trial_index_training = -1
#     best_trial_index_validation = -1
#     all_trials = pd.DataFrame()
#     solve_time = 0 
#     param_names = list(bounds.keys())

#     ### The training performed several times depend on the number of trials that are given
#     for i in range(num_trials):
#         # Random initialization inside bounds
#         init_vals = {param: np.random.uniform(*bounds[param]) for param in param_names}

#         try:
#             # Train model with this initialization
#             model, parameters = build_greybox_model(
#                 train_df=train_df, model_type=model_type, deltaT=deltaT, solver_name=solver_name,
#                 Tee=False, initialization=init_vals, bounds=bounds, show_plot=show_plot
#             )
#             rmse_training = parameters.get('Objective', np.inf)
#             solve_time = solve_time + parameters['Solve_time']

#             ## Take the last state values to use them in validation simulation 
#             last_index = max(model.T_in.index_set())
#             if model_type == "1R1C":
#                 initialize_validation = {"Tin": val_df['INDOOR_TEMP'].iloc[0]}
#             if model_type == "2R2C_A" or model_type == "3R2C" :
#                 initialize_validation = {"Tin": val_df['INDOOR_TEMP'].iloc[0], "Te": model.T_e[last_index]()}
#             if model_type == "4R3C":
#                 initialize_validation = {"Tint": model.T_int[last_index](),"Tin": val_df['INDOOR_TEMP'].iloc[0], "Te": model.T_e[last_index]()}
            
#             ## Run a validation simulation
#             validation, val_pred = simulate_and_evaluate(
#                 val_df=val_df, param_dict=parameters, model_type=model_type,
#                 initializations=initialize_validation, show_plot=show_plot
#             )
            
#             # In each iteration the result parameters (tuning variables), initial values, RMSE, and the number of trials is save. 
#             trial_data = parameters.copy()
#             for param in param_names:
#                 trial_data[param + "_init"] = init_vals[param]
#             trial_data['RMSE_Training'] = rmse_training
#             trial_data['RMSE_Validation'] = validation['RMSE']
#             trial_data['Trial'] = i 

#             all_trials = pd.concat([all_trials, pd.DataFrame([trial_data])], ignore_index=True) # Merge this trial result with the rest trials in a single df.

#             ## Take the last state values of validation phase to use them in testing simulation after from user.
#             if model_type == "1R1C":
#                 initialize_testing = {"Tin": val_pred['T_in_estimate'].iloc[-1]}
#             elif model_type == "2R2C_A":
#                 initialize_testing = {"Te": val_pred['T_e_estimate'].iloc[-1], "Tin": val_pred['T_in_estimate'].iloc[-1]}
#             elif model_type == "3R2C":
#                 initialize_testing = {"Te": val_pred['T_e_estimate'].iloc[-1], "Tin": val_pred['T_in_estimate'].iloc[-1]}
#             elif model_type == "4R3C":
#                 initialize_testing = {"Tint": val_pred['T_int_estimate'].iloc[-1], "Te": val_pred['T_e_estimate'].iloc[-1], "Tin": val_pred['T_in_estimate'].iloc[-1]}

#             # Update the best values if a better solution is been found 
#             if validation['RMSE'] < best_rmse_validation:
#                 best_rmse_validation = validation['RMSE']
#                 trainig_rmse_of_best_validation = rmse_training
#                 best_params_validation = parameters
#                 best_initialization_validation = init_vals
#                 best_model_validation = model
#                 best_trial_index_validation = i
#                 initialize_test=initialize_testing
#                 validation_pred=val_pred
            
#             if rmse_training < best_rmse_training:
#                 best_rmse_training = rmse_training
#                 best_params_training = parameters
#                 best_initializatio_training = init_vals
#                 best_trial_index_training = i
            
#             if log==True: # When log is enable print in each iteration the RMSE, initial values and tuning parameters.
#                 print("\n === Trial Result")
#                 print(f"Trial index #: {i}")
#                 print(f"Trial RMSE Validation: {validation['RMSE']:.4f}")
#                 print(f"Trial RMSE Training: {rmse_training:.4f}")
#                 print(f"Trial initialization: {init_vals}")
#                 print(f"Trial parameters: {parameters}")
#                 print(f"Solve Time: {parameters['Solve_time']}")
        
#         # If an error occur then just print out an exception error and continue with the next trial.
#         except Exception as e: 
#             print(f"[Trial {i+1}] Failed: {e}")
    
#     ## After the whole simulation finish, print the best trial number, RMSE, initialization, parameters and solve time
#     print("\n=== Best result ===")
#     print(f"Best Trial Validation #: {best_trial_index_validation}")
#     print(f"Best Trial Training #: {best_trial_index_training}")
#     print(f"Best RMSE Validation: {best_rmse_validation:.4f}")
#     print(f"RMSE Training of Best Validation : {trainig_rmse_of_best_validation:.4f}")
#     print(f"Best RMSE Training: {best_rmse_training:.4f}")
#     print(f"Best initialization Validation: {best_initialization_validation}")
#     print(f"Best initialization Training: {best_initializatio_training}")
#     print(f"Best parameters Validation: {best_params_validation}")
#     print(f"Best parameters Training: {best_params_training}")
#     print(f"Solve Time: {solve_time}")

#     return best_model_validation, best_params_validation, best_initialization_validation, validation_pred, initialize_test, all_trials