from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

latex_map = {
    "R_in_a": r"$R_{\mathrm{in,a}}$",
    "R_in_e": r"$R_{\mathrm{in,e}}$",
    "R_e_a":  r"$R_{\mathrm{e,a}}$",
    "C_in":   r"$C_{\mathrm{in}}$",
    "C_e":    r"$C_{\mathrm{e}}$",
    "Ain":    r"$A_{\mathrm{in}}$",
    "Ae":     r"$A_{\mathrm{e}}$",
    "fh":     r"$f_h$",
    "fh_e":   r"$f_{h,\mathrm{e}}$",
    "fh_in":  r"$f_{h,\mathrm{in}}$",
    "fh_int": r"$f_{h,\mathrm{int}}$"
}

def plot_residual(t_true, t_pred):
    """
    Plots the residual (t_true - t_pred).
    Args:
        t_true (array-like): Ground truth values
        t_pred (array-like): Predicted values
    """
    t_true = np.array(t_true)
    t_pred = np.array(t_pred)
    
    residual = t_true - t_pred

    plt.figure(figsize=(10,5))
    plt.plot(residual, linestyle='-', color='blue')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Residual [K]')
    plt.title('Residuals (True - Predicted)')
    plt.grid(True)
    plt.show()


def plot_parameter_vs_rmse(all_trials, param_name, y_start=6, rmse_threshold=100):
    """
    Creates a plot showing initial vs final parameter values and corresponding RMSE,
    starting from y_start and ignoring extremely large RMSE values.

    Args:
        all_trials: DataFrame with columns [param_name, param_name_init, RMSE]
        param_name: str, name of the parameter to plot (e.g., 'R_in_a' or 'C_in')
        y_start: float, starting y-axis value for initializations
        rmse_threshold: float, maximum RMSE allowed to include a point
    """
    fig, ax = plt.subplots(figsize=(8,6))

    # Φιλτράρισμα: Κρατάμε μόνο όσα RMSE < threshold
    filtered_trials = all_trials[all_trials['Objective'] < rmse_threshold]
    
    for idx, row in filtered_trials.iterrows():
        x_init = row[f"{param_name}_init"]
        x_final = row[param_name]
        y_final = row["RMSE"]

        # Draw line from (x_init, y_start) to (x_final, y_final)
        ax.plot([x_init, x_final], [y_start, y_final], color="gray", linestyle='-', marker='o', markersize=4, markerfacecolor='blue', markeredgecolor='blue')

        # Red bullet at final point
        ax.plot(x_final, y_final, 'o', color='red')
    latex_label = latex_map.get(param_name, param_name)
    ax.set_xlabel(f"{latex_label} value", fontsize=12)
    # ax.set_xlabel(f"{param_name} value", fontsize=12)
    ax.set_ylabel("RMSE [K]", fontsize=12)
    ax.set_title(f"{latex_label} vs RMSE", fontsize=14)
    # ax.set_title(f"Parameter Evolution: {param_name} vs RMSE", fontsize=14)
    ax.grid(True)
    plt.show()


def plot_parameters_vs_rmse_multi(all_trials, param_list, y_start=6, rmse_threshold=100, n_cols=3):
    """
    Creates multiple subplots showing the initial and final values of parameters vs RMSE.
    
    Args:
        all_trials: DataFrame containing parameter initializations, final values, and RMSE or Objective.
        param_list: List of parameter names to plot.
        y_start: Starting y-value for the initialization points.
        rmse_threshold: Maximum RMSE to display (filters out bad fits).
        n_cols: Number of columns in the subplot grid.
    """
    n_params = len(param_list)
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division to find rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharey=True)
    axes = axes.flatten()

    # Find correct RMSE column
    if "RMSE" in all_trials.columns:
        rmse_col = "RMSE"
    elif "Objective" in all_trials.columns:
        rmse_col = "Objective"
    else:
        raise ValueError("Neither 'RMSE' nor 'Objective' column found in DataFrame!")

    filtered_trials = all_trials[all_trials[rmse_col] < rmse_threshold]

    for idx, param in enumerate(param_list):
        if idx >= len(axes):
            break
            
        x_init = filtered_trials[f"{param}_init"]
        x_final = filtered_trials[param]
        y_rmse = filtered_trials[rmse_col]
        
        for i in range(len(filtered_trials)):
            axes[idx].plot([x_init.iloc[i], x_final.iloc[i]], [y_start, y_rmse.iloc[i]], 
                           'o-', color='gray', markersize=4)
            axes[idx].plot(x_final.iloc[i], y_rmse.iloc[i], 'ro')  # Final points red
        label_latex = latex_map.get(param, param)
        axes[idx].set_xlabel(f"{label_latex} value", fontsize=16)
        axes[idx].set_title(f"{label_latex} vs RMSE", fontsize=16)
        axes[idx].set_ylabel(f"RMSE [oC]", fontsize=12)
        
        # axes[idx].set_xlabel(f"{param} value")
        # axes[idx].set_title(f"{param} vs RMSE")
        axes[idx].grid(True)
    # Hide unused axes
    for ax in axes[n_params:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt          # Plotting library
import numpy as np                       # Numerical routines (arrays, math)
from sklearn.metrics import mean_squared_error  # RMSE calculation


def plot_model_validation(
    original_set,        # list of DataFrames, one per case study
    validation_set,      # list of lists of models, parallel to original_set
    model_labels,              # subplot titles, parallel to original_set
    test_case_labels,              # subplot titles, parallel to original_set
    colors=None,         # optional custom colors for model curves
    figsize=(8, 11),     # overall figure size in inches (width, height)
    sharex=True          # whether the x-axis is shared among subplots
):
    """
    Plot measured indoor temperature against multiple grey-box model
    predictions for any number of case studies, printing RMSEs as a side effect.
    """

    # --- Choose default colours if none supplied ---------------------------
    if colors is None:
        # Matplotlib’s default colour cycle – truncate to number of models
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(model_labels)]

    # --- Create the figure and the subplot array ---------------------------
    fig, axes = plt.subplots(nrows=len(original_set),ncols=1,figsize=figsize,sharex=sharex)

    # Matplotlib gives a single Axes object when nrows == ncols == 1.
    # Wrap it in a list, then flatten in case we have a 2-D ndarray.
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    axes = np.ravel(axes)   # guarantees a 1-D iterable of Axes objects

    # --- Iterate through all case studies ----------------------------------
    # zip stops at the shortest input, so make sure lists are same length.
    for ax, df, models, label, tc_label in zip(axes, original_set, validation_set, model_labels, test_case_labels):
        N = len(df)                       # number of time steps in this case
        x = np.arange(N)                  # x-axis indices (0, 1, 2, …, N-1)
        # Extract true indoor temperature as a NumPy array for plotting.
        T_true = df["INDOOR_TEMP"].values[:N]

        # Plot the measured series: black, dashed, slightly transparent.
        ax.plot(x, T_true, linestyle='--', color='black', alpha=0.7, label='Measured')
        # --- Plot each model for this case study ---------------------------
        for T_model, color, name in zip(models, colors, model_labels):

            # Plot the model curve. 
            ax.plot(x,T_model[:N],label=name,color=color,alpha=0.8)

            # Compute and print RMSE between measured and modelled temps.
            rmse = np.sqrt(mean_squared_error(df['INDOOR_TEMP'], T_model[:N]))
            print(f"RMSE {name} {tc_label}: {rmse:.2f} °C")

        # --- Cosmetic subplot settings -------------------------------------
        ax.set_ylabel("Tin [°C]")         # y-axis label
        ax.set_title(tc_label, fontsize=10)  # subplot title
        ax.grid(True)                     # light grid
        ax.legend(fontsize=7)             # show legend

    # Common x-axis label added after the loop (last row of subplots).
    axes[-1].set_xlabel("Time Step")

    # Tight layout prevents label overlap.
    plt.tight_layout()

    # Render everything on screen.
    plt.show()


