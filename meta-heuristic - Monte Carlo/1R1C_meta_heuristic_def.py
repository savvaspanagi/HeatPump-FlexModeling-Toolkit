# Required Libraries
import numpy as np
from sklearn.metrics import mean_squared_error
# PSO
from pyMetaheuristic.algorithm import particle_swarm_optimization
from pyMetaheuristic.utils import graphs

# Target Function: Easom Function
def easom(variables_values=[0, 0, 0], train_df=None):
    print(variables_values)
    try:
        R_in_a, C_in, Ain = variables_values

        # Check for values out of the bounds
        if R_in_a <= 0 or C_in <= 0:
            print(f"[Warning] Invalid values detected: R_in_a = {R_in_a}, C_in = {C_in}")
            return 1e6

        T_in_sim = np.zeros(len(train_df))
        deltaT = 1800
        fh = 1

        T_in_sim[0] = train_df["INDOOR_TEMP"].iloc[0]

        for t in range(len(train_df) - 1):
            T_out = train_df["OUTDOOR_TEMP"].iloc[t]
            Phi_h = train_df["PHI_H"].iloc[t]
            Q_irrad = train_df["PYRANOMETER"].iloc[t]

            update = deltaT * (
                (T_out - T_in_sim[t]) / (R_in_a * C_in) +
                fh * Phi_h / C_in +
                Ain * Q_irrad / C_in
            )

            if np.isnan(update) or np.isinf(update):
                print(f"[Warning] NaN or Inf in update detected: R_in_a = {R_in_a}, C_in = {C_in}")
                return 1e6

            T_in_sim[t + 1] = T_in_sim[t] + update
        
        # --- Evaluation ---
        y_true = train_df["INDOOR_TEMP"].values
        y_pred = T_in_sim

        mse = mean_squared_error(y_true, y_pred)
        if mse < 0:
            print(f"[Warning] Negative MSE detected: R_in_a = {R_in_a}, C_in = {C_in}")
            print(mse)
        rmse = np.sqrt(mse)

        # if np.isnan(rmse) or np.isinf(rmse):
        #     print(f"[Warning] NaN or Inf in RMSE detected: R_in_a = {R_in_a}, C_in = {C_in}")
        #     return 1e6

        return rmse

    except Exception as e:
        print(f"[Warning] Unexpected error: {e} at R_in_a = {R_in_a}, C_in = {C_in}")
        return 1e6
