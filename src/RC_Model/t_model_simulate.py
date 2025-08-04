import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def simulate_and_evaluate(val_df, param_dict,initializations, model_type="1R1C", deltaT=1800, show_plot=True):
    N = len(val_df)
    T_out = val_df["OUTDOOR_TEMP"].values
    Phi_h = val_df["PHI_H"].values
    Q_irrad = val_df["PYRANOMETER"].values
    T_meas = val_df["INDOOR_TEMP"].values

    # Initialize states
    T_in_sim = np.zeros(N)
    T_e_sim = np.zeros(N) if model_type in ["2R2C_A", "3R2C", "4R3C"] else None
    T_int_sim = np.zeros(N) if model_type in ["3R2C", "4R3C"] else None


    T_in_sim[0] = initializations.get("Tin", T_meas[0])
    if T_e_sim is not None: T_e_sim[0] = initializations.get("Te", T_meas[0]-1)
    if T_int_sim is not None: T_int_sim[0] = initializations.get("Tint", T_meas[0])

    # Parameters extraction
    R_in_a = param_dict.get("R_in_a", 1)
    C_in = param_dict.get("C_in", 1)
    R_in_e = param_dict.get("R_in_e", 1)
    R_e_a = param_dict.get("R_e_a", 1)
    C_e = param_dict.get("C_e", 1)
    R_int_in = param_dict.get("R_int_in", 1)
    C_int = param_dict.get("C_int", 1)
    Ain = param_dict.get("Ain", 0)
    Ae = param_dict.get("Ae", 0)
    Aint = param_dict.get("Aint", 0)
    fh = param_dict.get("fh", 1)
    fh_in = param_dict.get("fh_in", 1)
    fh_e = param_dict.get("fh_e", 0)
    fh_int = param_dict.get("fh_int", 0)
    # Simulation Loop
    for t in range(N-1):
        if model_type == "1R1C":
            T_in_sim[t+1] = T_in_sim[t] + deltaT * (
                (T_out[t] - T_in_sim[t]) / (R_in_a * C_in) +
                fh * Phi_h[t] / C_in +
                Ain * Q_irrad[t] / C_in
            )

        elif model_type == "2R2C_A":
            T_in_sim[t+1] = T_in_sim[t] + deltaT * (
                (T_e_sim[t] - T_in_sim[t]) / (R_in_e * C_in) +
                fh * Phi_h[t] / C_in +
                Ain * Q_irrad[t] / C_in
            )
            T_e_sim[t+1] = T_e_sim[t] + deltaT * (
                (T_in_sim[t] - T_e_sim[t]) / (R_in_e * C_e) +
                (T_out[t] - T_e_sim[t]) / (R_e_a * C_e) +
                (1-fh) * Phi_h[t] / C_e +
                Ae * Q_irrad[t] / C_e
            )

        elif model_type == "3R2C":
            T_in_sim[t+1] = T_in_sim[t] + deltaT * (
                (T_e_sim[t] - T_in_sim[t]) / (R_in_e * C_in) +
                (T_out[t] - T_in_sim[t]) / (R_in_a * C_in) +
                Ain * Q_irrad[t] / C_in +
                fh * Phi_h[t] / C_in
            )
            T_e_sim[t+1] = T_e_sim[t] + deltaT * (
                (T_out[t] - T_e_sim[t]) / (R_e_a * C_e) +
                (T_in_sim[t] - T_e_sim[t]) / (R_in_e * C_e) +
                Ae * Q_irrad[t] / C_e +
                (1-fh) * Phi_h[t] / C_e
            )

        elif model_type == "4R3C":
             # T_int update
            T_int_sim[t + 1] = T_int_sim[t] + deltaT * (
                (T_in_sim[t] - T_int_sim[t]) / (R_int_in * C_int) +
                Aint*Q_irrad[t] / C_int + fh_int * Phi_h[t] / C_int
            )

            # T_in update
            T_in_sim[t + 1] = T_in_sim[t] + deltaT * (
                (T_e_sim[t] - T_in_sim[t]) / (R_in_e * C_in) +
                (T_out[t] - T_in_sim[t]) / (R_in_a * C_in) +
                (T_int_sim[t] - T_in_sim[t]) / (R_int_in * C_in) +
                Ain*Q_irrad[t] / C_in + fh_in * Phi_h[t] / C_in
            )

            # T_e update
            T_e_sim[t + 1] = T_e_sim[t] + deltaT * (
                (T_out[t] - T_e_sim[t]) / (R_e_a * C_e) +
                (T_in_sim[t] - T_e_sim[t]) / (R_in_e * C_e) +
                Ae * Q_irrad[t] / C_e + fh_e * Phi_h[t] / C_e
            )

    # Evaluation
    val_df_sim = val_df.copy()
    val_df_sim["T_in_estimate"] = T_in_sim
    if model_type == "2R2C_A":
        val_df_sim["T_e_estimate"] = T_e_sim
    elif model_type == "3R2C":
        val_df_sim["T_e_estimate"] = T_e_sim
    elif model_type == "4R3C":
        val_df_sim["T_e_estimate"] = T_e_sim
        val_df_sim["T_int_estimate"] = T_int_sim

    y_true = val_df_sim["INDOOR_TEMP"].values
    y_pred = val_df_sim["T_in_estimate"].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    sse = np.sum((y_true - y_pred)**2)

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(val_df_sim.index, y_true, linestyle='--', label='Measured')
        plt.plot(val_df_sim.index, y_pred, linestyle='-', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Indoor Temp [°C]')
        plt.title(f'Simulation: {model_type}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {"RMSE": rmse, "SSE": sse}, val_df_sim


import numpy as np
import matplotlib.pyplot as plt