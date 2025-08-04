import numpy as np
import pandas as pd

def convert_heat_demand_to_power_series(Q_heat_series, T_source_series, heat_pump_model, P_nom_ht, COP_booster=1.0):
    """
    Converts heat demand and T_source series to electric power using HeatPumpModel.
    
    Parameters:
    -----------
    Q_heat_series : list or np.array
        Series of heat demand values [kWh or kW].
    T_source_series : list or np.array
        Series of source temperatures [°C].
    heat_pump_model : HeatPumpModel
        Instance of HeatPumpModel with COP functions.
    P_nom_ht : float
        Nominal HP power capacity [kW].
    COP_booster : float
        COP of booster heater (default 1.0).
    
    Returns:
    --------
    pd.DataFrame with:
        - Q_heat
        - T_source
        - T_supply
        - COP
        - P_hp
        - P_booster
        - P_electric
    """
    Q_heat_series = np.array(Q_heat_series)
    T_source_series = np.array(T_source_series)
    # Compute T_supply and ΔT
    T_supply_series=heat_pump_model.heat_distri.loc[T_source_series, "T_supply"]
    delta_T_series = T_supply_series - T_source_series
    delta_T_series=np.round(delta_T_series, 1)  # Round to 2 decimal places for consistency
    # Compute COP and correct it

    COP_series = heat_pump_model.cop_table.loc[delta_T_series, "COP"] * heat_pump_model.gamma_corr
    # Compute P_hp
    P_hp_series = np.minimum(Q_heat_series / COP_series, P_nom_ht)

    # Compute Q delivered by HP and remaining deficit
    Q_hp_output = P_hp_series * COP_series
    Q_deficit = np.maximum(0, Q_heat_series - Q_hp_output)

    # Compute P_booster and total P_electric
    P_booster_series = Q_deficit / COP_booster if COP_booster > 0 else np.zeros_like(Q_deficit)
    P_electric_series = P_hp_series  + P_booster_series
    # Build DataFrame
    results_df = pd.DataFrame({
        'Q_heat': Q_heat_series,
        'T_source': T_source_series,
        'T_supply': np.array(T_supply_series),
        'COP': np.array(COP_series),
        'P_hp': np.array(P_hp_series),
        'P_booster': np.array(P_booster_series),
        'P_electric': np.array(P_electric_series)
    })

    return results_df
