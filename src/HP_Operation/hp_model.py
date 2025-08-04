import numpy as np
import pandas as pd

class HeatPumpModel:
    def __init__(self, name, a0, a1, a2, gamma_corr=1.0):
        """
        Parameters:
        -----------
        name : str
            Identifier for the heat pump type.
        a0, a1, a2 : float
            COP regression coefficients.
        gamma_corr : float, optional
            Correction factor applied to COP (default is 1.0 = no correction).
        """
        self.name = name
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.gamma_corr = gamma_corr
        self.cop_table = None
        self.heat_distri = None

    def compute_cop(self, delta_T):
        """Compute COP for a given ΔT or array of ΔT."""
        delta_T = np.array(delta_T)
        cop = self.a0 + self.a1 * delta_T + self.a2 * delta_T**2
        return np.clip(cop, 1.0, None)

    def generate_cop_curve(self, delta_T_range):
        """
        Generate COP curve table over a range of ΔT values.
        
        Returns:
        --------
        pd.DataFrame with ΔT as index and COP as values.
        """
        delta_T = np.array(delta_T_range)
        cop_values = self.compute_cop(delta_T)
        df = pd.DataFrame({'COP': cop_values}, index=delta_T)
        df.index.name = "ΔT [°C]"
        self.cop_table = df

    def generate_sink_temp_curve(self, a0, a1, a2, T_source_range):
        """
        Generates a table of T_supply values for a given range of T_source,
        using the quadratic model:
        T_supply = a0 + a1 * T_source + a2 * T_source^2
        
        Parameters:
        -----------
        a0, a1, a2 : float
            Coefficients of the T_supply model.
        T_source_range : iterable
            Values of T_source [°C] to evaluate.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with T_source as index and T_supply as values.
        """
        T_source = np.array(T_source_range)
        T_supply = a0 + a1 * T_source + a2 * T_source**2
        df = pd.DataFrame({'T_supply': T_supply}, index=T_source)
        df.index.name = "T_source [°C]"
        self.heat_distri = df



