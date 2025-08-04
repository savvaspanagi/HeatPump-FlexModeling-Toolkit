from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import Var  

# ---- Main function ----
def build_greybox_q_model(train_df, model_type, deltaT, solver_name="ipopt", Tee=False, initialization=None, bounds=None, show_plot=False):
    model = ConcreteModel()
    N = len(train_df)
    model.T = RangeSet(0, N-1)
    
    # Load data
    Q_h = np.array(train_df["PHI_H"])
    Q_irrad = np.array(train_df["PYRANOMETER"])
    T_in_meas = np.array(train_df["INDOOR_TEMP"])
    T_out_meas = np.array(train_df["OUTDOOR_TEMP"])

    # Default initializations and bounds per model_type
    param_defaults, param_bounds = get_q_model_parameters(model_type, initialization, bounds)

    # Define variables
    define_q_model_variables(model, model_type, param_defaults, param_bounds)

    # Define states
    define_q_model_states(model, model_type)
    model.Q_h[0].fix(Q_h[0])
    # Objective
    def objective_rule(m):
        return sum((m.Q_h[t] - Q_h[t]) ** 2 for t in m.T) / len(m.T)
    model.obj = Objective(rule=objective_rule, sense=minimize)

    # Dynamics
    define_q_model_dynamics(model, model_type, deltaT, Q_h, Q_irrad, T_out_meas, T_in_meas)

    # Solve
    opt = SolverFactory(solver_name)
    results = opt.solve(model, tee=Tee)
    solve_time = results['Solver'][0]['Time']
    # Extract parameters
    param_values = extract_q_model_parameters(model, model_type,solve_time)
    if show_plot:
        plot_q_model_temperature_comparison(model, train_df, Q_h)

    return model, param_values

# --- Supporting functions ---

def get_q_model_parameters(model_type, initialization=None, bounds=None):
    param_defaults = {}
    param_bounds = {}

    if model_type == "1R1C":
        param_defaults = {"R_in_a": 0.16, "C_in": 1.2e6, "Ain": 0.03, "fh": 1}
        param_bounds = {"R_in_a": (0.001, 1), "C_in": (1e6, 1e8), "Ain": (0, 120000), "fh": (0,1)}

    elif model_type == "2R2C_A":
        param_defaults = {"R_in_e": 0.1, "R_e_a": 0.1, "C_in": 12e7, "C_e": 12e6, "Ain": 20, "Ae": 10,  "fh": 1}
        param_bounds = {"R_in_e": (0.001, 1), "R_e_a": (0.001, 1), "C_in": (1e6, 1e8), "C_e": (1e6, 1e8), "Ain": (0, 120000), "Ae": (0, 120000),  "fh": (0,1)}

    elif model_type == "3R2C":
        param_defaults = {"R_in_e": 0.016, "R_e_a": 0.016, "R_in_a": 0.016, "C_in": 1.2e7, "C_e": 1.2e7, "Ain": 1, "Ae": 1,  "fh": 1}
        param_bounds = {"R_in_e": (0.001, 1), "R_e_a": (0.001, 1), "R_in_a": (0.001, 1), "C_in": (1e6, 1e8), "C_e": (1e6, 1e8), "Ain": (0, 120000), "Ae": (0, 120000),  "fh": (0,1)}

    elif model_type == "4R3C":
        param_defaults = {"R_int_in": 0.002, "R_in_e": 0.002, "R_e_a": 0.002, "R_in_a": 0.002,
                          "C_int": 1.2e6, "C_in": 1.2e6, "C_e": 1.2e6,
                          "Aint": 1, "Ain": 1, "Ae": 1, "fh_int":0, "fh_in":0, "fh_e":0}
        param_bounds = {"R_int_in": (0.001, 1), "R_in_e": (0.001, 1), "R_e_a": (0.001, 1), "R_in_a": (0.001, 1),
                        "C_int": (1e6, 1e8), "C_in": (1e6, 1e8), "C_e": (1e6, 1e8),
                        "Aint": (0, 120000), "Ain": (0, 120000), "Ae": (0, 120000), "fh_int": (0,1), "fh_in": (0,1), "fh_e": (0,1)}

    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # Override defaults if user provides initialization and bounds
    if initialization:
        param_defaults.update(initialization)
    if bounds:
        param_bounds.update(bounds)
    return param_defaults, param_bounds


def define_q_model_variables(model, model_type, param_defaults, param_bounds):
    for pname, init in param_defaults.items():
        model.add_component(pname, Var(bounds=param_bounds[pname], initialize=init))
    # if model_type in ["1R1C"]:
    model.fh.fix(1)

def define_q_model_states(model, model_type):
    # Define T_in with bounds [-30, 50]
    model.Q_h = Var(model.T, bounds=(0, 20000))

    if model_type in ["2R2C_A", "3R2C", "4R3C"]:
        # Define T_e with bounds [-30, 50]
        model.T_e = Var(model.T, bounds=(-30, 50))

    if model_type in ["3R2C", "4R3C"]:
        # Define T_int with bounds [-30, 50]
        model.T_int = Var(model.T, bounds=(-30, 50))


def define_q_model_dynamics(model, model_type, deltaT, Q_h, Q_irrad, T_out_meas, T_in_meas):
    N = len(Q_h)

    def dynamics_rule_Tin(m, t):
        if t == N-1:
            return Constraint.Skip
        
        expr = T_in_meas[t]

        if model_type == "1R1C":
            expr += deltaT * ((T_out_meas[t] - T_in_meas[t]) / (m.R_in_a * m.C_in) + m.fh * m.Q_h[t] / m.C_in + m.Ain * Q_irrad[t] / m.C_in)

        elif model_type == "2R2C_A":
            expr += deltaT * ((m.T_e[t] - T_in_meas[t]) / (m.R_in_e * m.C_in) + m.fh * m.Q_h[t] / m.C_in + m.Ain * Q_irrad[t] / m.C_in)

        elif model_type == "3R2C":
            expr += deltaT * ((m.T_e[t] - T_in_meas[t]) / (m.R_in_e * m.C_in) + (T_out_meas[t] - T_in_meas[t]) / (m.R_in_a * m.C_in)  + m.fh * m.Q_h[t] / m.C_in + m.Ain * Q_irrad[t] / m.C_in)

        elif model_type == "4R3C":
            expr += deltaT * ((m.T_e[t] - T_in_meas[t]) / (m.R_in_e * m.C_in) + (T_out_meas[t] - T_in_meas[t]) / (m.R_in_a * m.C_in) + (m.T_int[t] - T_in_meas[t]) / (m.R_int_in * m.C_in) + m.fh_in * m.Q_h[t] / m.C_in + m.Ain * Q_irrad[t] / m.C_in)

        return T_in_meas[t+1] == expr

    model.dynamics_Tin = Constraint(model.T, rule=dynamics_rule_Tin)

    def last_value_rule_Tin1(m):
        return m.Q_h[N-1]-m.Q_h[N-2] <= 100

    model.last_value_Tin1 = Constraint(rule=last_value_rule_Tin1)

    def last_value_rule_Tin2(m):
        return Q_h[N-1]-m.Q_h[N-1] <= 100

    model.last_value_Tin2 = Constraint(rule=last_value_rule_Tin2)
        
    # === Now define T_e dynamics (only for models with T_e)
    if model_type in ["2R2C_A", "3R2C", "4R3C"]:
        def dynamics_rule_Te(m, t):
            if t == N-1:
                return Constraint.Skip
            expr = m.T_e[t]
            
            if model_type in ["2R2C_A", "3R2C"]:
                expr += deltaT * (
                    (T_out_meas[t] - m.T_e[t]) / (m.R_e_a * m.C_e) +
                    (T_in_meas[t] - m.T_e[t]) / (m.R_in_e * m.C_e) +
                    m.Ae * Q_irrad[t] / m.C_e + (1-m.fh) * m.Q_h[t] / m.C_e
                )
            elif model_type in [ "4R3C"]:
                expr += deltaT * (
                    (T_out_meas[t] - m.T_e[t]) / (m.R_e_a * m.C_e) +
                    (T_in_meas[t] - m.T_e[t]) / (m.R_in_e * m.C_e) +
                    m.fh_e * m.Q_h[t] / m.C_e +
                    m.Ae * Q_irrad[t] / m.C_e
                )

            return m.T_e[t+1] == expr

        model.dynamics_Te = Constraint(model.T, rule=dynamics_rule_Te)

    # === Now define T_int dynamics if needed
    if model_type in ["4R3C"]:
        def dynamics_rule_Tint(m, t):
            if t == N-1:
                return Constraint.Skip

            expr = m.T_int[t]


            expr += deltaT * (
                (T_in_meas[t] - m.T_int[t]) / (m.R_int_in * m.C_int) +
                m.fh_int * m.Q_h[t] / m.C_int +
                m.Aint * Q_irrad[t] / m.C_int
                )

            return m.T_int[t+1] == expr

        model.dynamics_Tint = Constraint(model.T, rule=dynamics_rule_Tint)

    # === Special constraint for 4R2C / 4R3C: balance heating fractions
    if model_type in ["4R3C"]:
        def dynamics_rule_phi(m):
            return m.fh_in + m.fh_e + m.fh_int == 1

        model.dynamics_phi = Constraint(rule=dynamics_rule_phi)


def extract_q_model_parameters(model, model_type,solve_time):
    # Extract all parameter variables
    # This line creates a list of all variable names in the model except for the state variables 'T_in', 'T_e', and 'T_int'.
    # It iterates over all components of type Var in the model, and excludes those with names 'T_in', 'T_e', or 'T_int'.
    param_names = [vname for vname in model.component_map(Var) if vname not in ['T_in', 'T_e', 'T_int', 'Q_h']]
    param_values = {p: value(getattr(model, p)) for p in param_names}
    
    # Also extract the objective value
    param_values["Objective"] = value(model.obj)
    param_values['Solve_time'] = solve_time
    return param_values


def plot_q_model_temperature_comparison(model, train_df, Q_h):
    N = len(train_df)
    T_meas_plot = Q_h[:N]
    T_model = np.array([value(model.Q_h[t]) for t in range(N)])

    plt.figure(figsize=(10, 5))
    plt.plot(train_df.index[:N], T_meas_plot, label='Measured Indoor Temperature', linestyle='--')
    plt.plot(train_df.index[:N], T_model, label='Modeled Indoor Temperature', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Temperature [°C]')
    plt.title('Comparison of Measured and Modeled Indoor Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()