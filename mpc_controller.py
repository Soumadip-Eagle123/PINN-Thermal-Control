import torch
import numpy as np
from scipy.optimize import minimize
from pinn_model import ThermalPINN

class NMPCController:
    """
    Neural Model Predictive Controller (NMPC).
    Uses the trained PINN model as an internal predictive engine over horizon Hp.
    """
    def __init__(self, model_path="models/trained_pinn.pth", Hp=10, dt=0.5):
        self.Hp = Hp  # Prediction horizon (number of look-ahead steps)
        self.dt = dt  # Time step (seconds)
        
        # Load pre-trained PyTorch PINN model
        self.pinn = ThermalPINN()
        try:
            self.pinn.load_state_dict(torch.load(model_path))
            self.pinn.eval()
            print("NMPC Controller: PINN model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load PINN weights from {model_path}. Using untrained weights.")

    def predict_future(self, T_start, u_sequence, T_amb=25.0):
        """Simulates future trajectory over horizon Hp using PINN predictions."""
        T_sim = T_start
        for u in u_sequence:
            x_input = torch.tensor([[T_sim, u, T_amb]], dtype=torch.float32)
            with torch.no_grad():
                dT_dt_pred = self.pinn(x_input).item()
            # Euler integration for future step
            T_sim += dT_dt_pred * self.dt
        return T_sim

    def cost_function(self, u_sequence, T_current, setpoint, T_amb):
        """Cost function evaluating setpoint tracking error and control action penalties."""
        cost = 0.0
        T_sim = T_current
        
        for u in u_sequence:
            x_input = torch.tensor([[T_sim, u, T_amb]], dtype=torch.float32)
            with torch.no_grad():
                dT_dt = self.pinn(x_input).item()
            T_sim += dT_dt * self.dt
            
            # Penalize deviation from target setpoint
            cost += (T_sim - setpoint) ** 2
            
        # Penalize excessive heater action change (smooth control effort)
        du = np.diff(u_sequence, prepend=u_sequence[0])
        cost += 0.05 * np.sum(du ** 2)
        
        return cost

    def compute_control(self, T_current, setpoint, T_amb=25.0, u_prev=0.0):
        """Solves optimization problem to find best future control sequence."""
        # Initial guess for control vector (length = Hp)
        u_init = np.ones(self.Hp) * u_prev
        
        # Bounds: Heater PWM must be between 0% and 100%
        bounds = [(0.0, 100.0) for _ in range(self.Hp)]
        
        res = minimize(
            self.cost_function,
            u_init,
            args=(T_current, setpoint, T_amb),
            method='SLSQP',
            bounds=bounds
        )
        
        # Return only the first optimal control action u(k)
        return float(res.x[0]) if res.success else float(u_init[0])


# Quick Verification Test
if __name__ == "__main__":
    nmpc = NMPCController()
    u_opt = nmpc.compute_control(T_current=30.0, setpoint=60.0)
    print(f"Computed NMPC Control Action (PWM %): {u_opt:.2f}%")