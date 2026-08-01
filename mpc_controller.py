import torch
import numpy as np
from scipy.optimize import minimize
from pinn_model import ThermalPINN

class NMPCController:
    """
    Neural Model Predictive Controller (NMPC).
    Uses the trained PINN model as an internal predictive engine over horizon Hp.
    """
    def __init__(self, model_path="models/trained_pinn.pth", Hp=10, dt=0.5, delay=4.0):
        self.Hp = Hp  # Prediction horizon (number of look-ahead steps)
        self.dt = dt  # Time step (seconds)
        self.delay = delay  
        self.delay_steps = int(delay / dt) 
        # Load pre-trained PyTorch PINN model
        self.pinn = ThermalPINN()
        try:
            self.pinn.load_state_dict(torch.load(model_path))
            self.pinn.eval()
            print("NMPC Controller: PINN model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load PINN weights from {model_path}. Using untrained weights.")

    def predict_future(
        self,
        T_start,
        u_sequence,
        u_prev,
        T_amb=25.0):

        T = T_start

        delay_buffer = [u_prev]*self.delay_steps

        for u in u_sequence:
            delayed_u = delay_buffer.pop(0)
            x = torch.tensor(
                [[T, delayed_u, T_amb]],
                dtype=torch.float32
            )

            with torch.no_grad():
                dT = self.pinn(x).item()

            T += dT * self.dt
            delay_buffer.append(u)

        return T

    def cost_function(
        self,
        u_sequence,
        T_current,
        setpoint,
        T_amb,
        u_prev):

        T = T_current

        delay_buffer = [u_prev]*self.delay_steps

        tracking_cost = 0
        effort_cost = 0

        prev = u_prev

        for u in u_sequence:

            delayed_u = delay_buffer.pop(0)

            x = torch.tensor(
                [[T, delayed_u, T_amb]],
                dtype=torch.float32
            )

            with torch.no_grad():
                dT = self.pinn(x).item()

            T += dT*self.dt

            tracking_cost += 10*(T-setpoint)**2

            effort_cost += 0.001*(u-prev)**2

            prev = u

            delay_buffer.append(u)

        terminal_cost = 100*(T-setpoint)**2

        return tracking_cost + effort_cost + terminal_cost

    def compute_control(self, T_current, setpoint, T_amb=25.0, u_prev=0.0):
        """Solves optimization problem to find best future control sequence."""
        # Initial guess for control vector (length = Hp)
        u_ss = max(0, min(100, (setpoint-T_amb)/1.2))
        u_init = np.ones(self.Hp) * u_ss 
        
        # Bounds: Heater PWM must be between 0% and 100%
        bounds = [(0.0, 100.0) for _ in range(self.Hp)]
        
        res = minimize(
            self.cost_function,
            u_init,
            args=(T_current, setpoint, T_amb, u_prev),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-5}
        )
        
        # Return only the first optimal control action u(k)
        return np.clip(float(res.x[0]), 0.0, 100.0) 


# Quick Verification Test
if __name__ == "__main__":
    nmpc = NMPCController()
    u_opt = nmpc.compute_control(T_current=30.0, setpoint=60.0)
    print(f"Computed NMPC Control Action (PWM %): {u_opt:.2f}%")