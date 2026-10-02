import torch
import numpy as np
from scipy.optimize import minimize
from pinn_model import ThermalPINN

class NMPCController:
    """
    Fast Neural Model Predictive Controller (NMPC).
    Decouples prediction horizon (Hp) from control horizon (Hc=4)
    to eliminate CPU optimization bottlenecks under transport delay.
    """
    def __init__(self, model_path="models/trained_pinn.pth", Hp=20, Hc=4, dt=0.5, delay=4.0):
        self.dt = dt
        self.delay = delay
        self.delay_steps = max(1, int(round(delay / dt)))
        self.Hp = max(self.delay_steps + 6, Hp)
        self.Hc = min(Hc, self.Hp)

        self.pinn = ThermalPINN()
        try:
            self.pinn.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.pinn.eval()
            print("NMPC Controller: Trained PINN weights loaded successfully.")
        except Exception:
            print(f"Warning: {model_path} not found. Running with uncalibrated weights.")

        # Disturbance estimator states
        self.d_hat = 0.0
        self.prev_model_pred = 25.0

    def update_disturbance(self, y_measured):
        """Offset-free bias estimation: d(k) = y_measured(k) - y_model(k)"""
        raw_error = y_measured - self.prev_model_pred
        # Low-pass filter to reject sensor noise while capturing slow offsets
        self.d_hat = 0.85 * self.d_hat + 0.15 * raw_error

    def cost_function(self, u_ctrl, T_current, setpoint, T_amb, u_prev):
        # Expand small Hc (4 variables) across the full prediction horizon Hp
        u_sequence = np.zeros(self.Hp)
        u_sequence[:self.Hc] = u_ctrl
        u_sequence[self.Hc:] = u_ctrl[-1]  # Hold final control action across remainder

        T = T_current
        delay_buf = [u_prev] * self.delay_steps
        cost = 0.0
        prev_u = u_prev

        # Adaptive Sigmoid Weights (Selvamurugan et al.)
        err = abs(T_current - setpoint)
        W_E = 15.0 / (1.0 + np.exp(-0.25 * err))
        W_u = 0.15 / (1.0 + np.exp(-0.1 * (8.0 - err)))

        for u in u_sequence:
            u_del = delay_buf.pop(0)
            x_in = torch.tensor([[T, u_del, T_amb]], dtype=torch.float32)
            with torch.no_grad():
                dT = self.pinn(x_in).item()

            T += dT * self.dt
            T_corrected = T + self.d_hat

            cost += W_E * (T_corrected - setpoint) ** 2
            cost += W_u * (u - prev_u) ** 2
            prev_u = u
            delay_buf.append(u)

        # Terminal state penalty
        cost += 80.0 * (T + self.d_hat - setpoint) ** 2
        return cost

    def compute_control(self, T_current, setpoint, T_amb=25.0, u_prev=0.0):
        # Initial guess informed by steady-state gain
        u_ss = np.clip((setpoint - T_amb) / 1.2, 0.0, 100.0)
        u_init = np.ones(self.Hc) * (0.6 * u_prev + 0.4 * u_ss)
        bounds = [(0.0, 100.0) for _ in range(self.Hc)]

        res = minimize(
            self.cost_function,
            u_init,
            args=(T_current, setpoint, T_amb, u_prev),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 15, 'ftol': 1e-2, 'eps': 5e-2}
        )

        u_optimal = float(res.x[0]) if res.success else float(u_init[0])
        u_optimal = float(np.clip(u_optimal, 0.0, 100.0))

        # Expected 1-step prediction for the next disturbance update cycle
        u_del_first = u_prev if self.delay_steps > 0 else u_optimal
        x_step = torch.tensor([[T_current, u_del_first, T_amb]], dtype=torch.float32)
        with torch.no_grad():
            dT_step = self.pinn(x_step).item()
        self.prev_model_pred = T_current + dT_step * self.dt

        return u_optimal