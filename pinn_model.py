import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

torch.manual_seed(42)

class ThermalPINN(nn.Module):
    def __init__(self, hidden_dim=32):
        super(ThermalPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class PINNTrainer:
    def __init__(self, tau_nominal=8.0, K_nominal=1.2, lambda_physics=0.1):
        self.tau = tau_nominal
        self.K = K_nominal
        self.lambda_physics = lambda_physics
        self.model = ThermalPINN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def load_dataset(self, csv_path="data/tclab_dynamic_data.csv"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing dataset at {csv_path}. Run generate_data.py first.")

        df = pd.read_csv(csv_path)
        
        # Load formatted dataset
        if "T_current" in df.columns and "u_delayed" in df.columns:
            X = df[["T_current", "u_delayed", "T_ambient"]].values
            y = df[["dT_dt"]].values
        # Fallback if raw Model_Data.csv is used
        elif "T1" in df.columns and "Q1" in df.columns:
            time_col = df["Time"].values
            dt = np.mean(np.diff(time_col)) if len(time_col) > 1 else 1.0
            T_raw = df["T1"].values
            u_raw = df["Q1"].values
            delay_steps = max(1, int(round(4.0 / dt)))
            u_del = np.roll(u_raw, delay_steps)
            u_del[:delay_steps] = 0.0
            dT = np.gradient(T_raw, dt)
            X = np.column_stack([T_raw, u_del, np.full(len(df), T_raw[0])])
            y = dT.reshape(-1, 1)
        else:
            raise ValueError(f"Unrecognized columns: {df.columns.tolist()}")

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def train(self, epochs=500):
        X_train, y_train = self.load_dataset()
        print(f"Training PINN on {len(X_train)} samples with physics weight λ={self.lambda_physics}...")

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            dT_pred = self.model(X_train)
            loss_data = nn.MSELoss()(dT_pred, y_train)

            # Physics Residual: dT/dt - (K*u - (T - T_amb)) / tau
            T_curr = X_train[:, 0:1]
            u_del = X_train[:, 1:2]
            T_amb = X_train[:, 2:3]

            physics_res = dT_pred - ((self.K * u_del - (T_curr - T_amb)) / self.tau)
            loss_physics = torch.mean(physics_res ** 2)

            # Thermodynamic Guardrail: when u == 0 and T > T_amb, cooling rate dT/dt must be <= 0
            cooling_violation = torch.relu(dT_pred * (u_del < 0.01).float() * (T_curr > T_amb).float())
            loss_guardrail = torch.mean(cooling_violation ** 2)

            total_loss = loss_data + self.lambda_physics * loss_physics + 0.1 * loss_guardrail
            total_loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}] | Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_physics.item():.6f}")

    def save_model(self, filepath="models/trained_pinn.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")


if __name__ == "__main__":
    trainer = PINNTrainer(tau_nominal=8.0, K_nominal=1.2, lambda_physics=0.1)
    trainer.train(epochs=600)
    trainer.save_model("models/trained_pinn.pth")