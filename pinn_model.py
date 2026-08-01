import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Set random seed for reproducibility
torch.manual_seed(42)

class ThermalPINN(nn.Module):
    """
    Physics-Informed Neural Network predicting future temperature state.
    Inputs:  [T_current, u_delayed, ambient_temp]
    Output:  [dT/dt (rate of temperature change)]
    """
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
    """
    Handles data generation, physics loss formulation, and PyTorch training loop.
    """
    def __init__(self, tau=8.0, K=1.2, lambda_physics=0.1):
        self.tau = tau                  # Physical time constant
        self.K = K                      # Physical system gain
        self.lambda_physics = lambda_physics
        self.model = ThermalPINN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def generate_synthetic_data(self, n_samples=1000):

        dt = 0.5

        X = []
        y = []

        T = 25.0
        T_amb = 25.0

        for _ in range(n_samples):

            # Random heater command
            u = np.random.uniform(0, 100)

            # Physical model
            dT_dt = (self.K * u - (T - T_amb)) / self.tau

            # Save current state
            X.append([T, u, T_amb])
            y.append([dT_dt])

            # Move to the next temperature
            T = T + dT_dt * dt

            # Keep temperature realistic
            T = np.clip(T, 20, 120)

        X = np.array(X)
        y = np.array(y)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

    def train(self, epochs=500):
        X_train, y_train = self.generate_synthetic_data()
        
        print("Training PINN Model with Physics Loss Enforcement...")
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass: Data loss (MSE)
            dT_pred = self.model(X_train)
            loss_data = nn.MSELoss()(dT_pred, y_train)

            # Physics loss enforcement: Residual of differential equation
            # Res: dT/dt_pred - (K * u - (T - T_amb)) / tau
            T_curr = X_train[:, 0:1]
            u_del = X_train[:, 1:2]
            T_amb = X_train[:, 2:3]

            physics_residual = dT_pred - ((self.K * u_del - (T_curr - T_amb)) / self.tau)
            loss_physics = torch.mean(physics_residual ** 2)

            # Total combined PINN loss
            total_loss = loss_data + self.lambda_physics * loss_physics

            total_loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}] | Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_physics.item():.6f}")

    def save_model(self, filepath="models/trained_pinn.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"PINN Model saved successfully to {filepath}")


# =====================================================================
# VERIFICATION RUN FOR TEAMMATE 2
# =====================================================================
if __name__ == "__main__":
    trainer = PINNTrainer(tau=8.0, K=1.2, lambda_physics=0.2)
    trainer.train(epochs=600)
    trainer.save_model("models/trained_pinn.pth")