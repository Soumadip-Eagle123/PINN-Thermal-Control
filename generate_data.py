import numpy as np
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Generate 1,000 synthetic operational data points
np.random.seed(42)
n_samples = 1000

tau = 8.0   # Thermal time constant
K = 1.2     # System gain

T_current = np.random.uniform(25.0, 85.0, n_samples)      # Current Temp (°C)
u_delayed = np.random.uniform(0.0, 100.0, n_samples)      # Delayed PWM (%)
T_ambient = np.random.uniform(20.0, 30.0, n_samples)      # Ambient Temp (°C)

# Heat differential equation: dT/dt = (K * u - (T - T_amb)) / tau
dT_dt = (K * u_delayed - (T_current - T_ambient)) / tau

# Create DataFrame
df = pd.DataFrame({
    'T_current': T_current,
    'u_delayed': u_delayed,
    'T_ambient': T_ambient,
    'dT_dt_target': dT_dt
})

# Save to CSV
df.to_csv("data/synthetic_thermal_data.csv", index=False)
print("`data/synthetic_thermal_data.csv` successfully generated with 1,000 data rows!")