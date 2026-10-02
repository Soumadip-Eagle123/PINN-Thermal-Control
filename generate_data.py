import os
import numpy as np
import pandas as pd

def generate_dynamic_dataset(filename="data/tclab_dynamic_data.csv", total_time=1200.0, dt=0.5):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.random.seed(42)
    
    n_steps = int(total_time / dt)
    t = np.linspace(0, total_time, n_steps)
    
    # Multi-level PRBS excitation sequences
    u = np.zeros(n_steps)
    step_duration = int(30.0 / dt)  # Switch every 30 seconds
    current_val = 0.0
    for i in range(n_steps):
        if i % step_duration == 0:
            current_val = float(np.random.choice([0.0, 25.0, 45.0, 70.0, 95.0, 35.0, 80.0]))
        u[i] = current_val

    # True process dynamics (tau = 9.2s, K = 1.35, delay = 4.0s)
    tau_true = 9.2
    K_true = 1.35
    delay_true = 4.0
    delay_steps = int(delay_true / dt)
    T_amb = 25.0
    
    T = np.zeros(n_steps)
    T[0] = T_amb
    dT_dt = np.zeros(n_steps)
    
    for i in range(1, n_steps):
        u_del = u[i - delay_steps] if i >= delay_steps else 0.0
        rate = (K_true * u_del - (T[i-1] - T_amb)) / tau_true
        T[i] = T[i-1] + rate * dt
        dT_dt[i] = rate
    
    # Real thermistor measurement noise (sigma = 0.25 C)
    noise = np.random.normal(0.0, 0.25, n_steps)
    T_measured = T + noise
    
    u_delayed_col = np.roll(u, delay_steps)
    u_delayed_col[:delay_steps] = 0.0
    
    df = pd.DataFrame({
        "time": t,
        "T_current": np.round(T_measured, 3),
        "u_delayed": np.round(u_delayed_col, 2),
        "T_ambient": np.full(n_steps, T_amb),
        "dT_dt": np.round(dT_dt, 4)
    })
    
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {len(df)} operational samples (dt={dt}s).")

if __name__ == "__main__":
    generate_dynamic_dataset()