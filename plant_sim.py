import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class ThermalPlant:
    """
    Simulates a first-order thermal process with time delay (dead time).
    Differential equation: tau * dT/dt + T(t) = K * u(t - delay) + d(t)
    """
    def __init__(self, tau=10.0, K=1.5, delay=5.0, T_ambient=25.0):
        self.tau = tau          # Thermal time constant (seconds)
        self.K = K              # System gain (deg C / % PWM)
        self.delay = delay      # Transport delay / dead time (seconds)
        self.T_ambient = T_ambient  # Ambient baseline temperature (deg C)
        
    def get_delayed_input(self, u_history, time_history, current_time):
        """Retrieves input u(t - delay) from past history buffer."""
        target_time = current_time - self.delay
        if target_time <= 0 or len(time_history) == 0:
            return 0.0  # Zero input before delay threshold
        
        # Linear interpolation of input history at (current_time - delay)
        return float(np.interp(target_time, time_history, u_history))

    def step(self, T_current, u_delayed, dt, disturbance=0.0):
        """
        Solves one time-step dt of the differential equation using RK45.
        dT/dt = (K * u_delayed - (T - T_ambient) + disturbance) / tau
        """
        def deriv(t, T):
            return (self.K * u_delayed - (T - self.T_ambient) + disturbance) / self.tau

        sol = solve_ivp(deriv, [0, dt], [T_current], method='RK45')
        return sol.y[0][-1]


class PIDController:
    """
    Standard discrete PID Controller with Anti-Windup.
    """
    def __init__(self, Kp=2.0, Ki=0.1, Kd=0.5, dt=0.5, u_min=0.0, u_max=100.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, current_value):
        error = setpoint - current_value
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup clamping
        self.integral += error * self.dt
        I = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / self.dt
        D = self.Kd * derivative
        
        # Total unconstrained control action
        u_raw = P + I + D
        
        # Clamp output to physical limits (0% to 100% PWM)
        u_clamped = np.clip(u_raw, self.u_min, self.u_max)
        
        # Anti-windup adjustment
        if u_raw != u_clamped and self.Ki != 0:
            self.integral -= error * self.dt  # Freeze integration during saturation

        self.prev_error = error
        return u_clamped


# =====================================================================
# VERIFICATION RUN & GRAPH GENERATION FOR REVIEW 1
# =====================================================================
if __name__ == "__main__":
    print("Running Teammate 1 Verification Test...")

    # Simulation setup
    sim_time = 120.0  # seconds
    dt = 0.5          # time step (s)
    n_steps = int(sim_time / dt)

    plant = ThermalPlant(tau=8.0, K=1.2, delay=4.0, T_ambient=25.0)
    
    # Ziegler-Nichols tuned PID (Aggressive to highlight time-delay instability)
    pid = PIDController(Kp=3.5, Ki=0.25, Kd=1.2, dt=dt)

    # History buffers
    time_hist = [0.0]
    temp_hist = [plant.T_ambient]
    u_hist = [0.0]
    setpoint_hist = [25.0]

    setpoint = 60.0  # Target temperature in Celsius

    for k in range(1, n_steps):
        t_now = k * dt
        T_now = temp_hist[-1]
        
        # 1. Compute control action u(t) from PID
        u_control = pid.compute(setpoint, T_now)
        
        # 2. Get past delayed input u(t - delay)
        u_delayed = plant.get_delayed_input(u_hist, time_hist, t_now)
        
        # 3. Simulate sudden cold draft disturbance at t = 70s
        dist = -5.0 if t_now > 70.0 else 0.0

        # 4. Step the physical plant forward in time
        T_next = plant.step(T_now, u_delayed, dt, disturbance=dist)

        # Store logs
        time_hist.append(t_now)
        temp_hist.append(T_next)
        u_hist.append(u_control)
        setpoint_hist.append(setpoint)

    # Plot results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_hist, setpoint_hist, 'r--', label='Target Setpoint (°C)')
    plt.plot(time_hist, temp_hist, 'b-', label='PID Response (T)')
    plt.axvline(x=70, color='gray', linestyle=':', label='Disturbance (Cold Draft)')
    plt.title("Classical PID Instability & Overshoot under 4.0s Dead Time Delay")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_hist, u_hist, 'g-', label='Control Effort (PWM %)')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heater PWM Output (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("pid_baseline_test.png")
    print("Simulation complete! Benchmark plot saved as 'pid_baseline_test.png'.")