import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class ThermalPlant:
    """
    Physical thermal process with transport delay and unmodeled parameter mismatch.
    Differential equation:
      tau_true * dT/dt + (T - T_ambient) = K_true * u(t - delay) + disturbance
    """
    def __init__(self, tau=9.2, K=1.38, delay=4.0, T_ambient=25.0, noise_std=0.2):
        self.tau = tau
        self.K = K
        self.delay = delay
        self.T_ambient = T_ambient
        self.noise_std = noise_std

    def get_delayed_input(self, u_history, time_history, current_time):
        """Retrieves past input u(t - delay) using linear interpolation."""
        target_time = current_time - self.delay
        if target_time <= 0 or len(time_history) == 0:
            return 0.0
        return float(np.interp(target_time, time_history, u_history))

    def step(self, T_current, u_delayed, dt, disturbance=0.0):
        """Solves one dt step using RK45 and adds sensor noise."""
        def deriv(t, T):
            return (self.K * u_delayed - (T - self.T_ambient) + disturbance) / self.tau

        sol = solve_ivp(deriv, [0, dt], [T_current], method='RK45')
        clean_temp = sol.y[0][-1]
        noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0
        return clean_temp + noise


class PIDController:
    """Standard discrete PID Controller with Anti-Windup clamping."""
    def __init__(self, Kp=3.5, Ki=0.25, Kd=1.2, dt=0.5, u_min=0.0, u_max=100.0):
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
        P = self.Kp * error
        self.integral += error * self.dt
        I = self.Ki * self.integral
        D = self.Kd * ((error - self.prev_error) / self.dt)

        u_raw = P + I + D
        u_clamped = np.clip(u_raw, self.u_min, self.u_max)

        # Anti-windup
        if u_raw != u_clamped and self.Ki != 0:
            self.integral -= error * self.dt

        self.prev_error = error
        return float(u_clamped)


class SmithPredictorController:
    """
    Classical Smith Predictor (SP) dead-time compensator.
    Controls delay-free internal model and cancels delay effects via prediction bias.
    """
    def __init__(self, Kp=2.8, Ki=0.18, Kd=0.8, tau_m=8.0, K_m=1.2, delay=4.0, dt=0.5):
        self.pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt)
        self.tau_m = tau_m
        self.K_m = K_m
        self.delay = delay
        self.dt = dt
        self.delay_steps = max(1, int(round(delay / dt)))

        self.T_model_nodelay = 25.0
        self.T_model_delayed = 25.0
        self.u_buffer = [0.0] * self.delay_steps

    def compute(self, setpoint, y_measured, T_amb=25.0):
        # 1. Disturbance estimation: difference between plant and delayed internal model
        d_est = y_measured - self.T_model_delayed

        # 2. Predicted delay-free state feedback
        T_pred = self.T_model_nodelay + d_est

        # 3. PID control action calculated on delay-free state
        u = self.pid.compute(setpoint, T_pred)

        # 4. Integrate delay-free model
        dT_nodelay = (self.K_m * u - (self.T_model_nodelay - T_amb)) / self.tau_m
        self.T_model_nodelay += dT_nodelay * self.dt

        # 5. Integrate delayed model using delayed control history
        u_delayed = self.u_buffer.pop(0)
        dT_delayed = (self.K_m * u_delayed - (self.T_model_delayed - T_amb)) / self.tau_m
        self.T_model_delayed += dT_delayed * self.dt

        self.u_buffer.append(u)
        return float(u)


if __name__ == "__main__":
    print("Testing Plant and Controller Simulation...")
    sim_time = 120.0
    dt = 0.5
    n_steps = int(sim_time / dt)

    plant = ThermalPlant(tau=9.2, K=1.38, delay=4.0, noise_std=0.2)
    pid = PIDController(Kp=3.5, Ki=0.25, Kd=1.2, dt=dt)
    sp = SmithPredictorController(delay=4.0, dt=dt)

    t_hist, T_pid_hist, T_sp_hist = [0.0], [25.0], [25.0]
    u_pid_hist, u_sp_hist = [0.0], [0.0]
    setpoint = 60.0

    for k in range(1, n_steps):
        t_now = k * dt
        dist = -5.0 if t_now > 70.0 else 0.0

        # PID Step
        u_p = pid.compute(setpoint, T_pid_hist[-1])
        u_p_del = plant.get_delayed_input(u_pid_hist, t_hist, t_now)
        T_pid_hist.append(plant.step(T_pid_hist[-1], u_p_del, dt, disturbance=dist))
        u_pid_hist.append(u_p)

        # Smith Predictor Step
        u_s = sp.compute(setpoint, T_sp_hist[-1])
        u_s_del = plant.get_delayed_input(u_sp_hist, t_hist, t_now)
        T_sp_hist.append(plant.step(T_sp_hist[-1], u_s_del, dt, disturbance=dist))
        u_sp_hist.append(u_s)

        t_hist.append(t_now)

    plt.figure(figsize=(10, 5))
    plt.plot(t_hist, [setpoint]*len(t_hist), 'r--', label='Target')
    plt.plot(t_hist, T_pid_hist, 'orange', label='Classical PID')
    plt.plot(t_hist, T_sp_hist, 'b-', label='Smith Predictor')
    plt.title("Baseline Plant Check: PID vs. Smith Predictor")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig("pid_baseline_test.png")
    print("Baseline test complete! Plot saved as 'pid_baseline_test.png'.")