import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plant_sim import ThermalPlant, PIDController, SmithPredictorController
from mpc_controller import NMPCController

st.set_page_config(page_title="PINN-NMPC Thermal Delay Benchmark", layout="wide")

st.title("🔥 Physics-Informed NMPC for Delayed Industrial Thermal Control")
st.markdown("""
**Advanced Verification Benchmark:** Comparing **Classical PID**, **Classical Smith Predictor (SP)**, and **Physics-Informed NMPC** under transport delay, parameter mismatch, and sensor noise.
""")

# Sidebar Controls
st.sidebar.header("🎛️️ Process & Environmental Conditions")
setpoint = st.sidebar.slider("Target Setpoint (°C)", 35.0, 80.0, 60.0)
delay_time = st.sidebar.slider("Transport Delay L (s)", 1.0, 10.0, 4.0)
tau_val = st.sidebar.slider("Plant Time Constant τ (s)", 5.0, 15.0, 9.2)
k_gain = st.sidebar.slider("Plant Gain K (°C/%)", 1.0, 2.0, 1.38)
add_disturbance = st.sidebar.checkbox("Inject Cold Draft (-6°C at t=60s)", value=True)
add_noise = st.sidebar.checkbox("Add Thermistor Measurement Noise (σ=0.2°C)", value=True)

st.sidebar.header("📊 PID Gains (Ziegler-Nichols)")
kp = st.sidebar.number_input("Kp", value=3.5, step=0.1)
ki = st.sidebar.number_input("Ki", value=0.25, step=0.05)
kd = st.sidebar.number_input("Kd", value=1.2, step=0.1)

if st.button("🚀 Run Triple-Controller Benchmark"):
    with st.spinner("Simulating dynamics & solving NMPC optimization..."):
        sim_time = 120.0
        dt = 0.5
        n_steps = int(sim_time / dt)
        noise_level = 0.2 if add_noise else 0.0

        # Physical Plants (with deliberate parameter mismatch)
        plant_pid = ThermalPlant(tau=tau_val, K=k_gain, delay=delay_time, noise_std=noise_level)
        plant_sp = ThermalPlant(tau=tau_val, K=k_gain, delay=delay_time, noise_std=noise_level)
        plant_nmpc = ThermalPlant(tau=tau_val, K=k_gain, delay=delay_time, noise_std=noise_level)

        # Dynamic Horizon Scaling: ensures optimizer always looks beyond dead-time buffer
        scaled_Hp = int(np.ceil(delay_time / dt)) + 8

        pid = PIDController(Kp=kp, Ki=ki, Kd=kd, dt=dt)
        sp = SmithPredictorController(Kp=2.8, Ki=0.18, Kd=0.8, delay=delay_time, dt=dt)
        nmpc = NMPCController(Hp=scaled_Hp, Hc=4, dt=dt, delay=delay_time)

        # Buffers
        t_hist = [0.0]
        T_pid_hist, T_sp_hist, T_nmpc_hist = [25.0], [25.0], [25.0]
        u_pid_hist, u_sp_hist, u_nmpc_hist = [0.0], [0.0], [0.0]

        for k in range(1, n_steps):
            t_now = k * dt
            dist = -6.0 if (add_disturbance and t_now > 60.0) else 0.0

            # 1. Classical PID Step
            T_pid_curr = T_pid_hist[-1]
            u_pid = pid.compute(setpoint, T_pid_curr)
            u_pid_del = plant_pid.get_delayed_input(u_pid_hist, t_hist, t_now)
            T_pid_next = plant_pid.step(T_pid_curr, u_pid_del, dt, disturbance=dist)

            # 2. Smith Predictor Step
            T_sp_curr = T_sp_hist[-1]
            u_sp = sp.compute(setpoint, T_sp_curr)
            u_sp_del = plant_sp.get_delayed_input(u_sp_hist, t_hist, t_now)
            T_sp_next = plant_sp.step(T_sp_curr, u_sp_del, dt, disturbance=dist)

            # 3. Offset-Free PINN-NMPC Step
            T_nmpc_curr = T_nmpc_hist[-1]
            nmpc.update_disturbance(T_nmpc_curr)
            u_nmpc = nmpc.compute_control(T_nmpc_curr, setpoint, u_prev=u_nmpc_hist[-1])
            u_nmpc_del = plant_nmpc.get_delayed_input(u_nmpc_hist, t_hist, t_now)
            T_nmpc_next = plant_nmpc.step(T_nmpc_curr, u_nmpc_del, dt, disturbance=dist)

            # Append logs
            t_hist.append(t_now)
            T_pid_hist.append(T_pid_next)
            T_sp_hist.append(T_sp_next)
            T_nmpc_hist.append(T_nmpc_next)
            u_pid_hist.append(u_pid)
            u_sp_hist.append(u_sp)
            u_nmpc_hist.append(u_nmpc)

        # Performance Metric Formulations
        err_pid = np.array(T_pid_hist) - setpoint
        err_sp = np.array(T_sp_hist) - setpoint
        err_nmpc = np.array(T_nmpc_hist) - setpoint

        os_pid = max(0.0, ((max(T_pid_hist) - setpoint) / setpoint) * 100)
        os_sp = max(0.0, ((max(T_sp_hist) - setpoint) / setpoint) * 100)
        os_nmpc = max(0.0, ((max(T_nmpc_hist) - setpoint) / setpoint) * 100)

        mae_pid = float(np.mean(np.abs(err_pid)))
        mae_sp = float(np.mean(np.abs(err_sp)))
        mae_nmpc = float(np.mean(np.abs(err_nmpc)))

        ise_pid = float(np.sum(err_pid ** 2) * dt)
        ise_sp = float(np.sum(err_sp ** 2) * dt)
        ise_nmpc = float(np.sum(err_nmpc ** 2) * dt)

        tv_pid = float(np.sum(np.abs(np.diff(u_pid_hist))))
        tv_sp = float(np.sum(np.abs(np.diff(u_sp_hist))))
        tv_nmpc = float(np.sum(np.abs(np.diff(u_nmpc_hist))))

        # Display Metrics Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PID Overshoot", f"{os_pid:.1f}%")
        c2.metric("Smith Predictor OS", f"{os_sp:.1f}%")
        c3.metric("PINN-NMPC OS", f"{os_nmpc:.1f}%", delta=f"-{os_pid - os_nmpc:.1f}%")
        c4.metric("NMPC ISE Reduction", f"{((ise_pid - ise_nmpc)/ise_pid)*100:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("PID MAE", f"{mae_pid:.2f} °C")
        c6.metric("Smith Predictor MAE", f"{mae_sp:.2f} °C")
        c7.metric("PINN-NMPC MAE", f"{mae_nmpc:.2f} °C", delta=f"-{mae_pid - mae_nmpc:.2f} °C")
        c8.metric("NMPC Total Variation (TV)", f"{tv_nmpc:.0f}", delta=f"-{tv_pid - tv_nmpc:.0f}")

        # Temperature Plot
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=t_hist, y=[setpoint]*len(t_hist), name="Setpoint Target", line=dict(dash='dash', color='red')))
        fig_temp.add_trace(go.Scatter(x=t_hist, y=T_pid_hist, name="Classical PID (Oscillatory)", line=dict(color='orange')))
        fig_temp.add_trace(go.Scatter(x=t_hist, y=T_sp_hist, name="Smith Predictor Baseline", line=dict(color='blue', dash='dot')))
        fig_temp.add_trace(go.Scatter(x=t_hist, y=T_nmpc_hist, name="PINN-NMPC (Offset-Free)", line=dict(color='green', width=3)))
        fig_temp.update_layout(title="Closed-Loop Temperature Tracking (°C)", xaxis_title="Time (s)", yaxis_title="Temperature (°C)")

        # Control Effort Plot
        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=t_hist, y=u_pid_hist, name="PID PWM %", line=dict(color='orange')))
        fig_u.add_trace(go.Scatter(x=t_hist, y=u_sp_hist, name="Smith Predictor PWM %", line=dict(color='blue', dash='dot')))
        fig_u.add_trace(go.Scatter(x=t_hist, y=u_nmpc_hist, name="PINN-NMPC PWM %", line=dict(color='green')))
        fig_u.update_layout(title="Heater Actuator Effort (PWM %)", xaxis_title="Time (s)", yaxis_title="PWM Duty Cycle (%)")

        st.plotly_chart(fig_temp, use_container_width=True)
        st.plotly_chart(fig_u, use_container_width=True)