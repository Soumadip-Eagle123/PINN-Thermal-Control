import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plant_sim import ThermalPlant, PIDController
from mpc_controller import NMPCController

st.set_page_config(page_title="PINN-NMPC Thermal Delay Control", layout="wide")

st.title("🔥 Physics-Informed NMPC for Delayed Industrial Thermal Control")
st.markdown("""
**Review 1 Software Prototype:** Comparing Classical PID vs. **Physics-Informed Neural Model Predictive Control (PINN-NMPC)** under physical transport delay (dead time).
""")

# Sidebar Controls
st.sidebar.header("🎛️ System Parameters")
setpoint = st.sidebar.slider("Target Setpoint (°C)", 35.0, 80.0, 60.0)
delay_time = st.sidebar.slider("Transport Delay / Dead Time (s)", 1.0, 10.0, 4.0)
tau_val = st.sidebar.slider("Thermal Time Constant τ (s)", 3.0, 15.0, 8.0)
add_disturbance = st.sidebar.checkbox("Inject Disturbance (Cold Draft at t=60s)", value=True)

st.sidebar.header("📊 PID Gains (Ziegler-Nichols)")
kp = st.sidebar.number_input("Kp", value=3.5, step=0.1)
ki = st.sidebar.number_input("Ki", value=0.25, step=0.05)
kd = st.sidebar.number_input("Kd", value=1.2, step=0.1)

# Run Simulation Button
if st.button("🚀 Run Comparative Simulation"):
    with st.spinner("Running Plant Simulation & NMPC Optimization..."):
        sim_time = 120.0
        dt = 0.5
        n_steps = int(sim_time / dt)

        plant_pid = ThermalPlant(tau=tau_val, K=1.2, delay=delay_time, T_ambient=25.0)
        plant_nmpc = ThermalPlant(tau=tau_val, K=1.2, delay=delay_time, T_ambient=25.0)

        pid = PIDController(Kp=kp, Ki=ki, Kd=kd, dt=dt)
        nmpc = NMPCController(Hp=8, dt=dt)

        # Buffers
        t_hist = [0.0]
        T_pid_hist = [25.0]
        T_nmpc_hist = [25.0]
        u_pid_hist = [0.0]
        u_nmpc_hist = [0.0]

        for k in range(1, n_steps):
            t_now = k * dt
            dist = -6.0 if (add_disturbance and t_now > 60.0) else 0.0

            # 1. PID Step
            T_pid_curr = T_pid_hist[-1]
            u_pid = pid.compute(setpoint, T_pid_curr)
            u_pid_del = plant_pid.get_delayed_input(u_pid_hist, t_hist, t_now)
            T_pid_next = plant_pid.step(T_pid_curr, u_pid_del, dt, disturbance=dist)

            # 2. NMPC Step
            T_nmpc_curr = T_nmpc_hist[-1]
            u_nmpc = nmpc.compute_control(T_nmpc_curr, setpoint, u_prev=u_nmpc_hist[-1])
            u_nmpc_del = plant_nmpc.get_delayed_input(u_nmpc_hist, t_hist, t_now)
            T_nmpc_next = plant_nmpc.step(T_nmpc_curr, u_nmpc_del, dt, disturbance=dist)

            # Append logs
            t_hist.append(t_now)
            T_pid_hist.append(T_pid_next)
            T_nmpc_hist.append(T_nmpc_next)
            u_pid_hist.append(u_pid)
            u_nmpc_hist.append(u_nmpc)

        # Plotting Results with Plotly
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=t_hist, y=[setpoint]*len(t_hist), name="Setpoint Target", line=dict(dash='dash', color='red')))
        fig_temp.add_trace(go.Scatter(x=t_hist, y=T_pid_hist, name="Classical PID (Oscillatory/Overshoot)", line=dict(color='orange')))
        fig_temp.add_trace(go.Scatter(x=t_hist, y=T_nmpc_hist, name="PINN-NMPC (Optimal/Smooth)", line=dict(color='green', width=3)))
        fig_temp.update_layout(title="Temperature Tracking (°C)", xaxis_title="Time (s)", yaxis_title="Temperature (°C)")

        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=t_hist, y=u_pid_hist, name="PID Heater Output (%)", line=dict(color='orange')))
        fig_u.add_trace(go.Scatter(x=t_hist, y=u_nmpc_hist, name="PINN-NMPC Heater Output (%)", line=dict(color='green')))
        fig_u.update_layout(title="Control Effort / PWM Duty Cycle (%)", xaxis_title="Time (s)", yaxis_title="Heater PWM (%)")

        # Performance Metrics Calculations
        pid_overshoot = max(0, ((max(T_pid_hist) - setpoint) / setpoint) * 100)
        nmpc_overshoot = max(0, ((max(T_nmpc_hist) - setpoint) / setpoint) * 100)
        pid_mae = np.mean(np.abs(np.array(T_pid_hist) - setpoint))
        nmpc_mae = np.mean(np.abs(np.array(T_nmpc_hist) - setpoint))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PID Overshoot", f"{pid_overshoot:.1f}%")
        col2.metric("PINN-NMPC Overshoot", f"{nmpc_overshoot:.1f}%", delta=f"-{pid_overshoot - nmpc_overshoot:.1f}%")
        col3.metric("PID Mean Error (MAE)", f"{pid_mae:.2f} °C")
        col4.metric("PINN-NMPC MAE", f"{nmpc_mae:.2f} °C", delta=f"-{pid_mae - nmpc_mae:.2f} °C")

        st.plotly_chart(fig_temp, use_container_width=True)
        st.plotly_chart(fig_u, use_container_width=True)