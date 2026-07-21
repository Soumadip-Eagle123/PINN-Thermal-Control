# Physics-Informed Neural Model Predictive Control (PINN-NMPC) for Industrial Thermal Delay Compensation

Welcome to the **PINN-NMPC Thermal Control** project repository! This project implements a **Physics-Informed Neural Network (PINN)** coupled with a **Neural Model Predictive Controller (NMPC)** to control a delayed thermal process. It features an interactive **Streamlit** dashboard comparing standard PID control against our predictive AI controller under dynamic dead-time conditions.

---

## 🛠️ Quick Start & Setup Guide (Using `uv`)

We use [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and project manager, to manage virtual environments and dependencies.

### 1. Install `uv`
If you haven't installed `uv` yet, run the appropriate command for your OS:

* **macOS / Linux:**
  ```bash
  curl -sSf https://astral.sh/uv/install.sh | sh
  ```

* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Clone the Repository & Navigate to the Project Directory

```bash
git clone <your-repository-url>
cd pinn_thermal_control
```

### 3. Create a Virtual Environment with `uv`

```bash
uv venv
```

Activate the virtual environment:

* **macOS / Linux:** `source .venv/bin/activate`
* **Windows:** `.venv\Scripts\activate`

### 4. Install Dependencies via `uv`

Install all required libraries using `uv pip`:

```bash
uv pip install -r requirements.txt
```

*(Note: `requirements.txt` contains `numpy`, `scipy`, `pandas`, `torch`, `streamlit`, `plotly`, `matplotlib`, and `pyserial`.)*

### 5. Generate Data & Train the Model

Run the following commands in sequence to create the synthetic dataset and train the PINN weights:

```bash
# Generate training data CSV
python generate_data.py

# Train the PyTorch PINN model (Saves to models/trained_pinn.pth)
python pinn_model.py
```

### 6. Launch the Interactive Web Dashboard

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 💡 What is This Project? (The Simple Explanation)

Imagine you are taking a **hot shower**. You turn the temperature knob to make it warmer, but the hot water takes **4 seconds** to travel through the pipe to reach you.

* **The Old Way (Standard PID Controller):** Because you don't feel the heat immediately, you panic and crank the knob all the way up! 4 seconds later, the water becomes **BOILING HOT**! You freeze, scream, and yank the knob back to ice-cold. The temperature swings up and down like a roller coaster. This is called **overshoot** and **oscillation**.
* **Our Way (PINN-NMPC Controller):** You have a **super-smart robot friend** who knows the exact length of the pipe and the laws of physics. It says: *"Don't turn the knob anymore! The heat is already on its way through the pipe."* It eases off the knob in advance, so the water reaches you at the **exact target temperature** with zero burning and zero freezing.

---

## 📂 Project Structure & File Explanations

```text
pinn_thermal_control/
│
├── data/
│   └── synthetic_thermal_data.csv    # Generated training dataset
├── models/
│   └── trained_pinn.pth             # Exported PyTorch PINN model weights
│
├── generate_data.py                 # Script to create the training CSV
├── plant_sim.py                     # Physical thermal system & PID baseline
├── pinn_model.py                    # PyTorch PINN architecture & training loop
├── mpc_controller.py                # Neural Model Predictive Control engine
├── app.py                           # Streamlit web interface
│
├── requirements.txt                 # List of Python package dependencies
└── README.md                        # Project documentation
```

### 1. `plant_sim.py` (The Physical World Simulator)

Simulates the physical world using dynamic differential equations. It implements a thermal plant with transport delay $L$ and includes a standard **Ziegler-Nichols tuned PID controller** with anti-windup to serve as our baseline benchmark.

### 2. `generate_data.py` (Dataset Generator)

Creates 1,000 synthetic operational samples based on physical heat balance and saves them to `data/synthetic_thermal_data.csv` for inspection and training.

### 3. `pinn_model.py` (The AI Brain)

Defines a PyTorch neural network that learns system dynamics. Unlike black-box neural networks, its loss function explicitly enforces physical laws ($\mathcal{L}_{\text{physics}}$), penalizing predictions that violate energy conservation. Saves weights to `models/trained_pinn.pth`.

### 4. `mpc_controller.py` (The Smart Decision Maker)

Implements the **Neural Model Predictive Controller (NMPC)** using `scipy.optimize.minimize`. It loads `trained_pinn.pth` to simulate future temperature trajectories over a prediction horizon $H_p$ and computes optimal heater PWM output values ($0 - 100\%$) that eliminate overshoot.

### 5. `app.py` (Interactive Web Dashboard)

Integrates all modules into a user-friendly Streamlit web application. Allows users to adjust setpoints, delays, and disturbances in real time, presenting live comparison graphs and metrics.

---

## 📊 Detailed Variable, Graph & Output Reference

### 1. Physical & Mathematical Variables

| Variable | Symbol / Name | Meaning & Unit | What It Signifies |
| --- | --- | --- | --- |
| **Setpoint** | $T_{\text{setpoint}}$ | Target Temperature ($^\circ\text{C}$) | The desired final temperature we want the system to reach and maintain. |
| **Current Temp** | $T(t)$ | Actual Temperature ($^\circ\text{C}$) | Real-time temperature measurement of the thermal plant. |
| **Control Signal** | $u(t)$ | Heater Power ($0 - 100\%$) | Duty cycle pulse-width modulation (PWM) sent to the heating element. |
| **Time Delay** | $L$ or `delay` | Dead Time (seconds) | Physical delay between sending a power signal and the sensor detecting heat transfer. |
| **Time Constant** | $\tau$ | Thermal Inertia (seconds) | Measure of how slowly or quickly the system naturally absorbs/dissipates heat. |
| **System Gain** | $K$ | Thermal Efficiency ($^\circ\text{C} / \%$ PWM) | How many degrees Celsius the temperature increases per $1\%$ of heater power. |
| **Disturbance** | $d(t)$ | External Heat Loss ($^\circ\text{C}$) | Simulated environmental shocks (e.g., opening a window or a sudden cold draft). |

---

### 2. Dashboard Metrics (The Scorecard)

* **PID Overshoot (%):** Percentage by which the PID controller exceeds the target setpoint. High overshoot ($>20\%$) indicates severe overheating risk.
* **PINN-NMPC Overshoot (%):** Percentage by which the AI controller exceeds the setpoint. Ideally $0.0\%$, showing smooth setpoint arrival.
* **PID Mean Absolute Error (MAE in °C):** Average temperature error across the 120-second run for the PID controller.
* **PINN-NMPC MAE (in °C):** Average temperature error for our AI controller. Lower values signify faster settling time and superior disturbance recovery.

---

### 3. Understanding the Graphs

#### Chart 1: Temperature Tracking Plot (°C)

* **Red Dashed Line (Setpoint Target):** Represents the target setpoint trajectory.
* **Orange Line (Classical PID Response):** Shows rapid initial rise, followed by massive overshoot, hunting oscillations, and slow recovery from external cold drafts at $t = 60\text{s}$. 
* **Green Line (PINN-NMPC Response):** Shows a smooth, critically damped temperature rise that converges directly onto the red setpoint line with zero overshoot and rapid disturbance rejection.

#### Chart 2: Control Effort / PWM Duty Cycle (%)

* **Orange Line (PID Heater Power):** Remains saturated at $100\%$ power for too long due to dead-time blindness, followed by sharp drops to $0\%$, resulting in aggressive relay-like chatter.
* **Green Line (PINN-NMPC Heater Power):** Ramps up initially, then pre-emptively throttles power down to steady-state holding levels ($\approx 30-40\%$) *before* the setpoint is reached, anticipating delayed heat arrival.

---

## 🔬 Mathematical Formulation

The physical system is governed by the delayed differential equation:

$$\tau \frac{dT(t)}{dt} + T(t) = K \cdot u(t - L) + d(t)$$

The PINN training loss combines empirical data loss with heat equation residuals:

$$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}$$

$$\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^{N} \left| \tau \frac{d\hat{T}_i}{dt} + \hat{T}_i - K \cdot u_i(t - L) \right|^2$$

The NMPC optimizer minimizes setpoint error and control action delta over prediction horizon $H_p$:

$$\min_{u} \sum_{i=1}^{H_p} \left( T_{\text{predicted}}(k+i) - T_{\text{setpoint}} \right)^2 + \rho \sum_{j=1}^{H_c} \left( \Delta u(k+j) \right)^2$$
