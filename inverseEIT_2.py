import numpy as np
from scipy.optimize import minimize

# --- 1. Problem Setup and Simulated Data Generation ---
L = 1.0  # m
A = 0.01 # m^2
rho = 8000 # kg/m^3
cp = 450 # J/(kg*K)
T0 = 20  # °C
T1 = 100 # °C
T2 = 20  # °C
xs = 0.5 # m
k_true = 50.0 # W/(m*K)
alpha_true = k_true / (rho * cp)
#t_meas = np.array([1,2,3,4,5,6,7,8,9,10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
t_meas = np.arange(60)+1
# Function to calculate the temperature at a specific point and time
def temperature_model(x, t, alpha, N_terms=50):
    """
    Calculates the temperature at a given point x and time t.
    N_terms is the number of terms to use in the Fourier series.
    """
    T_steady_state = (T2 - T1) * x / L + T1
    sum_series = 0.0
    for n in range(1, N_terms + 1):
        Bn = (2 / (n * np.pi)) * (T1 - T2 + (T0 - T1) * (-1)**n + (T2 - T0))
        sum_series += Bn * np.exp(-alpha * (n * np.pi / L)**2 * t) * np.sin(n * np.pi * x / L)
    return T_steady_state + sum_series

# Generate "noisy" experimental data
T_meas_true = temperature_model(xs, t_meas, alpha_true)
np.random.seed(42) # for reproducibility
noise_level = 0.25 # °C
T_meas = T_meas_true + np.random.normal(0, noise_level, len(t_meas))

# --- 2. The Inverse Problem Solver ---
def cost_function(k_guess, T_meas, t_meas, xs):
    """
    Calculates the sum of squared errors between the model's prediction
    and the measured data for a given thermal conductivity guess.
    """
    if k_guess <= 0: # Ensure k is physically meaningful
        return np.inf

    alpha_guess = k_guess[0] / (rho * cp)
    T_model = temperature_model(xs, t_meas, alpha_guess)
    return np.sum((T_model - T_meas)**2)

# Initial guess for thermal conductivity
k_initial_guess = [30.0]

# Minimize the cost function to find the best-fit k
# We use the L-BFGS-B method which is suitable for bound-constrained problems.
# Here, we constrain k to be positive.
result = minimize(cost_function, k_initial_guess, args=(T_meas, t_meas, xs), method='L-BFGS-B', bounds=[(1e-6, None)])

k_estimated = result.x[0]

# --- 3. Display Results ---
print(f"True thermal conductivity (k_true): {k_true:.2f} W/(m*K)")
print(f"Estimated thermal conductivity (k_estimated): {k_estimated:.2f} W/(m*K)")
print(f"Optimization successful: {result.success}")
print(f"Final cost function value: {result.fun:.4f}")

# Optional: Plotting the results to visualize the fit
import matplotlib.pyplot as plt

# Generate model temperatures for a smooth curve using the estimated k
t_plot = np.linspace(0, 70, 200)
T_model_fit = temperature_model(xs, t_plot, k_estimated / (rho * cp))

plt.figure(figsize=(10, 6))
plt.scatter(t_meas, T_meas, color='red', label='Noisy Measurements', zorder=5)
plt.plot(t_plot, T_model_fit, color='blue', label=f'Model Fit (k = {k_estimated:.2f})')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Inverse Problem: Temperature Measurements vs. Model Fit')
plt.legend()
plt.grid(True)
plt.show()