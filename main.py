import numpy as np
from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd

# Constants
R = 8.314  # J/mol·K
mu_B_K = 0.6717  # Bohr magneton in Kelvin/T
g = 2.0
#m_vals = np.arange(-7/2, 7/2 + 1, 1)  # [-3.5, -2.5, ..., 3.5]
m_vals = np.arange(-1/2, 1/2 + 1, 1)
T = np.arange(0.1, 10, 0.01)

# Multilevel Schottky model
def schottky_multilevel(T, B_eff):
    T = np.array(T)
    C = np.zeros_like(T)
    for i, temp in enumerate(T):
        E_i = -g * mu_B_K * B_eff * m_vals  # Energies in Kelvin
        Z = np.sum(np.exp(-E_i / temp))
        avg_E = np.sum(E_i * np.exp(-E_i / temp)) / Z
        avg_E2 = np.sum(E_i**2 * np.exp(-E_i / temp)) / Z
        C[i] = (avg_E2 - avg_E**2) / temp**2 * R
    return C


def schottky_multilevel_pobell(T, B):
    J = 2.2
    x = mu_B_K * g * B / T
    c = R*((x/2)**2 * np.sinh(x/2)**(-2) - (x*(2*J+1)/2)**2 * np.sinh(x*(2*J + 1)/2)**(-2))
    return c

#c = schottky_multilevel(T, 1)
#d = schottky_multilevel(T, 2)
#e = schottky_multilevel(T, 3)

#plt.plot(T, c, label='1')
#plt.plot(T, d, label='2')
#plt.plot(T, e, label='3')
#plt.xlabel('Temperature (K)')
#.ylabel('Heat Capacity (J/mol·K)')
#plt.legend()
#plt.show()

# --- Import data from CSV ---
filename = './hc/Ba3GdB9O18data_00000Oe.csv'  # <-- Change to your filename
#data = pd.read_csv(filename)
data = pd.read_excel("C:/Users/klinmarv/Nextcloud2/Doktor/10_Paper/MDPI_applSci_Focus_Issue_ADR/Data/MKL/GdB9.xlsx")

# Adjust column names as needed
T_all = data['T'].values
C_all = data['Cmag'].values

T_fit_min = 0.22    # <-- Set lower limit of fit range
T_fit_max = 0.8   # <-- Set upper limit of fit range

# Select only the data within the fit range
fit_mask = (T_all >= T_fit_min) & (T_all <= T_fit_max)
T_fit_data = T_all[fit_mask]
C_fit_data = C_all[fit_mask]

# --- Fit the model ---
initial_guess = [0.001]  # Initial B_eff guess (in Tesla)
popt, pcov = curve_fit(schottky_multilevel_pobell, T_fit_data, C_fit_data, p0=initial_guess)
B_eff_fit = popt[0]

# --- Generate model for plotting ---
T_plot = np.linspace(0.1, max(T_all), 1000)
C_plot = schottky_multilevel_pobell(T_plot, B_eff_fit)

plt.figure(figsize=(8,5))
plt.plot(T_all, C_all, 'o', label='Experimental Data')
plt.plot(T_plot, C_plot, '-', label=f'Schottky Fit (B_eff = {B_eff_fit:.3f} T)')
plt.axvline(T_fit_min, color='gray', linestyle='--', linewidth=1)
plt.axvline(T_fit_max, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Temperature (K)')
plt.xscale('log')
plt.ylabel('Heat Capacity (J/mol·K)')
plt.title('Multilevel Schottky Fit to Gd³⁺ Heat Capacity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- EXPORT ---
export_filename = 'Ba3GdB9O18data_00000Oe.csv'
export_df = pd.DataFrame({
    'Temperature (K)': T_plot,
    'Fitted Heat Capacity (J/mol·K)': C_plot
})

# Optional: include residuals in the fit range
C_fit_model_in_range = schottky_multilevel(T_fit_data, B_eff_fit)
residuals = C_fit_data - C_fit_model_in_range
fit_data_df = pd.DataFrame({
    'T_fit (K)': T_fit_data,
    'C_exp (J/mol·K)': C_fit_data,
    'C_model (J/mol·K)': C_fit_model_in_range,
    'Residual (C_exp - C_model)': residuals
})

# Export both datasets to CSV
with pd.ExcelWriter('Ba3GdB9O18data_00000OeADR.xlsx') as writer:
    export_df.to_excel(writer, sheet_name='Fitted Curve', index=False)
    fit_data_df.to_excel(writer, sheet_name='Fit Residuals', index=False)