import numpy as np
from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd

# Constants
R = 8.314  # J/mol·K
mu_B_K = 0.6717  # Bohr magneton in Kelvin/T
mu_B = 9.27400968E-24 # Bohr magneton in J/T
h = 6.62607015E-34 # Js
g = 2.0
m_vals = np.arange(-7/2, 7/2 + 1, 1)  # [-3.5, -2.5, ..., 3.5]
#m_vals = np.arange(-1/2, 1/2 + 1, 1)
T = np.arange(0.1, 1, 0.01)
H = np.arange(0, 0.7, 0.01)
k_b = 1.380649E-23 #J/K

def energy_Levels_Gd(S, m, H_in_Tesla, b0_2_in_Oe, theta = np.pi/2):
    E_zeeman = g * mu_B * H_in_Tesla * m
    E_zfs = (mu_B * b0_2_in_Oe/60000) * (3 * m ** 2 - S * (S + 1)) * (3 * np.cos(theta) ** 2 - 1)
    return E_zfs + E_zeeman

def draw_theta_zero_lines(b0_2):
    listofm = {-7/2, -5/2, -3/2, -1/2, 1/2, 3/2, 5/2, 7/2}
    plt.figure(figsize=(8,5))
    for m in listofm:
        E = energy_Levels_Gd(7/2, m, H, b0_2, 0)
        E = E / (h * 1E9)
        plt.plot(H, E, color='black', label= 'm: %.1f' %(m))
    plt.xlabel('H (T)')
    plt.ylabel('E (GHz)')
    plt.title('Energy Levels ZFS b = %.0f Oe + Zeeman, theta = 0' %(b0_2))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Multilevel Schottky model (not tested)
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

# Taken from Pobell, verified wit experiment
def schottky_multilevel_pobell(T, B):
    J = 7/2
    x = mu_B_K * g * B / T
    c = R*((x/2)**2 * np.sinh(x/2)**(-2) - (x*(2*J+1)/2)**2 * np.sinh(x*(2*J + 1)/2)**(-2))
    return c


def fitHC_with_pobell():
    # --- Import data from CSV ---
    filename = '/Users/user/Nextcloud/Doktor/10_Paper/MDPI_applSci_Focus_Issue_ADR/Data/AJE/HC_ohneSilber_ohnePhononen/Ba3GdB3O9data_70000Oe.csv'  # <-- Change to your filename
    data = pd.read_csv(filename)
    #data = pd.read_excel("C:/Users/klinmarv/Nextcloud2/Doktor/10_Paper/MDPI_applSci_Focus_Issue_ADR/Data/MKL/GdB9.xlsx")

    # Adjust column names as needed
    T_all = data['T'].values
    C_all = data['Cmag'].values

    T_fit_min = 1    # <-- Set lower limit of fit range
    T_fit_max = 40   # <-- Set upper limit of fit range

    # Select only the data within the fit range
    fit_mask = (T_all >= T_fit_min) & (T_all <= T_fit_max)
    T_fit_data = T_all[fit_mask]
    C_fit_data = C_all[fit_mask]

    # --- Fit the model ---
    initial_guess = [7]  # Initial B_eff guess (in Tesla)
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
    #with pd.ExcelWriter('Ba3GdB3O9data_70000Oe.xlsx') as writer:
    #    export_df.to_excel(writer, sheet_name='Fitted Curve', index=False)
    #    fit_data_df.to_excel(writer, sheet_name='Fit Residuals', index=False)

def calculate_HC_from_ESR():
    df = pd.read_csv("Energy_Eigenvaulues.csv")
    angles = df['theta'].unique()
    cmap = plt.cm.viridis

    for theta in angles:
        theta_specific_energy_levels = df[(df['theta'] == theta)]#.any(axis=1)]
        theta_specific_energy_levels = np.delete(theta_specific_energy_levels, 0, 1)
        #theta_specific_energy_levels = theta_specific_energy_levels.to_numpy()
        magnetic_field = theta_specific_energy_levels[:,0]

        for i in range(1, theta_specific_energy_levels.shape[1]):
            color = cmap((i - 1) / (theta_specific_energy_levels.shape[1] - 2))
            plt.plot(magnetic_field, theta_specific_energy_levels[:, i], label=f'col {i}', color=color)

        #for field in magnetic_field:
        #    for i, temp in enumerate(T):
        #        Z = np.sum(np.exp(-E_i / temp))
        #        avg_E = np.sum(E_i * np.exp(-E_i / temp)) / Z
        #        avg_E2 = np.sum(E_i ** 2 * np.exp(-E_i / temp)) / Z
        #        C[i] = (avg_E2 - avg_E ** 2) / temp ** 2 * R
        #    return C



    plt.show()
    print(theta_specific_energy_levels)






calculate_HC_from_ESR()
#draw_theta_zero_lines(660)