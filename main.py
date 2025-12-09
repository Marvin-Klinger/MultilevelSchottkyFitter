import time
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

# Constants
R = 8.314  # J/mol·K
mu_B_K = 0.6717  # Bohr magneton in Kelvin/T
mu_B = 9.27400968E-24 # Bohr magneton in J/T
h = 6.62607015E-34 # Js
g = 2.0
k_b = 1.380649E-23 # k-Boltzman in J/K
m_vals = np.arange(-7/2, 7/2 + 1, 1)  # [-3.5, -2.5, ..., 3.5] for Gadolinium

# Temperature scale
Tmin, Tmax = 1e-2, 1e1
n = 200
T = np.logspace(np.log10(Tmin), np.log10(Tmax), num=n) #T = np.arange(0.01, 10, 0.01) linear alternative

H = np.arange(0, 0.7, 0.01)


def energy_Levels_Gd(S, m, H_in_Tesla, b0_2_in_Oe, theta = np.pi/2):
    """
        Calculate energy levels in Gd(III) ion.
        This is only ok for very high fields or low ZFS!

        Parameters:
        -----------
        S : Total spin (7/2)
        m : Magnetic quantum number
        H_in_Tesla : magnetic field H in Tesla
        b0_2_in_Oe : CEF paramter of lowest order in Oersted
        theta : theta value in DEGREES


        Returns:
        --------
        E : Energy of the specified level
        """
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


def plot_energy_levels(filename, combined = False):
    """
    Plot energy levels calculated by SPEKTRUM program E(H)

    Parameters:
    -----------
    filename : file name of SPEKTRUM output file
    combined : bool combine plots into single graph (default False)
    --------
    """
    df = pd.read_csv(filename, header=None, names=['theta','H','0','1','2','3','4','5','6','7'])
    angles = df['theta'].unique()
    cmap = plt.cm.viridis

    for j, theta in enumerate(angles):
        theta_specific_energy_levels = df[(df['theta'] == theta)]
        theta_specific_energy_levels = np.delete(theta_specific_energy_levels, 0, 1)
        magnetic_field = theta_specific_energy_levels[:,0]

        for i in range(1, theta_specific_energy_levels.shape[1]):
            color = cmap((j - 1) / (len(angles) - 2))

            last_line = theta_specific_energy_levels[-1,1:]
            offset_ammount = np.min(last_line)
            to_draw = theta_specific_energy_levels[:, i] - offset_ammount
            plt.plot(magnetic_field, to_draw, label=f'col {i}', color=color)

        plt.xlabel('Feld (Oe)')
        plt.ylabel('Energie (GHz)')
        if not combined:
            plt.show()
    if combined:
        plt.show()


def plot_energy_levels_by_angle(filename):
    """
    Plot energy levels calculated by SPEKTRUM program E(H) for every angle

    Parameters:
    -----------
    filename : file name of SPEKTRUM output file
    --------
    """
    df = pd.read_csv(filename)
    angles = df['theta'].unique()
    n_angles = len(angles)

    # Create figure with subplots (5 rows if many angles, adjust as needed)
    n_cols = min(5, n_angles)
    n_rows = (n_angles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             sharex=True, sharey=True)
    if n_angles == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    norm = Normalize(vmin=0, vmax=8)
    cmap = plt.cm.viridis

    for i, theta in enumerate(angles):
        ax = axes[i]

        # Plot all field curves for this angle
        theta_specific_energy_levels = df[(df['theta'] == theta)]
        theta_specific_energy_levels = np.delete(theta_specific_energy_levels, 0, 1)
        magnetic_field = theta_specific_energy_levels[:, 0]
        for i in range(1, theta_specific_energy_levels.shape[1]):
            color = cmap(norm(i))
            ax.plot(magnetic_field, theta_specific_energy_levels[:, i], color=color)
            #ax.plot(T, C_3d[i, j, :], color=color, linewidth=1.5, alpha=0.8)

        ax.set_title(f'θ = {theta}°', fontsize=12, fontweight='bold')
        ax.set_xlabel('Field (T)')
        ax.set_ylabel('Energy (GHz)')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_angles, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Energieniveaus für alle θ',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def calculate_hc_from_esr(filename, T = np.logspace(-2, 1, 100)):
    """
    Calculates specific heat C(theta,H,T) from ESR-energy level simulation.
    The output of this function is used by other functions for weighing, averaging and plotting.


    Parameters:
    -----------
    filename : file name of SPEKTRUM output file
    T : array of temperature values

    Returns:
    --------
    C_3d : ndarray shape (n_angles, n_fields, n_temps)
    """
    df = pd.read_csv(filename, header=None, names=['theta','H','0','1','2','3','4','5','6','7'])
    angles = df['theta'].unique()
    R = 8.314  # Gas constant in J/mol·K

    # find all magnetic fields used in SPEKTRUM
    all_theta_data = df[df['theta'].isin(angles)].iloc[:, 1:].values
    unique_fields = np.unique(all_theta_data[:, 0])  # First column = fields
    n_unique_fields = len(unique_fields)

    # 3D array for storing the C data
    C_3d = np.zeros((len(angles), n_unique_fields, len(T)))

    field_to_idx = {field: idx for idx, field in enumerate(unique_fields)}

    for angle_idx, theta in enumerate(angles):
        theta_df = df[df['theta'] == theta].iloc[:, 1:].values
        magnetic_field = theta_df[:, 0]

        offset_energy_ghz = np.min(theta_df[-1, 1:9])
        for j, field in enumerate(magnetic_field):
            field_idx = field_to_idx[field]  # Map to unique field index
            E_i_GHz = theta_df[j, 1:9] - offset_energy_ghz
            E_i = E_i_GHz * 4.799243E-2 # convert GHz energy to Kelvin

            for i, temp in enumerate(T):
                if temp == 0: continue
                z = np.sum(np.exp(-E_i / temp))
                avg_e = np.sum(E_i * np.exp(-E_i / temp)) / z
                avg_e2 = np.sum(E_i ** 2 * np.exp(-E_i / temp)) / z
                C_3d[angle_idx, field_idx, i] = (avg_e2 - avg_e ** 2) / temp ** 2 * R

    return C_3d, angles, unique_fields  # unique_fields has no duplicates


def average_c_over_theta(C_3d, angles):
    """
    Calculate average specific heat C over theta angles.

    Parameters:
    -----------
    C_3d : ndarray shape (n_angles, n_fields, n_temps)
    angles : array of theta values

    Returns:
    --------
    C_avg : ndarray shape (n_fields, n_temps) - averaged over theta
    std_C : ndarray shape (n_fields, n_temps) - standard deviation over theta
    """
    import numpy as np

    # Average over axis 0 (theta dimension)
    C_avg = np.mean(C_3d, axis=0)  # shape: (n_fields, n_temps)
    std_C = np.std(C_3d, axis=0)  # shape: (n_fields, n_temps)

    return C_avg, std_C


def average_C_over_theta_weighted(C_3d, angles, weighting='sin_theta', deviation = 0.0):
    """
    Average specific heat C over theta angles with spherical weighting.

    Parameters:
    -----------
    C_3d : ndarray shape (n_angles, n_fields, n_temps)
    angles : array of theta values in DEGREES
    weighting : str {'sin_theta', 'uniform'} - 'sin_theta' for spherical weighting
    deviation : angle by which the average grain is rotated in the pellet

    Returns:
    --------
    C_avg : ndarray shape (n_fields, n_temps) - weighted average over theta
    std_C : ndarray shape (n_fields, n_temps) - weighted standard deviation
    weights : ndarray - normalization weights used
    """
    import numpy as np
    theta_rad = np.deg2rad(angles + deviation)  # Convert to radians

    if weighting == 'sin_theta':
        # Spherical weighting: sin(θ) dθ distribution
        weights = np.sin(theta_rad)
    else:
        # Uniform weighting
        weights = np.ones_like(theta_rad)

    # Normalize weights
    weights /= np.sum(weights)

    # Weighted mean: sum(w_i * x_i) / sum(w_i)
    C_avg = np.average(C_3d, axis=0, weights=weights)

    # Weighted standard deviation
    weighted_var = np.average((C_3d - C_avg) ** 2, axis=0, weights=weights)
    std_C = np.sqrt(weighted_var)

    return C_avg, std_C, weights


def plot_averaging_comparison(C_3d, angles, fields, T):
    """
    Create comprehensive comparison plots: uniform vs weighted averaging.

    Parameters:
    -----------
    C_3d : ndarray (n_angles, n_fields, n_temps)
    angles, fields, T : coordinate arrays
    """
    # Compute both averages
    C_avg_uniform, _ = average_c_over_theta(C_3d, angles)  # Uniform
    C_avg_weighted, _, weights = average_C_over_theta_weighted(C_3d, angles, 'sin_theta')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: C vs T comparison for selected fields
    fields_to_plot = [0, len(fields) // 2, -1]  # First, middle, last field
    colors = plt.cm.viridis(np.linspace(0, 1, 3))

    for idx, field_idx in enumerate(fields_to_plot):
        # Uniform (solid)
        axes[0, 0].plot(T, C_avg_uniform[field_idx], label=f'Unif B={fields[field_idx]:.1f} Oe',
                        color=colors[idx], linewidth=3)
        # Weighted (dashed)
        axes[0, 0].plot(T, C_avg_weighted[field_idx], label=f'Wt B={fields[field_idx]:.1f} Oe',
                        color=colors[idx], linestyle='--', linewidth=3)

    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Temperature (K)')
    axes[0, 0].set_ylabel('Specific Heat C (J/mol·K)')
    axes[0, 0].set_title('Uniform vs sin(θ)-Weighted: C vs T')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Difference contour (weighted - uniform)
    diff = C_avg_weighted - C_avg_uniform
    im = axes[0, 1].contourf(fields, T, diff.T, levels=20, cmap='RdBu_r', vmin=-np.max(np.abs(diff)),
                             vmax=np.max(np.abs(diff)))
    axes[0, 1].set_xlabel('Magnetic Field (Oe)')
    axes[0, 1].set_ylabel('Temperature (K)')
    axes[0, 1].set_title('ΔC = Weighted - Uniform')
    plt.colorbar(im, ax=axes[0, 1])

    # Plot 3: sin(θ) weights distribution
    theta_rad = np.deg2rad(angles)
    axes[1, 0].plot(angles, weights, 'o-', linewidth=3, markersize=8, color='darkblue')
    axes[1, 0].fill_between(angles, 0, weights, alpha=0.3, color='blue')
    axes[1, 0].set_xlabel('θ (degrees)')
    axes[1, 0].set_ylabel('Normalized Weight')
    axes[1, 0].set_title('Spherical Weighting: sin(θ)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Relative difference vs field at peak temperature
    mid_temp_idx = len(T) // 4  # ~T=0.1K where C peaks
    rel_diff = 100 * np.abs(C_avg_weighted[:, mid_temp_idx] - C_avg_uniform[:, mid_temp_idx]) / (
                C_avg_uniform[:, mid_temp_idx] + 1e-10)
    axes[1, 1].plot(fields, rel_diff, 'ro-', linewidth=3, markersize=6)
    axes[1, 1].set_xlabel('Magnetic Field (Oe)')
    axes[1, 1].set_ylabel('|ΔC|/C × 100%')
    axes[1, 1].set_title(f'Relative Difference at T≈{T[mid_temp_idx]:.3f}K')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print key stats
    print(f"Max relative difference: {np.max(rel_diff):.1f}%")
    print(f"Mean relative difference: {np.mean(rel_diff):.1f}%")
    print("Weighted averaging recommended for spherical coordinates.")

def plot_average(filename):
    C_3d, angles, fields = calculate_hc_from_esr(filename)
    #Compute averages
    C_avg,_ = average_c_over_theta(C_3d, angles)

    print("C_avg shape:", C_avg.shape)  # e.g., (n_fields, 100)

    # Plot averaged results
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(1, 0, len(fields)))

    for j, field in enumerate(fields):
        plt.plot(T, C_avg[j, :], label=f'B = {field:.1f} Oe',
                 color=colors[j], linewidth=2)

    plt.title('Average Specific Heat (over all θ)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Specific Heat C (J/mol·K)')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_all_C_vs_T_by_theta(C_3d, angles, fields, T):
    """
    Create subplots for each theta angle showing all C vs T curves
    colored by magnetic field with shared colorbar.

    Parameters:
    -----------
    C_3d : ndarray (n_angles, n_fields, n_temps)
    angles, fields, T : coordinate arrays
    """
    import matplotlib.pyplot as plt

    n_angles = len(angles)
    n_fields = len(fields)

    # Create figure with subplots (5 rows if many angles, adjust as needed)
    n_cols = min(5, n_angles)
    n_rows = (n_angles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             sharex=True, sharey=True)
    if n_angles == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    norm = Normalize(vmin=fields.min(), vmax=fields.max())
    cmap = plt.cm.viridis

    for i, theta in enumerate(angles):
        ax = axes[i]

        # Plot all field curves for this theta
        for j, field in enumerate(fields):
            color = cmap(norm(field))
            ax.plot(T, C_3d[i, j, :], color=color, linewidth=1.5, alpha=0.8)

        ax.set_title(f'θ = {theta}°', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlabel('Temperatur (K)')
        ax.set_ylabel('Spezifische Wärme (J/mol·K)')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_angles, len(axes)):
        axes[i].set_visible(False)

    # Add shared colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[0], orientation='vertical',
                        label='Feld (Oe)', pad=0.02, shrink=0.8)
    cbar.set_label('Feld (Oe)', fontsize=12, fontweight='bold')

    plt.suptitle('Spezifische Wärme c vs T für alle (θ,B)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_all_C_and_dual_averaging(C_3d, angles, fields, T):
    """
    Two subplots:
    1. Left: All individual C(θ,B,T) curves organized by theta
    2. Right: Uniform vs sin(θ)-weighted averaging comparison

    Parameters:
    -----------
    C_3d : ndarray (n_angles, n_fields, n_temps)
    angles, fields, T : coordinate arrays
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    # Compute both averages
    C_avg_uniform, _ = average_c_over_theta(C_3d, angles)  # Angle-independent
    C_avg_weighted, _, weights = average_C_over_theta_weighted(C_3d, angles, 'sin_theta')

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    # LEFT: All individual curves by theta (one subplot per theta, but stacked)
    n_angles = len(angles)
    cmap = plt.cm.viridis_r  # INVERTED: _r suffix reverses colormap
    norm = Normalize(vmin=fields.min(), vmax=fields.max())

    # Plot all curves for each theta on left subplot
    for i, theta in enumerate(angles):
        for j, field in enumerate(fields):
            color = cmap(norm(field))
            alpha = 0.6 if n_angles > 3 else 0.8  # Adjust transparency for crowded plots
            ax_left.plot(T, C_3d[i, j, :], color=color, linewidth=1, alpha=alpha)

    # Add theta labels as legend entries for key angles
    theta_labels = [f'θ={theta}°' for theta in angles[::max(1, len(angles) // 5)]]
    ax_left.plot([], [], color='gray', linestyle='-', linewidth=2, label='All θ')
    ax_left.legend(['All θ curves'], loc='upper right')

    ax_left.set_xscale('log')
    ax_left.set_xlabel('Temperatur (K)')
    ax_left.set_ylabel('Spezifische Wärme C (J/mol·K)')
    ax_left.set_title('Alle C(θ,B,T) kurven')
    ax_left.grid(True, alpha=0.3)

    # Colorbar for fields (shared) - also inverted
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax_left, orientation='vertical', pad=0.02)
    cbar.set_label('Feld (Oe)', fontsize=12, fontweight='bold')

    # RIGHT: Dual averaging comparison - also inverted colors
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(fields)))  # INVERTED colormap
    fields_to_plot = [0, len(fields) // 8, len(fields) // 4, 2 * len(fields) // 4, 3 * len(fields) // 4, -1]  # 5 representative fields

    for idx, field_idx in enumerate(fields_to_plot):
        field_val = fields[field_idx]
        color = colors[field_idx]

        # Uniform average (solid)
        ax_right.plot(T, C_avg_uniform[field_idx], color=color, linewidth=2,
                      label=f'Ø B={field_val}')
        # Weighted average (dashed)
        ax_right.plot(T, C_avg_weighted[field_idx], color=color, linewidth=2,
                      linestyle='--', label=f'sin(θ) B={field_val}')

    ax_right.set_xscale('log')
    ax_right.set_xlabel('Temperatur (K)')
    ax_right.set_ylabel('Spezifische Wärme c (J/mol·K)')
    ax_right.set_title('Mittelwert (Linie) vs Mittelwert * sinus(θ) (Striche)')
    ax_right.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_right.grid(True, alpha=0.3)

    plt.suptitle('Spezifische Wärme: Einzelne Kurven vs zwei Winkel-Durchschnitte',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Print statistics
    diff = C_avg_weighted - C_avg_uniform
    rel_diff = 100 * np.abs(diff) / (C_avg_uniform + 1e-10)
    print(f"Max ΔC: {np.max(np.abs(diff)):.3f} J/mol·K")
    print(f"Max relative difference: {np.max(rel_diff):.1f}%")
    print(f"Shape C_avg_uniform: {C_avg_uniform.shape}")


def plot_c_averaging(C_3d, angles, fields, T, fields_to_plot = None):
    """
    plot: sin(θ)-weighted averaging

    Parameters:
    -----------
    C_3d : ndarray (n_angles, n_fields, n_temps)
    angles, fields, T : coordinate arrays
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np

    if fields_to_plot is None:
        fields_to_plot = fields

    # Compute weighted average
    C_avg_weighted, _, weights = average_C_over_theta_weighted(C_3d, angles, 'sin_theta')

    fig = plt.figure(figsize=(8, 6), dpi=80)

    # RIGHT: Dual averaging comparison - also inverted colors
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(fields_to_plot)))  # INVERTED colormap
    #fields_to_plot = [0, len(fields) // 3, 2 * len(fields) // 3, -1]  # 4 representative fields

    for idx, field in enumerate(fields_to_plot):
        field_val = field#fields[field_idx]
        color = colors[idx]

        # Weighted average (dashed)
        plt.plot(T, C_avg_weighted[np.where(fields == field)[0][0]], color=color, linewidth=2,
                      label=f'sin(θ) B={field_val}')

    plt.xscale('log')
    plt.xlabel('Temperatur (K)')
    plt.ylabel('Spezifische Wärme c (J/mol·K)')
    plt.title('Mittelwert (Linie) vs Mittelwert * sinus(θ)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.suptitle('Spezifische Wärme: Kugelgewichtet',
                 fontsize=14, fontweight='bold')
    #plt.tight_layout()
    plt.show()


def plot_all_C_vs_T_by_offset_theta(C_3d, angles, fields, T, angle_offsets):
    """
    Create subplots for each offset angle showing all C vs T curves
    colored by magnetic field with shared colorbar.
    All orientations are averaged but shifted by the offset.

    Parameters:
    -----------
    C_3d : ndarray (n_angles, n_fields, n_temps)
    angles, fields, T : coordinate arrays
    angle_offsets : offsets for angles
    """
    n_angles = len(angles)
    n_fields = len(fields)
    n_offsets = len(angle_offsets)

    # Create figure with subplots (5 rows if many angles, adjust as needed)
    n_cols = min(5, n_offsets)
    n_rows = (n_angles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                             sharex=True, sharey=True)
    if n_angles == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    norm = Normalize(vmin=fields.min(), vmax=fields.max())
    cmap = plt.cm.viridis_r

    for i, theta in enumerate(angle_offsets):
        ax = axes[i]
        C_avg_weighted, _, weights = average_C_over_theta_weighted(C_3d, angles, 'sin_theta', theta)

        # Plot all field curves for this theta
        for j, field in enumerate(fields):
            color = cmap(norm(field))
            ax.plot(T, C_avg_weighted[j], color=color, linewidth=1.5, alpha=0.8)

        ax.set_title(f'offset = {theta}°', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlabel('Temperatur (K)')
        ax.set_ylabel('Spezifische Wärme (J/mol·K)')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_angles, len(axes)):
        axes[i].set_visible(False)

    # Add shared colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[0], orientation='vertical',
                        label='Feld (Oe)', pad=0.02, shrink=0.8)
    cbar.set_label('Feld (Oe)', fontsize=12, fontweight='bold')

    plt.suptitle('Spezifische Wärme c vs T, Kugelgewichtet mit Vorzugsrichtung',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def load_hc_xlsx(filename):
    hc_data = pd.read_excel(filename, header=None, names=['T','C'])


filename = "E.csv"
#filename = "2GHz4_g2.csv"

#plot_energy_levels(filename, True)
C_3d, angles, fields = calculate_hc_from_esr(filename, T)

#plot_all_C_vs_T_by_theta(C_3d, angles, fields, T)

plot_all_C_and_dual_averaging(C_3d, angles, fields, T)
#plot_c_averaging(C_3d, angles, fields, T, [0, 200, 400, 600, 2000])

plot_c_averaging(C_3d, angles, fields, T)

plot_all_C_vs_T_by_offset_theta(C_3d, angles, fields, T, angles)
