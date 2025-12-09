import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_spectra():
    # Load CSV file
    # Assumes I.csv is in the current working directory
    # and has columns: angle, intensity, magnetfield

    df = pd.read_csv('I.csv', header=None, names=['angle','intensity','magnetfield'])

    # Define Gaussian function
    def gaussian(x, mu, amp, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Choose a width (sigma) for the Gaussian in magnetfield units
    sigma = 100  # adjust as appropriate for your data

    # Prepare magnetfield grid for plotting
    mf_min = df['magnetfield'].min() - 3 * sigma
    mf_max = df['magnetfield'].max() + 3 * sigma
    magnetfield_grid = np.linspace(mf_min, mf_max, 10000)

    # Group by angle and sum Gaussians per angle
    summed_curves = {}
    for angle, group in df.groupby('angle'):
        total_intensity = np.zeros_like(magnetfield_grid)
        for _, row in group.iterrows():
            total_intensity += gaussian(
                magnetfield_grid,
                mu=row['magnetfield'],
                amp=row['intensity'],
                sigma=sigma,
            )
        summed_curves[angle] = total_intensity

    # Plot summed intensity over magnetfield for each angle
    plt.figure(figsize=(8, 6))
    for angle, intensity_sum in summed_curves.items():
        plt.plot(magnetfield_grid, intensity_sum, label=f'angle = {angle}')

    plt.xlabel('Magnetfield')
    plt.ylabel('Summed intensity')
    plt.title('Summed Gaussian curves per angle')
    plt.legend()
    plt.tight_layout()
    plt.show()


def draw_spectra_diff():
    # Load CSV file
    # Assumes I.csv is in the current working directory
    # and has columns: angle, intensity, magnetfield

    df = pd.read_csv('I.csv', header=None, names=['angle', 'intensity', 'magnetfield'])

    # Define Gaussian function
    def gaussian(x, mu, amp, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Choose a width (sigma) for the Gaussian in magnetfield units
    sigma = 100  # adjust as appropriate for your data

    # Prepare magnetfield grid for plotting
    mf_min = df['magnetfield'].min() - 3 * sigma
    mf_max = df['magnetfield'].max() + 3 * sigma
    magnetfield_grid = np.linspace(mf_min, mf_max, 10000)

    # Group by angle and sum Gaussians per angle
    summed_curves = {}
    for angle, group in df.groupby('angle'):
        total_intensity = np.zeros_like(magnetfield_grid)
        for _, row in group.iterrows():
            total_intensity += gaussian(
                magnetfield_grid,
                mu=row['magnetfield'],
                amp=row['intensity'],
                sigma=sigma,
            )
        summed_curves[angle] = total_intensity

    # First plot: summed intensity over magnetfield for each angle
    plt.figure(figsize=(8, 6))
    for angle, intensity_sum in summed_curves.items():
        plt.plot(magnetfield_grid, intensity_sum, label=f'angle = {angle}')

    plt.xlabel('Magnetfield')
    plt.ylabel('Summed intensity')
    plt.title('Summed Gaussian curves per angle')
    plt.legend()
    plt.tight_layout()

    # Second plot: derivative of each summed curve with respect to magnetfield
    plt.figure(figsize=(8, 6))
    for angle, intensity_sum in summed_curves.items():
        derivative = np.gradient(intensity_sum, magnetfield_grid)
        plt.plot(magnetfield_grid, derivative, label=f'd/dB angle = {angle}')

    plt.xlabel('Magnetfield')
    plt.ylabel('Derivative of summed intensity')
    plt.title('Derivative of summed Gaussian curves per angle')
    plt.legend()
    plt.tight_layout()

    plt.show()


def draw_avgeraged_spectrum():
    # Load CSV file without header
    # Columns by index: 0=angle (in degrees), 1=intensity, 2=magnetfield
    df = pd.read_csv('I.csv', header=None, names=['angle','intensity','magnetfield'])

    # Define Gaussian function
    def gaussian(x, mu, amp, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Choose a width (sigma) for the Gaussian in magnetfield units
    sigma = 100  # adjust as appropriate for your data

    # Prepare magnetfield grid for plotting
    mf_min = df['magnetfield'].min() - 3 * sigma
    mf_max = df['magnetfield'].max() + 3 * sigma
    magnetfield_grid = np.linspace(mf_min, mf_max, 10000)

    # Convert angle to radians for spherical weighting
    df['theta_rad'] = np.deg2rad(df['angle'])

    # Spherical symmetry weight ~ sin(theta)
    df['weight'] = np.sin(df['theta_rad'])

    # For each row, compute its Gaussian contribution on the grid, weighted by sin(theta)
    weighted_sum = np.zeros_like(magnetfield_grid)
    weight_norm = 0.0

    for _, row in df.iterrows():
        g = gaussian(magnetfield_grid, mu=row['magnetfield'], amp=row['intensity'], sigma=sigma)
        w = row['weight']
        weighted_sum += w * g
        # include intensity in the normalization so stronger peaks count more
        weight_norm += w * row['intensity']

    # Normalize by total weight to get an average
    if weight_norm != 0:
        averaged_curve = weighted_sum / weight_norm
    else:
        averaged_curve = weighted_sum

    # Plot spherically weighted average intensity over magnetfield
    plt.figure(figsize=(8, 6))
    plt.plot(magnetfield_grid, averaged_curve, label='Spherically weighted average')
    plt.xlabel('Magnetfield')
    plt.ylabel('Average summed intensity')
    plt.title('Spherically weighted average of Gaussian curves over all angles')
    plt.legend()
    plt.tight_layout()
    plt.show()

    derivative = np.gradient(averaged_curve, magnetfield_grid)

    plt.figure(figsize=(8, 6))
    plt.plot(magnetfield_grid, derivative, label='d(Average)/d(Magnetfield)')
    plt.xlabel('Magnetfield')
    plt.ylabel('Derivative of average intensity')
    plt.title('Derivative of spherically weighted average vs. magnetfield')
    plt.legend()
    plt.tight_layout()

    plt.show()

draw_spectra_diff()
draw_avgeraged_spectrum()