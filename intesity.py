import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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

    # Prepare plasma colormap for angles
    angles = np.array(sorted(summed_curves.keys()))
    cmap = plt.get_cmap('plasma')           # get plasma colormap [web:10]
    colors = cmap(np.linspace(0, 1, len(angles)))  # sample distinct colors [web:5]

    # Map angle -> color
    angle_to_color = {ang: col for ang, col in zip(angles, colors)}

    # First plot: summed intensity over magnetfield for each angle
    plt.figure(figsize=(8, 6))
    for angle, intensity_sum in summed_curves.items():
        plt.plot(
            magnetfield_grid,
            intensity_sum,
            label=f'angle = {angle}',
            color=angle_to_color[angle],
        )

    plt.xlabel('Magnetfield')
    plt.ylabel('Summed intensity')
    plt.title('Summed Gaussian curves per angle')
    plt.legend()
    plt.tight_layout()

    # Second plot: derivative of each summed curve with respect to magnetfield
    plt.figure(figsize=(8, 6))
    for angle, intensity_sum in summed_curves.items():
        derivative = np.gradient(intensity_sum, magnetfield_grid)
        plt.plot(
            magnetfield_grid,
            derivative,
            label=f'd/dB angle = {angle}',
            color=angle_to_color[angle],  # same color mapping as first plot
        )

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


def draw_weighted_averaged_spectra_sigma_scan():
    # Load CSV file
    df = pd.read_csv('I.csv', header=None, names=['angle', 'intensity', 'magnetfield'])

    # Define Gaussian function
    def gaussian(x, mu, amp, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Sigmas to scan (6 values)
    sigmas = [20, 50, 100, 150, 200, 300]

    # Prepare common magnetfield grid wide enough for all sigmas
    sigma_max = max(sigmas)
    mf_min = df['magnetfield'].min() - 3 * sigma_max
    mf_max = df['magnetfield'].max() + 3 * sigma_max
    magnetfield_grid = np.linspace(mf_min, mf_max, 10000)

    # Spherical symmetry weighting: weight ~ sin(angle)
    angle_rad = np.deg2rad(df['angle'])
    weights = np.sin(angle_rad)

    # Viridis colormap
    cmap = get_cmap('plasma')
    colors = [cmap(i / (len(sigmas) - 1)) for i in range(len(sigmas))]

    # Figure for averaged curves for all sigmas
    plt.figure(figsize=(8, 6))
    for sigma, color in zip(sigmas, colors):
        weighted_sum = np.zeros_like(magnetfield_grid)
        weight_norm = 0.0
        for (_, row), w in zip(df.iterrows(), weights):
            g = gaussian(magnetfield_grid, mu=row['magnetfield'],
                         amp=row['intensity'], sigma=sigma)
            weighted_sum += w * g
            weight_norm += w * row['intensity']
        averaged_curve = weighted_sum / weight_norm if weight_norm != 0 else weighted_sum
        plt.plot(magnetfield_grid, averaged_curve, label=f'sigma = {sigma}', color=color)

    plt.xlabel('Magnetfield')
    plt.ylabel('Spherically weighted average intensity')
    plt.title('Spherically weighted average for different sigma')
    plt.legend()
    plt.tight_layout()

    # Figure for derivatives of averaged curves for all sigmas
    plt.figure(figsize=(8, 6))
    for sigma, color in zip(sigmas, colors):
        weighted_sum = np.zeros_like(magnetfield_grid)
        weight_norm = 0.0
        for (_, row), w in zip(df.iterrows(), weights):
            g = gaussian(magnetfield_grid, mu=row['magnetfield'],
                         amp=row['intensity'], sigma=sigma)
            weighted_sum += w * g
            weight_norm += w * row['intensity']
        averaged_curve = weighted_sum / weight_norm if weight_norm != 0 else weighted_sum
        derivative = np.gradient(averaged_curve, magnetfield_grid)
        plt.plot(magnetfield_grid, derivative, label=f'sigma = {sigma}', color=color)

    plt.xlabel('Magnetfield')
    plt.ylabel('Derivative of spherically weighted average')
    plt.title('Derivative of spherically weighted average for different sigma')
    plt.legend()
    plt.tight_layout()

    plt.show()


def draw_weighted_averaged_spectra_sigma_scan_with_offset(delta_theta_deg=70.53):
    """Draw spectra for two angular datasets offset by delta_theta and average their simulated spectra.

    Parameters
    ----------
    delta_theta_deg : float
        Angular offset in degrees to be added to the angles of the second dataset.
    """

    # Load primary CSV file (original angles)
    df1 = pd.read_csv('I.csv', header=None, names=['angle', 'intensity', 'magnetfield'])

    # Create second dataset by copying df1 and offsetting the angle by delta_theta
    # If you have a real second file, load it instead and then apply the offset to its angle column.
    df2 = df1.copy()
    df2['angle'] = df2['angle'] + delta_theta_deg

    # Define Gaussian function
    def gaussian(x, mu, amp, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Sigmas to scan (6 values)
    sigmas = [10, 20, 50, 75, 100, 150]

    # Prepare common magnetfield grid wide enough for all sigmas
    sigma_max = max(sigmas)
    mf_min = min(df1['magnetfield'].min(), df2['magnetfield'].min()) - 3 * sigma_max
    mf_max = max(df1['magnetfield'].max(), df2['magnetfield'].max()) + 3 * sigma_max
    magnetfield_grid = np.linspace(mf_min, mf_max, 10000)

    # Spherical symmetry weighting: weight ~ sin(angle), angles in degrees
    angle_rad_1 = np.deg2rad(df1['angle'])
    weights_1 = np.sin(angle_rad_1)

    angle_rad_2 = np.deg2rad(df2['angle'])
    weights_2 = np.sin(angle_rad_2)

    # Viridis colormap
    cmap = get_cmap('plasma')
    colors = [cmap(i / (len(sigmas) - 1)) for i in range(len(sigmas))]

    # Figure for averaged curves (average of spectra from both datasets)
    plt.figure(figsize=(8, 6))
    for sigma, color in zip(sigmas, colors):
        # Dataset 1
        weighted_sum_1 = np.zeros_like(magnetfield_grid)
        weight_norm_1 = 0.0
        for (_, row), w in zip(df1.iterrows(), weights_1):
            g = gaussian(magnetfield_grid,
                         mu=row['magnetfield'],
                         amp=row['intensity'],
                         sigma=sigma)
            weighted_sum_1 += w * g
            weight_norm_1 += w * row['intensity']
        averaged_1 = weighted_sum_1 / weight_norm_1 if weight_norm_1 != 0 else weighted_sum_1

        # Dataset 2 (angle offset by delta_theta_deg)
        weighted_sum_2 = np.zeros_like(magnetfield_grid)
        weight_norm_2 = 0.0
        for (_, row), w in zip(df2.iterrows(), weights_2):
            g = gaussian(magnetfield_grid,
                         mu=row['magnetfield'],
                         amp=row['intensity'],
                         sigma=sigma)
            weighted_sum_2 += w * g
            weight_norm_2 += w * row['intensity']
        averaged_2 = weighted_sum_2 / weight_norm_2 if weight_norm_2 != 0 else weighted_sum_2

        # Average of the two simulated spectra
        averaged_curve = 0.5 * (averaged_1 + averaged_2)

        plt.plot(magnetfield_grid, averaged_curve, label=f'sigma = {sigma}', color=color)

    plt.xlabel('Magnetfield')
    plt.ylabel('Spherically weighted average intensity (2 datasets)')
    plt.title('Average of two offset angular datasets for different sigma')
    plt.legend()
    plt.tight_layout()

    # Figure for derivatives of the averaged curves
    plt.figure(figsize=(8, 6))
    for sigma, color in zip(sigmas, colors):
        # Dataset 1
        weighted_sum_1 = np.zeros_like(magnetfield_grid)
        weight_norm_1 = 0.0
        for (_, row), w in zip(df1.iterrows(), weights_1):
            g = gaussian(magnetfield_grid,
                         mu=row['magnetfield'],
                         amp=row['intensity'],
                         sigma=sigma)
            weighted_sum_1 += w * g
            weight_norm_1 += w * row['intensity']
        averaged_1 = weighted_sum_1 / weight_norm_1 if weight_norm_1 != 0 else weighted_sum_1

        # Dataset 2
        weighted_sum_2 = np.zeros_like(magnetfield_grid)
        weight_norm_2 = 0.0
        for (_, row), w in zip(df2.iterrows(), weights_2):
            g = gaussian(magnetfield_grid,
                         mu=row['magnetfield'],
                         amp=row['intensity'],
                         sigma=sigma)
            weighted_sum_2 += w * g
            weight_norm_2 += w * row['intensity']
        averaged_2 = weighted_sum_2 / weight_norm_2 if weight_norm_2 != 0 else weighted_sum_2

        # Average of the two simulated spectra
        averaged_curve = 0.5 * (averaged_1 + averaged_2)

        # Numerical derivative with respect to magnetfield
        derivative = np.gradient(averaged_curve, magnetfield_grid)

        plt.plot(magnetfield_grid, derivative, label=f'sigma = {sigma}', color=color)

    plt.xlabel('Magnetfield')
    plt.ylabel('Derivative of averaged intensity (2 datasets)')
    plt.title('Derivative of averaged spectra of two offset datasets for different sigma')
    plt.legend()
    plt.tight_layout()

    plt.show()



draw_spectra_diff()
draw_avgeraged_spectrum()
#draw_weighted_averaged_spectra_sigma_scan()
draw_weighted_averaged_spectra_sigma_scan_with_offset()