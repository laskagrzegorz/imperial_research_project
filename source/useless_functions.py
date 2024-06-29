import numpy as np
import matplotlib.pyplot as plt


def plot_sdf_with_theoretical_plt(frequencies, empirical_sdf, theoretical_sdf, log_scale=False,
                              title='Spectral Matrix with Theoretical Spectrum'):
    """
    Plot the SDF for each pair of dimensions with theoretical spectrum.

    Parameters:
    frequencies (np.ndarray): Array of frequency values.
    empirical_sdf (np.ndarray): Empirical spectral density function matrix.
    theoretical_sdf (np.ndarray): Theoretical spectral density function matrix.
    title (str): Title of the plot.
    log_scale (bool): If True, use logarithmic scale for the y-axis.
    """
    # Calculate the length of the spectral density functions
    length = empirical_sdf.shape[2]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        for j in range(3):
            if i != j:
                axes[i, j].plot(frequencies, np.abs(empirical_sdf[i, j, :]), label='Empirical')
                axes[i, j].plot(frequencies, np.abs(theoretical_sdf[i, j, :]), label='Theoretical', linestyle='--')
                axes[i, j].set_title(f'S_{i + 1}{j + 1}')
            else:
                axes[i, j].plot(frequencies, np.abs(empirical_sdf[i, j, :]), label='Empirical')
                axes[i, j].plot(frequencies, np.abs(theoretical_sdf[i, j, :]), label='Theoretical', linestyle='--')
                axes[i, j].set_title(f'S_{i + 1}{i + 1}')

            if log_scale:
                axes[i, j].set_yscale('log')

            if i == 2:
                axes[i, j].set_xlabel('Frequency (Hz)')
            if j == 0:
                axes[i, j].set_ylabel('Spectral Density')

            axes[i, j].grid(True)
            axes[i, j].legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()