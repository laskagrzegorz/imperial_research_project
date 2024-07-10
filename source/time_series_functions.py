import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from plotly.subplots import make_subplots
import scipy.signal


### Functions ###

def generate_latex_equation(coeff_matrix):
    """
    Generate and display LaTeX for a given 3x3, 5x5, or 7x7 coefficient matrix.

    Parameters:
    coeff_matrix (list of lists or numpy array): 3x3, 5x5, or 7x7 matrix of coefficients.
    """
    var_p = len(coeff_matrix)
    coeff_matrix = coeff_matrix[-1]
    # Determine the size of the matrix
    size = len(coeff_matrix)
    assert size in [3, 5, 7], "Matrix must be 3x3, 5x5, or 7x7"

    if size == 3:
        # Extract coefficients for 3x3 matrix
        a11, a12, a13 = coeff_matrix[0]
        a21, a22, a23 = coeff_matrix[1]
        a31, a32, a33 = coeff_matrix[2]

        # Create the LaTeX string using f-string formatting for 3x3 matrix
        latex_string = rf'''
        \left(\begin{{array}}{{l}}
        x_{{1, t}} \\
        x_{{2, t}} \\
        x_{{3, t}}
        \end{{array}}\right)=\left(\begin{{array}}{{ccc}}
        {a11} & {a12} & {a13} \\
        {a21} & {a22} & {a23} \\
        {a31} & {a32} & {a33}
        \end{{array}}\right)\left(\begin{{array}}{{l}}
        x_{{1, t-{var_p} }} \\
        x_{{2, t-{var_p} }} \\
        x_{{3, t-{var_p} }}
        \end{{array}}\right)+\left(\begin{{array}}{{l}}
        \varepsilon_{{1, t}} \\
        \varepsilon_{{2, t}} \\
        \varepsilon_{{3, t}}
        \end{{array}}\right)
        '''
    elif size == 5:
        # Extract coefficients for 5x5 matrix
        a11, a12, a13, a14, a15 = coeff_matrix[0]
        a21, a22, a23, a24, a25 = coeff_matrix[1]
        a31, a32, a33, a34, a35 = coeff_matrix[2]
        a41, a42, a43, a44, a45 = coeff_matrix[3]
        a51, a52, a53, a54, a55 = coeff_matrix[4]

        # Create the LaTeX string using f-string formatting for 5x5 matrix
        latex_string = rf'''
        \left(\begin{{array}}{{l}}
        x_{{1, t}} \\
        x_{{2, t}} \\
        x_{{3, t}} \\
        x_{{4, t}} \\
        x_{{5, t}}
        \end{{array}}\right)=\left(\begin{{array}}{{ccccc}}
        {a11} & {a12} & {a13} & {a14} & {a15} \\
        {a21} & {a22} & {a23} & {a24} & {a25} \\
        {a31} & {a32} & {a33} & {a34} & {a35} \\
        {a41} & {a42} & {a43} & {a44} & {a45} \\
        {a51} & {a52} & {a53} & {a54} & {a55}
        \end{{array}}\right)\left(\begin{{array}}{{l}}
        x_{{1, t-{var_p} }} \\
        x_{{2, t-{var_p} }} \\
        x_{{3, t-{var_p} }} \\
        x_{{4, t-{var_p} }} \\
        x_{{5, t-{var_p} }}
        \end{{array}}\right)+\left(\begin{{array}}{{l}}
        \varepsilon_{{1, t}} \\
        \varepsilon_{{2, t}} \\
        \varepsilon_{{3, t}} \\
        \varepsilon_{{4, t}} \\
        \varepsilon_{{5, t}}
        \end{{array}}\right)
        '''
    elif size == 7:
        # Extract coefficients for 7x7 matrix
        a11, a12, a13, a14, a15, a16, a17 = coeff_matrix[0]
        a21, a22, a23, a24, a25, a26, a27 = coeff_matrix[1]
        a31, a32, a33, a34, a35, a36, a37 = coeff_matrix[2]
        a41, a42, a43, a44, a45, a46, a47 = coeff_matrix[3]
        a51, a52, a53, a54, a55, a56, a57 = coeff_matrix[4]
        a61, a62, a63, a64, a65, a66, a67 = coeff_matrix[5]
        a71, a72, a73, a74, a75, a76, a77 = coeff_matrix[6]

        # Create the LaTeX string using f-string formatting for 7x7 matrix
        latex_string = rf'''
        \left(\begin{{array}}{{l}}
        x_{{1, t}} \\
        x_{{2, t}} \\
        x_{{3, t}} \\
        x_{{4, t}} \\
        x_{{5, t}} \\
        x_{{6, t}} \\
        x_{{7, t}}
        \end{{array}}\right)=\left(\begin{{array}}{{ccccccc}}
        {a11} & {a12} & {a13} & {a14} & {a15} & {a16} & {a17} \\
        {a21} & {a22} & {a23} & {a24} & {a25} & {a26} & {a27} \\
        {a31} & {a32} & {a33} & {a34} & {a35} & {a36} & {a37} \\
        {a41} & {a42} & {a43} & {a44} & {a45} & {a46} & {a47} \\
        {a51} & {a52} & {a53} & {a54} & {a55} & {a56} & {a57} \\
        {a61} & {a62} & {a63} & {a64} & {a65} & {a66} & {a67} \\
        {a71} & {a72} & {a73} & {a74} & {a75} & {a76} & {a77}
        \end{{array}}\right)\left(\begin{{array}}{{l}}
        x_{{1, t-{var_p} }} \\
        x_{{2, t-{var_p} }} \\
        x_{{3, t-{var_p} }} \\
        x_{{4, t-{var_p} }} \\
        x_{{5, t-{var_p} }} \\
        x_{{6, t-{var_p} }} \\
        x_{{7, t-{var_p} }}
        \end{{array}}\right)+\left(\begin{{array}}{{l}}
        \varepsilon_{{1, t}} \\
        \varepsilon_{{2, t}} \\
        \varepsilon_{{3, t}} \\
        \varepsilon_{{4, t}} \\
        \varepsilon_{{5, t}} \\
        \varepsilon_{{6, t}} \\
        \varepsilon_{{7, t}}
        \end{{array}}\right)
        '''

    # Display the LaTeX in Streamlit
    st.latex(latex_string)


def make_var_1_to_var_p(All_A_list, p):
    All_A_list_var_p = [sublist.copy() for sublist in All_A_list]
    for sublist in All_A_list_var_p:
        for _ in range(p-1):
            sublist.insert(0, np.zeros_like(sublist[0]))

    return All_A_list_var_p


# Initial functions
def generate_var_process(A, num_samples, burn_in=100, seed=None):
    """
    Generate a sample of a VAR(p) process.

    Parameters:
    A (list of np.ndarray): List of coefficient matrices for the VAR process.
    num_samples (int): Number of samples to generate.
    burn_in (int): Number of initial samples to discard to reduce the effect of initial values.
    seed (int or None): Random seed for reproducibility.

    Returns:
    np.ndarray: Generated sample of the VAR process.
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(A)  # Order of the VAR process
    k = A[0].shape[0]  # Number of dimensions

    total_samples = num_samples + burn_in

    # Initialize the time series array
    X = np.zeros((k, total_samples))

    # Generate the white noise (innovation) terms
    epsilon = np.random.normal(size=(k, total_samples))

    # Simulate the VAR process
    for t in range(p, total_samples):
        X[:, t] = epsilon[:, t]
        for i in range(1, p + 1):
            X[:, t] += A[i - 1] @ X[:, t - i]

    # Discard the burn-in period and return the remaining samples
    return X[:, burn_in:]


def plot_time_series(x):
    """
    Plot the time series on one graph using Plotly.

    Parameters:
    x (numpy.ndarray): The generated time series.

    Returns:
    go.Figure: The Plotly figure object.
    """
    T = x.shape[1]
    time = np.arange(T)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time, y=x[0], mode='lines', name='x1_t'))
    fig.add_trace(go.Scatter(x=time, y=x[1], mode='lines', name='x2_t', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=time, y=x[2], mode='lines', name='x3_t', line=dict(color='green')))

    fig.update_layout(
        # title='Time Series',
        xaxis_title='Time',
        yaxis_title='Values',
        legend=dict(title='Series'),
        template='plotly_white'
    )



    return fig


def is_var_process_stationary(coef_matrices):
    """
    Check if a VAR(p) process is stationary given its coefficient matrices.

    Parameters:
    - coef_matrices: List of numpy arrays, where each array represents the coefficients
                     for each lag in the VAR(p) model.

    Returns:
    - Boolean indicating whether the VAR(p) process is stationary.
    """
    p = len(coef_matrices)  # Order of the VAR process
    n = coef_matrices[0].shape[0]  # Dimension of the VAR process

    # Initialize the companion matrix
    companion_matrix = np.zeros((n * p, n * p))

    # Fill the upper part of the companion matrix with the negated coefficients
    for i in range(p):
        companion_matrix[:n, n * i:n * (i + 1)] = -coef_matrices[i]

    # Fill the lower part of the companion matrix with identity matrices
    for i in range(1, p):
        companion_matrix[n * i:n * (i + 1), n * (i - 1):n * i] = np.eye(n)

    # Compute the eigenvalues of the companion matrix
    eigenvalues = np.linalg.eigvals(companion_matrix)

    print('Companion matrix: ')
    print(companion_matrix)

    print("eigenvalues")
    print(eigenvalues)

    # Check if all eigenvalues are outside the unit circle
    return np.all(np.abs(eigenvalues) < 1)


def check_series(A_list):
    check = is_var_process_stationary(A_list)
    if check:
        st.success(
            body="Process is stationary",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            body="Process is not stationary",
            icon="⚠️",
        )


def true_sdf_var_process(A, Sigma, num_points=512):
    """
    Calculate the spectral density function of a VAR process.

    Parameters:
    A (list of np.ndarray): List of coefficient matrices for the VAR process.
    Sigma (np.ndarray): Covariance matrix of the white noise.
    num_points (int): Number of frequency points to calculate the SDF.

    Returns:
    frequencies (np.ndarray): Frequencies at which the SDF is calculated.
    sdf (np.ndarray): Spectral density function values.
    """
    p = len(A)  # Order of the VAR process
    k = A[0].shape[0]  # Number of dimensions
    frequencies = np.linspace(0, 0.5, num_points)  # Normalized frequency (0 to 0.5 corresponds to Nyquist frequency)
    sdf = np.zeros((k, k, num_points), dtype=complex)

    I_k = np.eye(k, dtype=np.complex_)  # Identity matrix of size k

    for freq_idx, freq in enumerate(frequencies):
        exp_sum = I_k.copy()
        for matrix_idx in range(1, p + 1):
            omega = -2j * np.pi * matrix_idx * freq
            exp_sum -= A[matrix_idx - 1] * np.exp(omega)

        # Try computing the determinant
        try:
            det = np.linalg.det(exp_sum)
        except Exception as e:
            print(f"Error computing determinant: {e}")

        inv_exp_sum = np.linalg.inv(exp_sum)
        # TODO keep the T or not
        sdf[:, :, freq_idx] = (1 / (2 * np.pi)) * inv_exp_sum @ Sigma @ np.linalg.inv(np.conj(exp_sum.T))

    return frequencies, sdf


def plot_sdf_with_theoretical(frequencies, empirical_sdf_list, empirical_names, theoretical_sdf, log_scale=False,
                              function=np.abs, different_frequencies=None):
    """
    Plot the SDF for each pair of dimensions with theoretical spectrum using Plotly.

    Parameters:
    frequencies (np.ndarray): Array of frequency values.
    empirical_sdf_list (list of np.ndarray): List of empirical spectral density function matrices.
    empirical_names (list of str): List of names corresponding to each empirical SDF matrix.
    theoretical_sdf (np.ndarray): Theoretical spectral density function matrix.
    log_scale (bool): If True, use logarithmic scale for the y-axis.
    function (function): Function to apply to the SDF values (e.g., np.abs).
    different_frequencies (list of np.ndarray or None): List of frequency arrays corresponding to each empirical SDF.
                                                         If None, use the same frequencies array for all.
    """
    # Define a fixed color scheme
    colors = ['green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    theoretical_color = 'red'

    n_series = empirical_sdf_list[0].shape[0]

    # Create a subplot grid
    fig = make_subplots(rows=n_series, cols=n_series, subplot_titles=[f'S_{i}{j}' for i in range(n_series) for j in range(n_series)],
                        vertical_spacing=0.1, horizontal_spacing=0.1)

    # Add traces and settings for each subplot
    for i in range(n_series):
        for j in range(n_series):
            for idx, (empirical_sdf, name) in enumerate(zip(empirical_sdf_list, empirical_names)):
                freqs = frequencies if different_frequencies is None else different_frequencies[idx]
                fig.add_trace(go.Scatter(x=freqs, y=function(empirical_sdf[i, j, :]),
                                         mode='lines', name=f'{name} Empirical',
                                         line=dict(color=colors[idx % len(colors)]),
                                         legendgroup=f'{name} Empirical', showlegend=(i == 0 and j == 0)),
                              row=i + 1, col=j + 1)

            if theoretical_sdf is not None:
                fig.add_trace(go.Scatter(x=frequencies, y=function(theoretical_sdf[i, j, :]),
                                         mode='lines', name='Theoretical', line=dict(color=theoretical_color, dash='dash'),
                                         legendgroup='Theoretical', showlegend=(i == 0 and j == 0)),
                              row=i + 1, col=j + 1)

            # Update y-axis to log scale if required
            if log_scale:
                fig.update_yaxes(type="log", row=i + 1, col=j + 1)

            # Set axis titles
            if i == n_series - 1:
                fig.update_xaxes(title_text='Frequency (Hz)', row=i + 1, col=j + 1)
            if j == 0:
                fig.update_yaxes(title_text='Spectral Density', row=i + 1, col=j + 1)

    # Update layout with title and overall settings
    fig.update_layout(showlegend=True, height=800, width=800)
    return fig


def invert_spectral_matrix(spectral_matrix, noise_level=1e-10):
    """
    Invert the spectral matrix for each frequency, adding a small noise to avoid singularity.

    Parameters:
    spectral_matrix (numpy.ndarray): Spectral matrix for each frequency.
    noise_level (float): The standard deviation of the Gaussian noise to be added to avoid singularity.

    Returns:
    inv_spectral_matrix (numpy.ndarray): Inverse of the spectral matrix for each frequency.
    """
    num_series, _, N = spectral_matrix.shape

    # Initialize the inverse spectral matrix
    inv_spectral_matrix = np.zeros_like(spectral_matrix, dtype=complex)

    # Invert the spectral matrix for each frequency
    for k in range(N):
        # Add a small noise to the diagonal elements
        noise = np.eye(num_series) * noise_level
        inv_spectral_matrix[:, :, k] = np.linalg.inv(spectral_matrix[:, :, k] + noise)

    return inv_spectral_matrix

## Estimates ##

# Basic and cosine taper

def cosine_tapering_window(n, p):
    """
    Generate a cosine tapering window of length n with tapering parameter p.

    Parameters:
    - n (int): Length of the tapering window.
    - p (float): Tapering parameter, expressed as a fraction of the length of the window.

    Returns:
    - np.ndarray: Cosine tapering window of length n.
    """
    # Calculate the number of points to taper on each side
    taper_points = int(np.floor(p * n))

    half_taper_points = int(taper_points / 2)

    # Generate the tapering window
    taper = np.ones(n)
    taper[:half_taper_points] = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, half_taper_points + 1) / (taper_points + 1)))
    taper[-half_taper_points:] = taper[:half_taper_points][::-1]  # Symmetrically copy the tapering window

    # Normalize the tapering window
    taper /= np.sqrt(np.sum(taper ** 2))

    return taper


# Periodogram
def calculate_periodogram(X, taper=None, noise_level=1e-10):
    """
    Calculate the power spectral density matrix \hat{S}^{(P)}(f) for a given time series matrix X,
    with added noise to ensure the matrix is not singular.

    Parameters:
    X (numpy.ndarray): A 2D array where each row represents a time series.
    taper (numpy.ndarray): A 1D array to be multiplied element-wise to each time series.
    noise_level (float): The standard deviation of the Gaussian noise to be added.

    Returns:
    S_P (numpy.ndarray): The power spectral density matrix for each frequency.
    frequencies (numpy.ndarray): The frequency bins corresponding to the power spectral density.
    """
    N = X.shape[1]  # Number of time steps
    sqrt_N = np.sqrt(N)

    # Apply taper to each time series if provided
    if taper is not None:
        if taper.shape[0] != N:
            raise ValueError("The taper vector must have the same length as the time series.")
        X = X * taper
    else:
        X = X * 1 / sqrt_N

    # Compute the FFT for each time series
    W = np.fft.fft(X, axis=1)[:, :N // 2]

    # Add small Gaussian noise to W to avoid singularity
    # noise = np.random.normal(scale=noise_level, size=W.shape)
    # W += noise

    # Compute the frequencies
    frequencies = np.fft.fftfreq(N)[:N // 2]

    # Initialize the power spectral density matrix
    S_P = np.zeros((X.shape[0], X.shape[0], N // 2), dtype=np.complex128)

    # Calculate the power spectral density matrix for each frequency
    for i in range(N // 2):
        W_f = W[:, i]  # FFT values for the ith frequency
        S_P[:, :, i] = np.outer(W_f, np.conj(W_f).T)  # W(f) * W^H(f)

    return frequencies, S_P


def calculate_freq_avg_periodogram(x, m, f_minus_j_0=False):
    """
    Calculate the smoothed spectral density estimate using log-window with sinusoidal tapers.

    Parameters:
    x (np.ndarray): time series (num_series, num_series, num_samples).
    m (int): Number of tapers to use.

    Returns:
    w_k (np.ndarray): Sinusoidal taper weights of shape (m,).
    f_hat (np.ndarray): Smoothed spectral density estimate of shape (num_series, num_series, num_frequencies).
    """

    frequences, I = calculate_periodogram(x)
    _, _, num_frequencies = I.shape

    f_hat = np.zeros_like(I, dtype=complex)
    k_values = np.arange(-m // 2, m // 2 + 1)
    w_k = np.cos(np.pi * k_values / m)

    if f_minus_j_0:
        w_k[m // 2] = 0

    # Ensure weights sum to 1
    w_k /= np.sum(w_k)

    # First not complete
    for f in range(m // 2):
        w_k_temp = w_k[(m // 2 - f):]
        w_k_temp /= np.sum(w_k_temp)
        weighted_sum = np.sum(I[:, :, : len(w_k_temp)] * w_k_temp, axis=2)
        f_hat[:, :, f] = weighted_sum

    # Compete
    for f in range(num_frequencies - len(w_k) + 1):
        weighted_sum = np.sum(I[:, :, f:f + len(w_k)] * w_k, axis=2)
        f_hat[:, :, f + m // 2] = weighted_sum

    # Last not complete
    for f in range(m // 2):
        w_k_temp = w_k[:-(m // 2 - f)]
        w_k_temp /= np.sum(w_k_temp)

        weighted_sum = np.sum(I[:, :, -len(w_k_temp):] * w_k_temp, axis=2)
        f_hat[:, :, -f - 1] = weighted_sum

    return w_k, f_hat, frequences

# Sinusoidal Multi-Taper

def sinusoidal_tapers(num_samples, num_tapers):
    """
    Generate sinusoidal tapers.

    Parameters:
    num_samples (int): Number of time samples.
    num_tapers (int): Number of sinusoidal tapers.

    Returns:
    tapers (np.ndarray): Array of sinusoidal tapers with shape (num_tapers, num_samples).
    """

    if num_tapers >= num_samples:
        raise ValueError(f'More tapers than samples: \n T:{num_tapers} >= N:{num_samples}')

    tapers = np.zeros((num_tapers, num_samples))
    for k in range(1, num_tapers + 1):
        n = np.arange(num_samples)
        tapers[k - 1, :] = np.sqrt(2 / (num_samples + 1)) * np.sin(np.pi * k * (n + 1) / (num_samples + 1))
    return tapers


def sinusoidal_multitaper_sdf_matrix(X, num_tapers=5, sampling_frequency=1.0):
    """
    Calculate the sinusoidal multitaper spectral density matrix estimate for a multivariable time series.

    Parameters:
    X (np.ndarray): Input multivariable time series with shape (num_variables, num_samples).
    num_tapers (int): Number of sinusoidal tapers. Default is 5.
    sampling_frequency (float): Sampling frequency of the time series. Default is 1.0.

    Returns:
    freqs (np.ndarray): Frequencies at which the SDF is estimated.
    SDF (np.ndarray): Spectral density function matrix with shape (num_variables, num_variables, num_freqs).
    """
    num_variables, num_samples = X.shape

    # Generate sinusoidal tapers
    tapers = sinusoidal_tapers(num_samples, num_tapers)

    # Frequencies at which to estimate the SDF
    freqs = np.fft.fftfreq(num_samples, d=1 / sampling_frequency)
    num_freqs = len(freqs)


    # Initialize SDF matrix
    SDF = np.zeros((num_variables, num_variables, num_freqs), dtype=complex)

    # Multi-taper spectral estimation
    for k in range(num_tapers):
        # Taper the data
        tapered_data = X * tapers[k]

        # Fourier transform of tapered data
        tapered_fft = np.fft.fft(tapered_data, axis=1)

        # Accumulate the periodogram estimates
        for i in range(num_variables):
            for j in range(num_variables):
                new = tapered_fft[i, :] * np.conjugate(tapered_fft[j, :])
                SDF[i, j, :] += new

    # Normalize by the number of tapers and samples
    SDF /= num_tapers

    # Only return the positive frequencies
    positive_freqs = freqs[:num_samples // 2]
    SDF = SDF[:, :, :num_samples // 2]

    return positive_freqs, SDF


def choose_num_tapers(N, W):
    """
    Choose the number of sinusoidal tapers for spectral density estimation using the specified bandwidth.

    Parameters:
    N (int): Length of the time series.
    W (float): Desired bandwidth.

    Returns:
    int: Number of sinusoidal tapers.
    """
    # Calculate the time-bandwidth product
    NW = N * W

    # Choose the number of tapers
    K = int(np.floor(2 * NW) - 1)

    return K

# Random package

def estimate_cross_sdf_matrix(time_series, fs=1.0, nperseg=None, noverlap=None, nfft=None):
    """
    Calculate the estimate of the cross-spectral density function (SDF) matrix using scipy's csd function.

    Parameters:
    time_series (np.ndarray): Multivariate time series data of shape (num_series, num_samples).
    fs (float): Sampling frequency of the time series. Default is 1.0.
    nperseg (int): Length of each segment for computing the periodogram. Default is None.
    noverlap (int): Number of points to overlap between segments. Default is None.
    nfft (int): Length of the FFT used. Default is None.

    Returns:
    sdf_matrix (np.ndarray): Estimated cross-SDF matrix of shape (num_series, num_series, num_frequencies).
    frequencies (np.ndarray): Array of frequency values.
    """
    num_series, num_samples = time_series.shape
    if nperseg is None:
        nperseg = num_samples // 8
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = num_samples

    frequencies, Pxy = scipy.signal.csd(time_series[0], time_series[1], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    num_frequencies = len(frequencies)

    sdf_matrix = np.zeros((num_series, num_series, num_frequencies), dtype=complex)

    for i in range(num_series):
        for j in range(i, num_series):
            frequencies, Pxy = scipy.signal.csd(time_series[i], time_series[j], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            sdf_matrix[i, j, :] = Pxy
            if i != j:
                sdf_matrix[j, i, :] = np.conj(Pxy)

    return frequencies, sdf_matrix


### Hypothesis Testing ###

def calculate_kl_divergence(T1, T2, T):
    """
    Calculate the Kullback-Leibler divergence between T1 and T2.

    Parameters:
    T1 (np.ndarray): The first matrix of shape (n, n, num_freq).
    T2 (np.ndarray): The second matrix of shape (n, n, num_freq).

    Returns:
    float: The estimated Kullback-Leibler divergence.
    """
    n, _, num_freq = T1.shape
    kl_div = 0

    for k in range(1, num_freq):
        T1_k = T1[:, :, k]
        T2_k = T2[:, :, k]
        inv_T2_k = np.linalg.inv(T2_k)
        matrix_term = T1_k @ inv_T2_k
        term1 = np.trace(matrix_term)
        term2 = np.log(np.linalg.det(matrix_term))



        if k != 0:
            kl_div += term1 - term2 - n

    kl_div /= T

    return kl_div


def test_statistic(T1, T2, N, M, D, m1, m2, C):
    """
    Calculate the test statistic Z(T1, T2).

    Parameters:
    T1 (np.ndarray): The first matrix of shape (n, n, num_freq).
    T2 (np.ndarray): The second matrix of shape (n, n, num_freq).
    N (int): Sample size.
    M (int): Number of iterations.
    D (int): Dimensionality.
    m1 (int): Lower index.
    m2 (int): Upper index.
    C (float): A constant.

    Returns:
    float: The test statistic Z.
    """
    kl_div = calculate_kl_divergence(T1, T2, N)
    term1 = np.sqrt(M * N / (D * (m2 - m1)))
    term2 = (kl_div - (C * (m2 - m1) / M))
    Z = term1 * term2
    return Z


def single_step_update(T, E):
    """
    Perform a single step update for the 3D matrix T (n, n, num_freq) based on the edge set E.

    Parameters:
    T (np.ndarray): The initial matrix T(0) of shape (n, n, num_freq).
    E (set of tuple): The edge set E, where each element is a tuple (j, k).

    Returns:
    np.ndarray: Updated matrix T after one step for each frequency.
    """
    n, _, num_freq = T.shape
    T_new = T.copy()

    for f in range(num_freq):
        T_f = T[:, :, f]
        T_f_new = T_f.copy()

        for j, k in E:
            if j < k:
                # Calculate the inverse elements required
                T_f_inv = np.linalg.inv(T_f)
                T_kj_inv = T_f_inv[k, j]
                T_jk_inv = T_f_inv[j, k]
                T_jj_inv = T_f_inv[j, j]
                T_kk_inv = T_f_inv[k, k]

                # Update T_f[j, k] and T_f[k, j]
                numerator_jk = T_jk_inv
                denominator_jk = T_jj_inv * T_kk_inv - T_jk_inv * T_kj_inv
                update_value_jk = numerator_jk / denominator_jk

                numerator_kj = T_kj_inv
                denominator_kj = T_kk_inv * T_jj_inv - T_kj_inv * T_jk_inv
                update_value_kj = numerator_kj / denominator_kj

                T_f_new[j, k] += update_value_jk
                T_f_new[k, j] += update_value_kj

        T_new[:, :, f] = T_f_new

    return T_new


def iterate_algorithm(T, E, N):
    """
    Perform N iterations of the single step update for the 3D matrix T.

    Parameters:
    T (np.ndarray): The initial matrix T(0) of shape (n, n, num_freq).
    E (set of tuple): The edge set E, where each element is a tuple (j, k).
    N (int): Number of iterations.

    Returns:
    np.ndarray: Updated matrix T after N iterations.
    """
    T_current = T.copy()

    for _ in range(N):
        T_current = single_step_update(T_current, E)

    return T_current


def calculate_test_stat(T_0, E_1, E_2, best_m, T, C, D, m1, m2, num_iter):
    if len(E_1) == 0:
        T_1 = T_0.copy()

    else:
        T_1 = iterate_algorithm(T_0, E_1, num_iter)

    T_2 = iterate_algorithm(T_0, E_2, num_iter)

    # TODO: Delete those plots

    # frequences = np.arange(T_0.shape[2])
    # T_2_inv = invert_spectral_matrix(T_2, 0)
    # T_1_inv = invert_spectral_matrix(T_1, 0)
    # plot_sdf_with_theoretical_plt(frequences, T_2_inv, T_1_inv, title=f"{E_1} vs {E_2} inverse")
    # plot_sdf_with_theoretical_plt(frequences, T_2, T_1, title=f"{E_1} vs {E_2} normal")


    test_stat = test_statistic(T_1, T_2, T, best_m, D, m1, m2, C)
    return T_1, T_2, test_stat


def check_last_element(value):
    if len(value) == 0:
        return False
    return value[-1] == np.inf


def C_k(alpha, L_k, type="metsuda"):
    """
    Compute C_k(alpha) for a given alpha and L_k.

    Parameters:
    alpha (float): The significance level.
    L_k (int): The given parameter L_k.

    Returns:
    float: The computed value of C_k(alpha).
    """
    if type == "metsuda":
        # Calculate the value of (1 - alpha)^(1 / L_k)
        value = (1 - alpha) ** (1 / L_k)
    elif type == "walden":
        value = (1 - alpha / L_k)

    # Calculate the inverse of the standard Gaussian CDF (probit function)
    C_k_value = stats.norm.ppf(value)

    return C_k_value


def cvll_criterion(x, m):
    """
    Calculate the CVLL criterion for a given bandwidth m.

    Parameters:
    I (np.ndarray): Periodogram of shape (num_series, num_series, num_frequencies).
    m (int): Bandwidth.
    num_frequencies (int): Number of frequencies.

    Returns:
    float: CVLL criterion value.
    """

    if m % 2 != 0:
        m += 1

    frequences, I = calculate_periodogram(x)

    N = x.shape[1]

    _, f_hat_minus_j_0, _ = calculate_freq_avg_periodogram(x, m, f_minus_j_0=True)

    num_freq = I.shape[2]

    cvll = 0.0
    for j in range(num_freq):  # TODO: now only half frequencies //4
        hat_f_minus_j = f_hat_minus_j_0[:, :, j]
        inv_hat_f_minus_j = np.linalg.inv(hat_f_minus_j)
        term1 = np.trace(I[:, :, j] @ inv_hat_f_minus_j)
        term2 = np.log(np.linalg.det(hat_f_minus_j))
        cvll += term1 + term2

    cvll /= N
    return np.real(cvll)


def find_best_m(x, m_range):
    cvll_values = []
    for m in m_range:
        cvll = cvll_criterion(x, m)
        cvll_values.append(cvll)
    # Find the bandwidth with the minimum CVLL criterion value
    min_m = m_range[np.argmin(np.real(cvll_values))]

    return cvll_values, min_m


def plot_cvll_criterion(m_range, cvll_values, min_m):
    """
    Plot the CVLL criterion values for a range of bandwidth values using Plotly.

    Parameters:
    m_range (list or np.ndarray): Range of bandwidth values to evaluate.
    cvll_values (list or np.ndarray): CVLL criterion values corresponding to the bandwidth values.
    min_m (float): Bandwidth value that minimizes the CVLL criterion.
    """
    # Ensure cvll_values contains only real numbers
    cvll_values = np.real(cvll_values)

    # Create the plot
    fig = go.Figure()

    # Add the CVLL criterion line
    fig.add_trace(go.Scatter(
        x=m_range,
        y=cvll_values,
        mode='lines',
        name='CVLL Criterion',
        line=dict(color='blue')
    ))

    # Add the vertical line indicating the minimum CVLL value
    fig.add_vline(x=min_m, line=dict(color='red', dash='dash'), name=f'Minimum CVLL at m={min_m}')

    # Add annotations
    # fig.add_annotation(
    #     x=min_m,
    #     y=np.min(cvll_values),
    #     text=f'Minimum CVLL at m={min_m}',
    #     showarrow=True,
    #     arrowhead=1
    # )

    # Set plot title and labels
    fig.update_layout(
        # title='CVLL Criterion vs Bandwidth',
        xaxis_title='Bandwidth (m)',
        yaxis_title='CVLL Criterion Value',
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
    )

    return fig


def backward_stepwise_selection(x, best_m, T, C, D, m1, m2, num_iter, alpha):
    """
    Perform the backward stepwise selection algorithm to identify the best graphical model.

    Parameters:
    x (np.ndarray): The input data.
    best_m (int): The best bandwidth parameter.
    T (np.ndarray): Some precomputed matrix needed for the test statistic calculation.
    C (float): A constant used in the test statistic calculation.
    D (float): A constant used in the test statistic calculation.
    E_0 (set): Initial set of edges representing the complete graph. (only pairs (j,k) s.t. j<k)
    m1 (int): A constant used in the test statistic calculation.
    m2 (int): A constant used in the test statistic calculation.
    num_iter (int): Number of iterations for the T algorithm.
    alpha (float): Significance level for the statistical test.

    Returns:
    pd.DataFrame: DataFrame containing the results of each iteration, including edges, test statistics,
                  critical value, and selected edge.
    """

    E_0 = generate_all_possible_edges(x.shape[0])

    # Initialize variables
    final_table = {key: [] for key in E_0}
    c_k_s = []
    L_k = len(final_table)
    last_removed_edge = None
    all_edge_removed = set()

    _, T_0, _ = calculate_freq_avg_periodogram(x, best_m)
    edge_list = list(final_table.keys())

    while True:
        results = []
        for edge, value in final_table.items():
            if edge == last_removed_edge or check_last_element(value):
                results.append(np.inf)
                final_table[edge].append(np.inf)
            else:
                # Define models missing edges
                if last_removed_edge is None:
                    E = {edge}
                else:
                    E = {edge} | all_edge_removed

                T_1, T_2, test_stat = calculate_test_stat(T_0, all_edge_removed, E, best_m,
                                                          T, C, D, m1, m2, num_iter)
                results.append(np.real(test_stat))
                value.append(np.round(np.real(test_stat), 4))

        # Step 2: Find C_k(alpha) and check the stopping criterion
        C_k_alpha = C_k(alpha, L_k, type="metsuda")
        c_k_s.append(np.round(C_k_alpha, 4))


        if all(z > C_k_alpha for z in results):
            break  # Stop the procedure and select E_k as the graphical model

        # Find the candidate with the smallest test statistic
        min_j = np.argmin(results)
        last_removed_edge = edge_list[min_j]
        all_edge_removed = all_edge_removed | {last_removed_edge}

        # Step 3: Update edge count
        L_k -= 1

        if L_k == 0:
            break

    # Convert results to a DataFrame
    df_results = pd.DataFrame(final_table)
    df_results.index = [f'k={i}' for i in df_results.index]
    df_results.columns = [str(col) for col in df_results.columns]
    df_results.replace(np.inf, None, inplace=True)
    df_results = df_results.transpose()

    # Add a custom row to the transposed DataFrame
    df_results.loc[f'C_k({alpha})'] = c_k_s

    edges_left = E_0 - all_edge_removed

    # st.write(f'MATUSA EDGES LEFT: {edges_left}')

    # Save DataFrame to a CSV file
    # df_results.to_csv('output.csv', index=False)

    return df_results, edges_left


def walden_one_step_algorithm(x, best_m, T, C, D, m1, m2, num_iter, alpha):
    E_0 = generate_all_possible_edges(x.shape[0])

    # Initialize variables
    final_table = {key: [] for key in E_0}
    L_k = len(final_table)
    C_k_list = []

    _, T_0, _ = calculate_freq_avg_periodogram(x, best_m)

    for edge, value in final_table.items():
        E = {edge}
        T_1, T_2, test_stat = calculate_test_stat(T_0, set(), E, best_m,
                                                  T, C, D, m1, m2, num_iter)

        value.append(np.round(np.real(test_stat), 4))

        C_k_alpha = C_k(alpha, L_k, type="walden")
        C_k_list.append(np.round(C_k_alpha, 4))

        L_k -= 1

    # sort the test statistics
    sorted_items = sorted(final_table.items(), key=lambda item: item[1][0], reverse=True)
    # Convert the sorted items back to a dictionary
    final_table = dict(sorted_items)

    for i, (_, value) in enumerate(final_table.items()):
        value.append(C_k_list[i])

    # Convert results to a DataFrame
    df_results = pd.DataFrame(final_table)
    df_results.index = ['Z_i', f'C_k({alpha})']
    df_results.columns = [str(col) for col in df_results.columns]
    df_results.replace(np.inf, None, inplace=True)
    df_results = df_results.transpose()

    edges_left = E_0 - {key for key, values in final_table.items() if values[0] < values[1]}

    # st.write(f'WALDEN EDGES LEFT: {edges_left}')

    return df_results, edges_left


def generate_all_possible_edges(num_nodes):
    """
    Generate a set of all possible edges for a given number of nodes.

    Parameters:
    num_nodes (int): The number of nodes.

    Returns:
    set: A set of tuples representing all possible edges.
    """
    nodes = range(num_nodes)
    all_edges = set(combinations(nodes, 2))
    return all_edges


def plot_graph(num_vertices, pred_edges, true_edges):
    """
    Plot a graph structure given a number of vertices, predicted edges, and true edges.

    Parameters:
    num_vertices (int): The number of vertices.
    pred_edges (set of tuples): A set of tuples representing predicted edges between nodes.
    true_edges (set of tuples): A set of tuples representing true edges between nodes.
    """

    # Clear the current figure
    plt.clf()

    # Create a NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))  # Ensure all vertices are added
    G.add_edges_from(pred_edges | true_edges)

    # Rename nodes to X0, X1, X2, ...
    mapping = {node: f"X{node}" for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    # Map edges to the new node labels
    pred_edges_mapped = {(mapping[u], mapping[v]) for u, v in pred_edges}
    true_edges_mapped = {(mapping[u], mapping[v]) for u, v in true_edges}

    # Separate edges into different categories
    common_edges = pred_edges_mapped.intersection(true_edges_mapped)
    only_pred_edges = pred_edges_mapped.difference(true_edges_mapped)
    only_true_edges = true_edges_mapped.difference(pred_edges_mapped)

    # Positions for all nodes
    pos = nx.circular_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000)
    nx.draw_networkx_labels(G, pos)

    # Draw edges with different styles
    nx.draw_networkx_edges(G, pos, edgelist=common_edges, edge_color='green')
    nx.draw_networkx_edges(G, pos, edgelist=only_pred_edges, edge_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=only_true_edges, edge_color='orange', style='dashed')

    # Create a legend
    legend_labels = {
        'Correct classification': 'green',
        'False positive': 'red',
        'False negative': 'orange'
    }

    legend_handles = [
        plt.Line2D([0], [0], color=color, linewidth=3, linestyle='-' if color != 'orange' else '--')
        for label, color in legend_labels.items()
    ]
    # Add legend below the plot
    plt.legend(legend_handles, legend_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    # Adjust the layout to make space for the legend
    plt.subplots_adjust(bottom=0.2)

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Close the plot to free memory
    plt.close()


### Extra graph plotting ###

def find_missing_edges(A_list):
    zero_entries = set()
    A_list = [A_list[-1]]
    for A in A_list:
        n = A.shape[0]
        for j in range(n):
            for k in range(j + 1, n):  # ensuring j < k
                if A[j, k] == 0 and A[k, j] == 0:
                    zero_entries.add((j, k))
    return zero_entries


def true_non_missing_edges(A_list):
    num_nodes = A_list[0].shape[0]
    all_edges = generate_all_possible_edges(num_nodes)
    missing_edges = find_missing_edges(A_list)
    non_missing_edges = all_edges - missing_edges
    return non_missing_edges


def main():

    A_list_single = [np.array([
        [0.5, 0.0, 0.0],
        [0.5, 0, 0.0],
        [0.5, 0.0, 0]
    ])]

    A_list_independent = [np.array([
        [0.9, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, 0.0, 0.7]
    ])]

    A_list_complex = [np.array([
        [0.5, 0.5, 0.1],
        [0.25, 0.5, 0.0],
        [0.25, 0.0, 0.5]
    ])]

    T = 1024
    C = 0.617
    D = 0.446
    m1 = 0
    m2 = 1
    num_iter = 10
    alpha = 0.05

    for A_list in [A_list_single, A_list_independent, A_list_complex]:

        x = generate_var_process(A_list, T, 1000)
        x = x - np.mean(x, axis=1, keepdims=True)

        m_range = np.arange(30, 300, 2)
        _, best_m = find_best_m(x, m_range)

        ans_table, _ = backward_stepwise_selection(x, best_m, T, C, D,
                                                m1, m2, num_iter, alpha)

        print("Coeff matrix:")
        print(A_list)
        print('------')
        print("Result of the Matsuda Algorithm")
        print(ans_table)


if __name__ == "__main__":
    main()





