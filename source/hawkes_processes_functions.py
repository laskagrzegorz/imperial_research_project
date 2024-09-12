from .time_series_functions import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from numba import njit
import plotly.graph_objects as go
import streamlit as st
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence, plot_gaussian_process


### Simulation of Mutually Exciting Hawkes Processes ###


def mutual_exp_simulate_by_thinning(theta, T):
    """
    The lambda is an m-vector which shows the starting intensity for
    each process.

    Each alpha[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The beta is an m-vector which shows the intensity decay rates for
    each process intensity.
    """
    lambda_, alpha, beta = theta
    m = len(lambda_)

    lambda_current = lambda_
    times = []

    t = 0

    while True:
        M = np.sum(lambda_current)
        delta_t = rnd.exponential() / M
        t += delta_t
        if t > T:
            break

        lambda_current = lambda_ + (lambda_current - lambda_) * np.exp(-beta * delta_t)

        u = M * rnd.rand()
        if u > np.sum(lambda_current):
            continue # No arrivals (they are 'thinned' out)

        cumulative_lambda_current = 0

        for i in range(m):
            cumulative_lambda_current += lambda_current[i]
            if u < cumulative_lambda_current:
                times.append((t, i))
                lambda_current += alpha[i]
                break

    return times


def bin_hawkes_process(times, ids, T, num_bins):
    """
    Bin the mutually exciting Hawkes process into specified time intervals.

    Parameters:
    times (list of float): List of event times.
    ids (list of int): List of process IDs corresponding to the event times.
    T (float): Time horizon.
    num_bins (int): Number of bins.

    Returns:
    bins (numpy array): Array of bin edges.
    binned_counts (numpy array): Array of binned event counts for each process.
    """
    bins = np.linspace(0, T, num_bins + 1)
    num_processes = len(set(ids))
    binned_counts = np.zeros((num_processes, num_bins))

    for t, i in zip(times, ids):
        bin_index = np.searchsorted(bins, t) - 1
        if bin_index < num_bins:
            binned_counts[i, bin_index] += 1

    return bins, binned_counts


def check_hawkes_process(theta):
    lambda_, alpha, beta = theta
    check = is_stationary(alpha, beta)
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


def is_stationary(alpha, beta):
    """
    Check if a multivariate Hawkes process is stationary.

    Parameters:
    alpha (numpy.ndarray): The excitation matrix (m x m).
    beta (numpy.ndarray): The decay rates (m).

    Returns:
    bool: True if the process is stationary, False otherwise.
    """
    # Ensure beta is a diagonal matrix
    D_beta_inv = np.diag(1.0 / beta)

    # Compute the matrix A
    A = np.dot(alpha, D_beta_inv)

    # Compute the spectral radius (maximum absolute eigenvalue) of A
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eigenvalues))

    # Check if the spectral radius is less than 1
    return spectral_radius < 1


@njit(nogil=True)
def mutual_exp_hawkes_intensity(t, times, ids, theta):
    """
    The lambda_ is an m-vector which shows the starting intensity for
    each process.

    Each alpha[i] is an m-vector which shows the jump in intensity
    for each of the processes when an arrival comes to process i.

    The beta is an m-vector which shows the intensity decay rates for
    each process intensity.
    """
    lambda_, alpha, beta = theta

    lambda_current = lambda_.copy()
    for (t_i, d_i) in zip(times, ids):
        if t_i < t:
            lambda_current += alpha[d_i] * np.exp(-beta * (t - t_i))

    return lambda_current


def display_hawkes_equation(theta):
    """
    Display the equation of mutually exciting processes using Streamlit.

    Parameters:
    lambda_ (numpy array): Baseline intensities for each process.
    alpha_ (numpy array): Excitation matrix.
    beta (numpy array): Decay rates for each process.
    """

    lambda_, alpha_, beta = theta

    num_processes = len(lambda_)

    equation = r""
    for i in range(num_processes):
        lambda_i = f"\\lambda_{i + 1}(t) &= {lambda_[i]}"
        excitation_terms = ""
        for j in range(num_processes):
            if alpha_[i, j] != 0:
                excitation_terms += f" + {alpha_[i, j]} \\sum_{{t_{{ {j+1}, k}} < t}} e^{{-\\beta_{i + 1} (t - t_{{ {j+1}, k}})}}"
        equation += lambda_i + excitation_terms + r"\\\\"

    beta_terms = ", ".join([f"\\beta_{{{i + 1}}}={beta[i]}" for i in range(num_processes)])
    equation += f"\n \\quad where {beta_terms}"

    st.markdown(rf"""
    ### Mutually Exciting Processes
    The equations for the mutually exciting processes are given by:

    $$
    \begin{{align*}}
    {equation}
    \end{{align*}}
    $$
    """, unsafe_allow_html=True)


def compute_intensity_over_time(times, ids, theta, T, num_points=1000):
    """
    Compute the intensity functions over a range of time points.

    Parameters:
    times (list of float): List of event times.
    ids (list of int): List of process IDs corresponding to the event times.
    theta (tuple): Tuple containing lambda_, alpha, and beta.
    T (float): Time horizon.
    num_points (int): Number of time points to evaluate the intensity.

    Returns:
    time_points (numpy array): Array of time points.
    intensities (numpy array): Array of intensities for each process over time.
    """
    time_points = np.linspace(0, T, num_points)
    num_processes = len(theta[0])
    intensities = np.zeros((num_processes, num_points))

    for j, t in enumerate(time_points):
        intensities[:, j] = mutual_exp_hawkes_intensity(t, times, ids, theta)

    return time_points, intensities

### Bayesian Optimalization ###


def bayesian_search_cvll(binned_counts, num_bins, percentage_of_frequencies=1):
    # Wrapper function for optimization
    def cvll_criterion_wrapper(params):
        m = int(params[0])  # Ensure m is treated as an integer
        return cvll_criterion(binned_counts, m, percentage_of_frequencies)

    # Define the search space for m (example range: 1 to 50)
    space = [
        Integer(20, num_bins // 2, name='m')
    ]

    # Perform Bayesian optimization
    result = gp_minimize(
        func=cvll_criterion_wrapper,
        dimensions=space,
        n_calls=30,
        random_state=42,
        n_initial_points=15,
        # noise=1e-10
    )

    return result

### Plotting ###

def plot_cumulative_counts(event_times, T, num_processes):
    """
    Plot the cumulative counts of events for each process over time.

    Parameters:
    event_times (list of arrays): List of arrays, where each array contains the event times for a process.
    T (float): Time horizon for the plot.
    num_processes (int): Number of processes.
    """
    time_points = np.linspace(0, T, 1000)
    cumulative_counts = np.zeros((num_processes, len(time_points)))

    for i in range(num_processes):
        events = event_times[i]
        current_count = 0
        for j, t in enumerate(time_points):
            while current_count < len(events) and events[current_count] <= t:
                current_count += 1
            cumulative_counts[i, j] = current_count

    # Create a figure
    fig = go.Figure()

    # Add traces for each process
    for i in range(num_processes):
        fig.add_trace(go.Scatter(
            x=time_points,
            y=cumulative_counts[i],
            mode='lines',
            name=f'Cumulative Count Process {i + 1}'
        ))

    # Update layout
    fig.update_layout(
        title='Cumulative Counts of Events for Multivariate Hawkes Processes',
        xaxis_title='Time',
        yaxis_title='Cumulative Count',
        legend_title='Processes',
        width=800,
        height=600
    )

    return fig


def plot_binned_event_counts(bins, binned_counts, lambda_):
    """
    Plot binned event counts of multivariate Hawkes processes using Plotly.

    Parameters:
    - bins: Array of bin edges.
    - binned_counts: List of arrays containing binned event counts for each process.
    - lambda_: List or array containing the rate parameter for each process.
    """

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Create a figure
    fig = go.Figure()

    # Add traces for each process
    for i in range(len(lambda_)):
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=binned_counts[i],
            mode='lines',
            name=f'Process {i + 1}'
        ))

    # Update layout
    fig.update_layout(
        title='Binned Event Counts of Multivariate Hawkes Processes',
        xaxis_title='Time',
        yaxis_title='Binned Event Counts',
        legend_title='Processes',
        width=800,
        height=600
    )

    return fig


### Combining functions ###

def simulate_and_bin_hawkes(theta, T, num_bins, seed=None):
    """
    Simulate the multivariate Hawkes process and bin the events.

    Parameters:
    theta (tuple): Tuple containing lambda_, alpha, and beta.
    T (float): Time horizon.
    num_bins (int): Number of bins for the histogram.

    Returns:
    event_times (list of list of float): List of event times for each process.
    bins (numpy.ndarray): Array of bin edges.
    binned_counts (numpy.ndarray): 2D array of binned counts for each process.
    """

    if seed is not None:
        np.random.seed(seed)

    # Simulate the processes
    times = mutual_exp_simulate_by_thinning(theta, T)

    lambda_ = theta[0]

    # Separate the event times by process
    event_times = [[] for _ in range(len(lambda_))]
    ids = []
    for t, i in times:
        event_times[i].append(t)
        ids.append(i)

    # Bin the Hawkes process
    bins, binned_counts = bin_hawkes_process([e[0] for e in times], ids, T, num_bins)

    return event_times, bins, binned_counts


### Simulation Functions ###

def simulate_maximin_stepdown_test_one_edge_m_hawkes(theta, T, num_bins, m, edge, alpha=0.05):
    # Generate time series
    event_times, bins, binned_counts = simulate_and_bin_hawkes(theta, T, num_bins)

    binned_counts -= np.mean(binned_counts, axis=1, keepdims=True)

    frequencies, T_0 = sinusoidal_multitaper_sdf_matrix(binned_counts, m)

    partial_coherence = calculate_partial_coherence(T_0)

    num_series, _, num_frequencies = partial_coherence.shape

    critical_values_holm = np.zeros(num_frequencies)
    # Calculate critical values for each frequency bin
    for l in range(1, num_frequencies + 1):
        critical_values_holm[l - 1] = calculate_critical_region_holm(alpha, num_frequencies, l, m, num_series)

    j, k = edge
    R_l = partial_coherence[j, k, :]
    R_ordered = np.sort(R_l)[::-1]
    # Check where R_ordered is greater than critical_values
    comparison = R_ordered > critical_values_holm
    comparison = np.sum(comparison)
    RHH = np.round(comparison / num_frequencies, 2)

    return RHH


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (tuple, list)):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj


def run_simulation_and_save_results_hawkes(config, max_value=25, num_simulations=1000, force_save=False, tag=""):
    m_list = config['m_list']
    T = config["T"]
    delta = config['delta']
    addition_term = np.array(config['addition_term'])
    edge = config['edge']
    initial_theta = config['theta']
    num_bins = config['num_bins']

    initial_theta_list = list(initial_theta)

    print(len(initial_theta_list))

    # Create a directory name excluding the timestamp
    dir_name_without_timestamp = f'size={initial_theta_list[1].shape[0]}_T={T}_n={num_simulations}_max={max_value}'

    # Set path to the neighbor directory
    base_dir = '../../../../Library/Application Support/JetBrains/PyCharm2023.3/scratches/hawkes_simulations'

    # Check if a directory with the same name (excluding timestamp) exists
    existing_dir_path = directory_exists(base_dir, dir_name_without_timestamp)
    if existing_dir_path and not force_save:
        print(f"Directory with the same configuration already exists: {existing_dir_path}")
        # Load the existing results
        config_file_path = os.path.join(existing_dir_path, 'simulation_config.json')
        results_file_path = os.path.join(existing_dir_path, 'results_per_m.pkl')

        with open(config_file_path, 'r') as file:
            loaded_config = json.load(file)

        with open(results_file_path, 'rb') as file:
            results_per_m = pickle.load(file)

        print(f"Loaded existing results from {existing_dir_path}")

        plot_simulation_results(loaded_config, results_per_m)

        return loaded_config, results_per_m

    # Create a new directory with timestamp
    timestamp = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    dir_name = f'{dir_name_without_timestamp}_{timestamp}'
    full_dir_path = os.path.join(base_dir, dir_name) + tag  # TODO: Chenge the name here
    os.makedirs(full_dir_path, exist_ok=True)

    results_per_m = dict()

    for m in m_list:
        print(f"Processing m = {m}")
        results_list = []

        theta_list = initial_theta_list.copy()

        lambda_, alpha_, beta = theta_list[0], theta_list[1], theta_list[2]

        theta = lambda_, alpha_, beta

        for i in range(max_value):
            items = range(num_simulations)  # Example list of items to process
            try:
                # Parallelize the for loop
                results = Parallel(n_jobs=-1)(
                    delayed(simulate_maximin_stepdown_test_one_edge_m_hawkes)(theta, T, num_bins, m, edge,
                                                                              alpha=0.05)
                    for item in tqdm(items, desc=f"Processing items for m={m}, iteration={i + 1}")
                )
                results_list.append(results)
            except Exception as e:
                print(f"An error occurred during iteration {i + 1} for m={m}: {e}")
                results_list.append([])  # Append empty results to maintain list structure in case of error

            lambda_, alpha_, beta = theta

            alpha_ = alpha_ + delta * addition_term


            theta = lambda_, alpha_, beta

        results_per_m[m] = results_list

    # Convert the config to a JSON-serializable format
    serializable_config = convert_to_serializable(config)

    # Save the configuration to a JSON file
    config_file_path = os.path.join(full_dir_path, 'simulation_config.json')
    with open(config_file_path, 'w') as file:
        json.dump(serializable_config, file, indent=4)

    # Save the results to a file
    results_file_path = os.path.join(full_dir_path, 'results_per_m.pkl')
    with open(results_file_path, 'wb') as file:
        pickle.dump(results_per_m, file)

    # save figure
    plot_simulation_results(config, results_per_m, path=full_dir_path)

    print(f"Results saved in directory: {full_dir_path}")

    return config, results_per_m


def run_one_hawkes_simulation_hawkes(theta, T, num_bins, best_m, edge=(1, 2), bayesian_test=False, percent=1):
    event_times, bins, binned_counts = simulate_and_bin_hawkes(theta, T, num_bins)

    binned_counts -= np.mean(binned_counts, axis=1, keepdims=True)
    # Constants
    C = 0.617
    D = 0.446
    m1 = 0
    m2 = 1
    num_iter = 10

    if bayesian_test:

        m_results = bayesian_search_cvll(binned_counts, num_bins, percentage_of_frequencies=percent)

        if int(m_results.x[0]) % 2 != 0:
            best_m = m_results.x[0] + 1
        else:
            best_m = m_results.x[0]

        _ = plot_gaussian_process(m_results)
        # Display the plot in Streamlit
        plt.show()

    _, T_0_mirror, frequencies = calculate_freq_avg_periodogram_mirrored(binned_counts, best_m)

    all_edge_removed = set()

    E = {edge} | all_edge_removed

    _, _, test_stat_mirror = calculate_test_stat(T_0_mirror, all_edge_removed, E, best_m,
                                                 num_bins, C, D, m1, m2, num_iter)

    return np.real(test_stat_mirror), best_m



