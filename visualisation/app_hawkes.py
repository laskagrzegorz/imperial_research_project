from app import *
from hawkes_processes_functions import *



### Parameters for the simulation ###

## 3x3 ##

lambda_ = np.array([0.2, 0.3, 0.2])  # Baseline intensities for each process
alpha_ = np.array([
    [0.4, 0.3, 0.0],  # Process 1 excites Process 1 and 2, but not 3
    [0.4, 0.3, 0.0],  # Process 2 excites Process 1 and 2, but not 3
    [0.0, 0.0, 0.5]   # Process 3 excites only itself
])
beta = np.array([1.0, 1.0, 1.0])     # Decay rates for each process

theta_3x3 = lambda_, alpha_, beta


# Define the parameters for a stationary 5x5 system
lambda_ = np.array([0.2, 0.3, 0.2, 0.4, 0.1])  # Baseline intensities for each process
alpha_ = np.array([
    [0.5, 0.3, 0.0, 0.2, 0.0],  # Process 1 excites multiple processes
    [0.0, 0.5, 0.0, 0.4, 0.2],  # Process 2 excites multiple processes
    [0.0, 0.0, 0.5, 0.3, 0.0],  # Process 3 excites multiple processes
    [0.0, 0.0, 0.0, 0.5, 0.4],  # Process 4 excites multiple processes
    [0.0, 0.0, 0.0, 0.0, 0.5]   # Process 5 excites multiple processes
])
beta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Decay rates for each process


theta_5x5 = lambda_, alpha_, beta


def run_hawkes_for_theta(T, seed, theta):

    lambda_, alpha_, beta = theta

    display_hawkes_equation(theta)
    check_hawkes_process(theta)
    ### Produce binned Hawkes processes ###
    binned_counts_list, bins_list, event_times_list, means_list, num_bins_list = produce_binned_hawkes_list(theta, T,
                                                                                                            seed)
    ### Plot the sdf estimates ###
    _, frequencies = calculate_periodogram(binned_counts_list[0])
    freq_avg_periodogram_sdf_list = []
    frequencies_list = []
    name_list = []
    for num_bins, binned_counts in zip(num_bins_list, binned_counts_list):
        name_list.append(f'num_bins {num_bins}')
        m = num_bins // 10
        _, freq_avg_periodogram_sdf, frequencies = calculate_freq_avg_periodogram(binned_counts, m)
        freq_avg_periodogram_sdf_list.append(freq_avg_periodogram_sdf)
        frequencies_list.append(frequencies)

    tabs = st.tabs(['Normal', 'Log-scale'])
    with tabs[0]:
        st.subheader("SDF estimates on normal scale")
        st.plotly_chart(
            figure_or_data=plot_sdf_with_theoretical(frequencies, freq_avg_periodogram_sdf_list, name_list,
                                                     theoretical_sdf=None, log_scale=False,
                                                     different_frequencies=frequencies_list),
            use_container_width=True,
        )
    with tabs[1]:
        st.subheader("SDF estimates on log scale")
        st.plotly_chart(
            figure_or_data=plot_sdf_with_theoretical(frequencies, freq_avg_periodogram_sdf_list, name_list,
                                                     theoretical_sdf=None, log_scale=True,
                                                     different_frequencies=frequencies_list),
            use_container_width=True,
        )
    show_plots = True
    if show_plots:
        st.subheader(body=f"Realisation of Mutually Exciting Hawkes processes ({alpha_.shape[0]})")
        st.plotly_chart(
            figure_or_data=plot_cumulative_counts(event_times_list[0], T, alpha_.shape[0]),
            use_container_width=True,
        )

        st.subheader(body=f"Realisation of Mutually Exciting Hawkes processes ({alpha_.shape[0]})")
        st.write(f'Mean of each Hawkes process is : {means_list[0]}')
        st.plotly_chart(
            figure_or_data=plot_binned_event_counts(bins_list[0], binned_counts_list[0], lambda_),
            use_container_width=True,
        )

    tab_names = []
    for num_bins in num_bins_list:
        tab_names.append(f"Number of bins: {num_bins}")
    tabs_bins = st.tabs(tab_names)

    for i, (num_bins, binned_counts, means) in enumerate(zip(num_bins_list, binned_counts_list, means_list)):
        with tabs_bins[i]:
            st.subheader('Number of bins: ' + str(num_bins))
            st.write(f"Mean of each process is: {means}")
            run_hawkes_for_num_bin(num_bins, binned_counts, theta, T)


def produce_binned_hawkes_list(theta, T, seed):
    num_bins_list = [1024, 2048, 4096, 8192, 16384]
    binned_counts_list = []
    means_list = []
    event_times_list = []
    bins_list = []
    for num_bins in num_bins_list:
        event_times, bins, binned_counts = simulate_and_bin_hawkes(theta, T, num_bins, seed)
        means = np.round(np.mean(binned_counts, axis=1), 2)

        st.write(np.sum(binned_counts, axis=1))
        st.write(np.max(binned_counts, axis=1))

        binned_counts -= np.mean(binned_counts, axis=1, keepdims=True)
        # Save
        binned_counts_list.append(binned_counts)
        means_list.append(means)
        event_times_list.append(event_times)
        bins_list.append(bins)
    return binned_counts_list, bins_list, event_times_list, means_list, num_bins_list


def run_hawkes_for_num_bin(num_bins, binned_counts, theta, T):
    # Constants
    C = 0.617
    D = 0.446
    m1 = 0
    m2 = 1
    num_iter = 10
    alpha = 0.05
    _, alpha_, _ = theta

    ### Bayesian Optimilization ###
    m_results = bayesian_search_cvll(binned_counts, num_bins)
    if int(m_results.x[0]) % 2 != 0:
        best_m = m_results.x[0] + 1
    else:
        best_m = m_results.x[0]
    st.write(f"Best m = {best_m}")
    cols = st.columns(2)
    with cols[0]:
        # Clear the current figure
        plt.clf()
        plot_convergence(m_results)
        # Display the plot in Streamlit
        st.pyplot(plt)

        # Close the plot to free memory
        plt.close()
    with cols[1]:
        # Clear the current figure
        plt.clf()
        _ = plot_gaussian_process(m_results)
        # Display the plot in Streamlit
        st.pyplot(plt)

        # Close the plot to free memory
        plt.close()

    run_algorithms_given_m(binned_counts, best_m, alpha, num_bins, C, D,
                           m2, m1, num_iter, [alpha_])


def main():

    T = 10240  # Time horizon for the simulation
    seed = np.random.randint(1, 1000)  # Seed for reproducibility

    st.set_page_config(
        page_title="Gregg's Research Hawkes",
        page_icon=":material/compare_arrows:",
        layout="wide",
    )

    st.header(body="Gregg's research project")
    st.markdown("---")

    tabs = st.tabs(["3x3 Hawkes processes", "5x5 Hawkes processes"])

    with tabs[0]:
        run_hawkes_for_theta(T, seed, theta_3x3)
    with tabs[1]:
        run_hawkes_for_theta(T, seed, theta_5x5)

    st.markdown("---")
    st.write(f"Random number used: {seed}")



if __name__ == "__main__":
    main()
