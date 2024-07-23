import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, probplot
from app_hawkes import *
import os
import json
import pickle


### First and second algorithm functions ###

def split_filename(filename):
    # Split the filename on "_"
    parts = filename.split("_")

    # Remove the last part (which is "2048.csv") and split again to discard ".csv"
    parts_without_extension = parts[:-1] + [parts[-1].split(".")[0]]

    num_time_series = int(parts_without_extension[0][0])
    num_simulatons = int(parts_without_extension[1])
    T = int(parts_without_extension[-1].split("=")[-1])
    m_value = parts_without_extension[-2].split("=")[-1]
    if m_value == 'variable':
        m_value = None
    else:
        m_value = int(m_value)

    return num_time_series, num_simulatons, T, m_value


def plot_histogram_with_gaussian(test_stats):
    # Calculate mean and standard deviation (which is the square root of variance)
    mean_result = np.mean(test_stats)
    std_result = np.std(test_stats)

    # Create histogram trace
    histogram = go.Histogram(
        x=test_stats,
        nbinsx=50,
        histnorm='probability density',
        marker=dict(color='rgba(100, 100, 255, 0.6)', line=dict(color='black', width=1)),
        name='Empirical Test Statistics'
    )

    # Create Gaussian fit trace using mean and std from the data
    x = np.linspace(min(test_stats), max(test_stats), 100)
    gaussian_fit = go.Scatter(
        x=x,
        y=norm.pdf(x, mean_result, std_result),
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Gaussian Fit: μ={mean_result:.2f}, σ={std_result:.2f}'
    )

    # Create Gaussian fit trace for standard normal distribution
    gaussian_standard = go.Scatter(
        x=x,
        y=norm.pdf(x, 0, 1),
        mode='lines',
        line=dict(color='black', width=2),
        name='Density N(0,1)'
    )

    # Create the figure and add traces
    fig = go.Figure(data=[histogram, gaussian_fit, gaussian_standard])

    # Add titles and labels
    fig.update_layout(
        title='Histogram with Gaussian Fit',
        xaxis_title='Value',
        yaxis_title='Density',
        legend_title='Legend'
    )

    return fig


def plot_qq_plot(test_stats):
    # Calculate the theoretical quantiles and sample quantiles
    qq_data = probplot(test_stats, dist="norm", fit=False)[:2]
    osm = qq_data[0]  # Theoretical quantiles
    osr = qq_data[1]  # Sample quantiles

    # Create scatter plot for Q-Q data
    qq_scatter = go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(symbol='cross', color='black', size=5),  # "cross" for "+" symbol
        name='Q-Q Data'
    )

    # Create line plot for the reference line from (-5, -5) to (5, 5)
    qq_line = go.Scatter(
        x=[-5, 5],
        y=[-5, 5],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Reference Line'
    )

    # Create the figure and add traces
    fig = go.Figure(data=[qq_scatter, qq_line])

    # Add titles, labels, and set axis limits
    fig.update_layout(
        title='Q-Q Plot of Results vs. Standard Normal Distribution',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[-5, 5])
    )

    return fig


def plot_histogram_with_mean(m_values):
    # Calculate the mean
    mean_value = np.mean(m_values)

    # Create histogram trace
    histogram = go.Histogram(
        x=m_values,
        nbinsx=30,
        marker=dict(color='rgba(100, 100, 255, 0.7)', line=dict(color='black', width=1)),
        name='Histogram'
    )

    # Create vertical line trace at the mean
    mean_line = go.Scatter(
        x=[mean_value, mean_value],
        y=[0, max(np.histogram(m_values, bins=30)[0])],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name=f'Mean: {mean_value:.2f}'
    )

    # Create the figure and add traces
    fig = go.Figure(data=[histogram, mean_line])

    # Add titles and labels
    fig.update_layout(
        title='Histogram of m_values with Mean Line',
        xaxis_title='Value',
        yaxis_title='Frequency',
        legend_title='Legend'
    )

    return fig


def simulate_for_given_path(file_path):
    full_path = '../simulations_data/' + file_path
    num_series, _, T, m_value = split_filename(file_path)
    df = pd.read_csv(full_path)
    test_stats = df['Value']
    st.subheader(f"Test Statistics for missing edge of {num_series}x{num_series} Var(1)")
    cols = st.columns(2)
    with cols[0]:
        st.write("Number of simulations: ", len(test_stats))
        st.write(f'Number of samples in each time series: ', T)
        if m_value is not None:
            st.write("Constant parameter m equal to ", m_value)
        else:
            m_values = df['M_value']
            st.write("Variable parameter m with average equal to ", m_values.mean())

        # Calculate the skewness
        skewness = stats.skew(test_stats)

        # Print the skewness
        st.write(f'Skewness of the data: ', np.round(skewness, 2))

        # Compare with standard normal distribution
        if skewness == 0:
            st.write("The distribution is symmetrical (similar to a standard normal distribution).")
        elif skewness > 0:
            st.write("The distribution is positively skewed (right tail is longer).")
        else:
            st.write("The distribution is negatively skewed (left tail is longer).")
    with cols[1]:
        if m_value is None:
            with st.expander("Parameter M distribution"):
                st.plotly_chart(
                    figure_or_data=plot_histogram_with_mean(m_values),
                    use_container_width=True,
                )
        else:
            pass
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(
            figure_or_data=plot_histogram_with_gaussian(test_stats),
            use_container_width=True,
        )
    with cols[1]:
        st.plotly_chart(
            figure_or_data=plot_qq_plot(test_stats),
            use_container_width=True,
        )


### Third algorithm function ###

def get_simulation_directories_third_algorithm(base_dir='../third_algorithm_simulations'):
    """
    List all directories starting with 'size' from the given base directory and sort them by name.

    Parameters:
    base_dir (str): The base directory to search in. Defaults to '../third_algorithm_simulations'.

    Returns:
    list: A sorted list of directory names starting with 'size'.
    """
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        raise ValueError(f"The base directory '{base_dir}' does not exist.")

    # List all directories starting with 'size'
    directories = [os.path.join(base_dir, name) for name in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, name)) and name.startswith('size')]

    # Sort the directories by name
    directories.sort()

    return directories


def categorize_directories(directories):
    categories = {
        'independent': [],
        'add_one': [],
        'complex': [],
        'other': []
    }

    for dir in directories:
        if 'independent' in dir:
            categories['independent'].append(dir)
        elif 'one_add' in dir:
            categories['add_one'].append(dir)
        elif 'complex' in dir:
            categories['complex'].append(dir)
        else:
            categories['other'].append(dir)

    return categories


def get_simulation_data_third_algorithm(dir_path):
    # Load the existing results
    config_file_path = os.path.join(dir_path, 'simulation_config.json')
    results_file_path = os.path.join(dir_path, 'results_per_m.pkl')

    with open(config_file_path, 'r') as file:
        loaded_config = json.load(file)

    with open(results_file_path, 'rb') as file:
        results_per_m = pickle.load(file)

    return loaded_config, results_per_m


def show_json_config(config):

    with st.expander(f"Parameters: T = {config['T']}", expanded=False):

        cols = st.columns(3)

        with cols[0]:
            st.subheader("Configuration")
            # Display each element of the JSON configuration
            st.write(f"T =", config['T'])

            st.write(f"Alpha =", config['alpha'])

            st.write(f"Edge = {config['edge']}")

            st.write(f"m_list = {config['m_list']}" )

            st.write(f"Delta =", config['delta'])

        with cols[1]:

            st.subheader("Addition Term Matrix")
            st.write(pd.DataFrame(config['addition_term']))

        with cols[2]:
            st.subheader("Initial A Matrix List")
            st.write(pd.DataFrame(config['initial_A_list']))


def show_one_path_third_algorithm(path):
    loaded_config, results_per_m = get_simulation_data_third_algorithm(path)
    show_json_config(loaded_config)
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(
            figure_or_data=plot_simulation_third_algorithm(loaded_config, results_per_m, plot_type='power'),
            use_container_width=True,
        )
    with cols[1]:
        st.plotly_chart(
            figure_or_data=plot_simulation_third_algorithm(loaded_config, results_per_m, plot_type='average'),
            use_container_width=True,
        )

def extract_T_from_path(path):
    """
    Extract the T value from the directory path.

    Parameters:
    path (str): The directory path.

    Returns:
    str: The extracted T value.
    """
    base_name = os.path.basename(path)
    T_value = base_name.split('_')[1].split('=')[1]
    return T_value



def plot_simulation_third_algorithm(config, results_per_m, plot_type='power'):
    """
    Plots the percentage of results not equal to zero or the average results per iteration
    for different m values using Plotly.

    Parameters:
    config (dict): Configuration dictionary containing 'm_list' and 'delta'.
    results_per_m (dict): Dictionary containing results for different m values.
    plot_type (str): The type of plot to generate. Either 'percent_zero' or 'average'.

    Returns:
    go.Figure: The Plotly figure object.
    """
    fig = go.Figure()

    if plot_type == 'power':
        for m in config['m_list']:
            percent_zero_results = [(np.array(results) != 0).mean() * 100 for results in results_per_m[m]]
            fig.add_trace(go.Scatter(x=np.arange(0, len(percent_zero_results) * config['delta'], config['delta']),
                                     y=percent_zero_results,
                                     mode='lines',
                                     name=f'm={m}'))

        fig.add_trace(go.Scatter(x=[0, len(percent_zero_results) * config['delta']],
                                 y=[5, 5],
                                 mode='lines',
                                 line=dict(dash='dash'),
                                 name='5% threshold'))

        fig.update_layout(title='Percentage of Results Not Equal to Zero per Iteration for Different m Values',
                          xaxis_title='Iteration',
                          yaxis_title='Percentage of Results Not Equal to Zero (%)')
    elif plot_type == 'average':
        for m in config['m_list']:
            averaged_results = [np.mean(results) for results in results_per_m[m]]
            fig.add_trace(go.Scatter(x=np.arange(0, len(averaged_results) * config['delta'], config['delta']),
                                     y=averaged_results,
                                     mode='lines',
                                     name=f'm={m}'))

        fig.update_layout(title='Average Results per Iteration for Different m Values',
                          xaxis_title='Iteration',
                          yaxis_title='Average Result')
    else:
        raise ValueError("Invalid graph_type. Choose 'percentage' or 'average'.")

    fig.update_layout(showlegend=True)

    return fig


def show_all_simulations_third_algorithm(base_dir='../third_algorithm_simulations'):

    path_list = get_simulation_directories_third_algorithm(base_dir)
    categorised_paths = categorize_directories(path_list)

    # tab_names = [f"simulation {i+1}" for i in range(len(path_list))]
    tab_names = [f"simulation {name}" for name in categorised_paths.keys()]

    tabs = st.tabs(tab_names)

    for i, (category_name, name) in enumerate(zip(categorised_paths.keys(), tab_names)):

        with tabs[i]:
            for path in categorised_paths[category_name]:
                show_one_path_third_algorithm(path)

            st.markdown(f"---")
            st.subheader("Analysis by m")

            plot_results_for_m(categorised_paths[category_name])


def plot_results_for_m(paths):
    """
    Create a plot for each value of m with lines for each directory.

    Parameters:
    paths (list): A list of paths to the directories containing the results.

    Returns:
    None
    """
    all_results = []
    for path in paths:
        config, results_per_m = get_simulation_data_third_algorithm(path)
        all_results.append((config, results_per_m, path))

    m_values = config['m_list']

    # Initialize two columns in Streamlit
    col1, col2 = st.columns(2)

    for idx, m in enumerate(m_values):
        fig = go.Figure()

        for i, (config, results_per_m, path) in enumerate(all_results):
            percent_zero_results = [(np.array(results) != 0).mean() * 100 for results in results_per_m[m]]
            iterations = list(np.arange(0, len(percent_zero_results) * config['delta'], config['delta']))
            T_value = extract_T_from_path(path)
            fig.add_trace(go.Scatter(
                x=iterations,
                y=percent_zero_results,
                mode='lines',
                name=f'T={T_value}: Dir {i + 1}'
            ))

            # Add 5% horizontal line
            fig.add_shape(
                type="line",
                x0=0,
                x1=iterations[-1] if iterations else 0,
                y0=5,
                y1=5,
                line=dict(color="black", width=2, dash="dash"),
                xref='x',
                yref='y'
            )

        fig.update_layout(
            title=f'Percentage of Results Not Equal to Zero per Iteration for m={m}',
            xaxis_title='Iteration',
            yaxis_title='Percentage of Results Not Equal to Zero (%)',
            legend_title='Directory',
            template='plotly_white'
        )

        # Display the plot in the appropriate column
        if idx % 2 == 0:
            col1.plotly_chart(fig)
        else:
            col2.plotly_chart(fig)







def main():

    show_first_part = False
    show_second_part = True


    st.set_page_config(
        page_title="Gregg's Research Simulations",
        page_icon=":material/compare_arrows:",
        layout="wide",
    )

    st.header(body="Gregg's research project")
    st.markdown("---")


    big_tabs = st.tabs(['First and Second Algorithms', 'Third Algorithms'])

    if show_first_part:
        with big_tabs[0]:

            file_path = '5x5_2000_samples_m=variable_T=2048.csv'

            simulate_for_given_path(file_path)

            file_paths = ['5x5_5000_samples_m=50_T=2048.csv', '5x5_5000_samples_m=100_T=2048.csv', '5x5_5000_samples_m=200_T=2048.csv',
                          '5x5_5000_samples_m=300_T=2048.csv', '5x5_5000_samples_m=400_T=2048.csv']

            tab_names = []

            for file_path in file_paths:
                num_series, _, T, m_value = split_filename(file_path)
                tab_names.append(m_value)

            for i, tab_name in enumerate(tab_names):
                tab_names[i] = f'm={tab_name}'
            tabs = st.tabs(tab_names)

            for i, file_path in enumerate(file_paths):
                with tabs[i]:
                    simulate_for_given_path(file_path)

            file_paths_2 = ['5x5_5000_samples_m=40_T=10240.csv', '5x5_5000_samples_m=100_T=10240.csv',
                            '5x5_5000_samples_m=400_T=10240.csv', '5x5_5000_samples_m=500_T=10240.csv',
                            '5x5_5000_samples_m=1000_T=10240.csv', '5x5_5000_samples_m=2000_T=10240.csv',]

            tab_names = []

            for file_path in file_paths_2:
                num_series, _, T, m_value = split_filename(file_path)
                tab_names.append(m_value)

            for i, tab_name in enumerate(tab_names):
                tab_names[i] = f'm={tab_name}'
            tabs = st.tabs(tab_names)

            for i, file_path in enumerate(file_paths_2):
                with tabs[i]:
                    simulate_for_given_path(file_path)

    if show_second_part:
        with big_tabs[1]:
            show_all_simulations_third_algorithm()






if __name__ == "__main__":
    main()
