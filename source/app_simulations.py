import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, probplot
from app_hawkes import *
import os
import json
import pickle
from collections import defaultdict


### First and second algorithm functions ###

def categorize_files_first_algorithm(files, n):
    categories = defaultdict(lambda: defaultdict(list))
    n_str = f"{n}x{n}"

    for file_name in files:
        if n_str not in file_name:
            continue

        parts = file_name.split('_')

        # Extracting m and T values
        m_part = [part for part in parts if part.startswith('m=')][0]
        T_part = [part for part in parts if part.startswith('T=')][0]
        type_part = 'independentcon' if 'independentcon' in file_name else 'independent'

        m_value = m_part.split('=')[1]
        T_value = T_part.split('=')[1]

        # Constructing dictionary keys
        m_key = m_value
        T_key = f"T={T_value}"

        # Adding file to appropriate category
        categories[type_part][T_key].append({m_key: file_name})

    return dict(categories)


def get_all_files(directory):
    # List all files in the given directory
    file_list = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return file_list


def filter_files_with_m_none(organized_files_independent):
    filtered_files = {}

    for T_key, m_files in organized_files_independent.items():
        for m_file_dict in m_files:
            for m_key, file_path in m_file_dict.items():
                if m_key == 'None':
                    filtered_files[T_key] = file_path
                    break

    return filtered_files



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


def split_filename(path):
    # Extract the filename from the path
    filename = path.split("/")[-1]

    # Split the filename on "_"
    parts = filename.split("_")

    # Extract the number of time series from the first part
    num_time_series = int(parts[0].split("x")[0])

    # Extract the number of simulations from the second part
    num_simulations = int(parts[1].split("=")[1])

    # Extract T value from the second to last part
    T = int(parts[-2].split("=")[1])

    # Extract m value from the third to last part
    m_value_part = parts[-3].split("=")[1]
    if m_value_part == 'None':
        m_value = None
    else:
        m_value = int(m_value_part)

    return num_time_series, num_simulations, T, m_value


def show_one_test_stat_given_m_none(T, m_values, test_stats):
    cols = st.columns(2)
    with cols[0]:
        st.write("Number of simulations: ", len(test_stats))
        st.write(f'Number of samples in each time series: ', T)

        st.write("Variable parameter m with average equal to ", m_values.mean())

        # Calculate the skewness
        skewness = stats.skew(test_stats)

        # Compare with standard normal distribution
        if skewness == 0:
            st.write("The distribution is symmetrical (similar to a standard normal distribution).")
        elif skewness > 0:
            st.write(f'Skewness of the data: ', np.round(skewness, 2),
                     "The distribution is positively skewed (right tail is longer).")
        else:
            st.write(f'Skewness of the data: ', np.round(skewness, 2),
                     "The distribution is negatively skewed (left tail is longer).")

        percentile_95 = norm.ppf(0.95, loc=0, scale=1)

        st.write(f'The power of the test (percentage of hypothesis rejected): ',
                 np.sum(test_stats > percentile_95)/len(test_stats))


    with cols[1]:
        pass
        # st.plotly_chart(
        #     figure_or_data=plot_histogram_with_mean(m_values),
        #     use_container_width=True,
        # )
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


def show_one_test_stat_given_m(test_stats):
    # # Calculate the skewness
    # skewness = stats.skew(test_stats)
    #
    # # Print the skewness
    # st.write(f'Skewness of the data: ', np.round(skewness, 2))
    #
    # # Compare with standard normal distribution
    # if skewness == 0:
    #     st.write("The distribution is symmetrical (similar to a standard normal distribution).")
    # elif skewness > 0:
    #     st.write("The distribution is positively skewed (right tail is longer).")
    # else:
    #     st.write("The distribution is negatively skewed (left tail is longer).")

    percentile_95 = norm.ppf(0.95, loc=0, scale=1)

    st.write(f'The power of the test (percentage of hypothesis rejected): ',
             np.sum(test_stats > percentile_95) / len(test_stats))

    st.plotly_chart(
        figure_or_data=plot_qq_plot(test_stats),
        use_container_width=True,
    )


    st.plotly_chart(
        figure_or_data=plot_histogram_with_gaussian(test_stats),
        use_container_width=True,
    )




def simulate_for_given_path_fix_m(m, file_path):

    num_series, _, T, m_value = split_filename(file_path)

    df = pd.read_csv(file_path)

    def extract_real(complex_str):
        return complex(complex_str).real

    # Apply the function to the relevant columns
    df['Test_Stat_Mirror'] = df['Test_Stat_Mirror'].apply(extract_real)
    df['Test_Stat_Gregg'] = df['Test_Stat_Gregg'].apply(extract_real)

    # Extract the columns to variables
    test_stats_mirror = df['Test_Stat_Mirror']
    test_stats_gregg = df['Test_Stat_Gregg']

    with st.expander(f"M parameter = {m}", expanded=True):

        cols = st.columns(2)

        with cols[0]:
            st.subheader('Test_Stat_Mirror')
            show_one_test_stat_given_m(test_stats_mirror)

        with cols[1]:
            st.subheader('test_stats_Gregg')
            show_one_test_stat_given_m(test_stats_gregg)










def simulate_for_given_path_variable_m(file_path):

    num_series, _, T, m_value = split_filename(file_path)

    df = pd.read_csv(file_path)

    def extract_real(complex_str):
        return complex(complex_str).real

    # Apply the function to the relevant columns
    df['Test_Stat_Mirror'] = df['Test_Stat_Mirror'].apply(extract_real)
    df['Test_Stat_Gregg'] = df['Test_Stat_Gregg'].apply(extract_real)

    # Extract the columns to variables
    test_stats_mirror = df['Test_Stat_Mirror']
    test_stats_gregg = df['Test_Stat_Gregg']
    m_values = df['m']

    st.subheader(f"Test Statistics for missing edge of {num_series}x{num_series} Var(1)")

    with st.expander("Test Statistics of Mirror Frequency Average", expanded=True):

        show_one_test_stat_given_m_none(T, m_values, test_stats_mirror)

    with st.expander("Test Statistics of Clamped Frequency Average", expanded=True):
        show_one_test_stat_given_m_none(T, m_values, test_stats_gregg)


def simulation_pre_T_and_m_FS_algorithm(data_dict):
    def parse_key(key):
        """Parse the 'm' key and return None or the integer value."""
        if key == 'None':
            return None
        return int(key)

    def get_sorted_paths(data):
        """Get a sorted list of (m_value, path) tuples from the given data."""
        parsed_data = []
        for entry in data:
            for m_key, path in entry.items():
                m_value = parse_key(m_key)
                if m_value is not None:  # Exclude None values
                    parsed_data.append((m_value, path))
        # Sort the data by the m_value
        parsed_data.sort(key=lambda x: x[0])
        return parsed_data

    # Streamlit app
    st.subheader("Simulation by m value")

    # Create tabs for each T=... key
    tabs = st.tabs(list(data_dict.keys()))

    for i, T_key in enumerate(data_dict.keys()):
        with tabs[i]:
            sorted_paths = get_sorted_paths(data_dict[T_key])
            for m_value, path in sorted_paths:
                simulate_for_given_path_fix_m(m_value, path)


def analysis_by_type_FS_algorithm(organized_files_selected):
    files_m_none = filter_files_with_m_none(organized_files_selected)
    tabs = st.tabs(list(files_m_none.keys()))
    for i, path in enumerate(files_m_none.values()):
        with tabs[i]:
            simulate_for_given_path_variable_m(path)
    st.markdown("---")
    simulation_pre_T_and_m_FS_algorithm(organized_files_selected)



def show_all_simulations_first_second_algorithm(directory_path='../simulations_data'):

    # Get all files from the directory
    file_list = get_all_files(directory_path)

    # Categorize the files
    organized_files_5 = categorize_files_first_algorithm(file_list, 5)
    organized_files_7 = categorize_files_first_algorithm(file_list, 7)

    tabs = st.tabs(['5x5 independent','5x5 conditional independent','7x7 conditional independent'])

    with tabs[0]:
        organized_files_selected = organized_files_5['independent']
        analysis_by_type_FS_algorithm(organized_files_selected)
    with tabs[1]:
        organized_files_selected = organized_files_5['independentcon']
        analysis_by_type_FS_algorithm(organized_files_selected)
    with tabs[2]:
        organized_files_selected = organized_files_7['independentcon']
        analysis_by_type_FS_algorithm(organized_files_selected)






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
    categories = {}

    for dir in directories:
        # Extract the size from the directory name
        size_part = dir.split('size=')[1].split('_')[0]

        # Determine the base category
        if 'independent' in dir:
            base_category = 'independent'
        elif 'one_add' in dir:
            base_category = 'add_one'
        elif 'complex_2' in dir:
            base_category = 'complex_2'
        elif 'complex' in dir:
            base_category = 'complex'
        else:
            base_category = 'other'

        # Create the final category name
        category = f"{base_category}_{size_part}"

        # Add the directory to the appropriate category
        if category not in categories:
            categories[category] = []
        categories[category].append(dir)

    # Sort the categories dictionary by keys
    sorted_categories = {k: categories[k] for k in sorted(categories)}

    return sorted_categories


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


def estimate_m_for_target_percent(loaded_config, results_per_m, target_percent=5):
    """
    Estimate the value of m that would give exactly the target percentage of zeros in percent_zero_results.

    Parameters:
    loaded_config (dict): The loaded configuration containing m_list.
    results_per_m (dict): Dictionary where keys are values of m and values are lists of results.
    target_percent (float): The target percentage of zeros. Default is 5.

    Returns:
    float: The estimated value of m that would give exactly the target percentage of zeros or None if out of range.
    """
    m_list = loaded_config['m_list']
    percent_zero_results = [(np.array(results[0]) != 0).mean() * 100 for m, results in results_per_m.items()]

    # Convert to numpy arrays for easier manipulation
    m_array = np.array(m_list)
    percent_zero_array = np.array(percent_zero_results)

    # Sort the values for interpolation
    sorted_indices = np.argsort(percent_zero_array)
    m_array = m_array[sorted_indices]
    percent_zero_array = percent_zero_array[sorted_indices]

    # Check if the target_percent is within the range of percent_zero_results
    if target_percent < percent_zero_array.min() or target_percent > percent_zero_array.max():
        return None

    # Find the closest points around target_percent for linear interpolation
    for i in range(len(percent_zero_array) - 1):
        if percent_zero_array[i] <= target_percent <= percent_zero_array[i + 1]:
            x1, y1 = m_array[i], percent_zero_array[i]
            x2, y2 = m_array[i + 1], percent_zero_array[i + 1]
            break

    # Perform linear interpolation
    estimated_m = x1 + (target_percent - y1) * (x2 - x1) / (y2 - y1)

    return np.round(estimated_m, decimals=1)


def show_one_path_third_algorithm(path):

    loaded_config, results_per_m = get_simulation_data_third_algorithm(path)

    estimate_m = estimate_m_for_target_percent(loaded_config, results_per_m)
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

    st.markdown(f'Estimated m giving 5% power is **{estimate_m}** (using linear interpolation).')

    st.markdown(f'---')

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
    tab_names = [f"{name}" for name in categorised_paths.keys()]

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

    show_first_part = True
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
            show_all_simulations_first_second_algorithm()


    if show_second_part:
        with big_tabs[1]:
            show_all_simulations_third_algorithm()






if __name__ == "__main__":
    main()
