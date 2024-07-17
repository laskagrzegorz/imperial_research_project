import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, probplot
from app_hawkes import *


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


def main():
    st.set_page_config(
        page_title="Gregg's Research Simulations",
        page_icon=":material/compare_arrows:",
        layout="wide",
    )

    st.write("ddddd111")

    st.header(body="Gregg's research project")
    st.markdown("---")

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







if __name__ == "__main__":
    main()
