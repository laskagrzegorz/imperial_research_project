from time_series_functions import *

series_3x3_show = True
series_5x5_show = False
series_7x7_show = False

show_t_different = False

# Shift of the processes (for second part)
p = 3

### Define Coefficient Matricies ###

## 3x3 Var(p) processes ##

A_list_positive_2 = [np.array([
    [0.5, 0.5, 0.1],
    [0.25, 0.5, 0.0],
    [0.25, 0.0, 0.5]
])]

A_list_positive_1 = [np.array([
    [0.5, 0.0, 0.25],
    [0.0, 0.7, 0.0],
    [0.25, 0.0, 0.5]
])]

A_list_independent = [np.array([
    [0.9, 0.0, 0.0],
    [0.0, 0.7, 0.0],
    [0.0, 0.0, 0.7]
])]

A_list_negative_1 = [np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.5, -0.25],
    [0.0, -0.25, 0.5]
])]

A_list_mix_2 = [np.array([
    [0.5, 0.0, 0.2],
    [0.0, 0.5, -0.25],
    [0.2, -0.25, 0.5]
])]

A_list_mix_3 = [np.array([
    [0.5, 0.2, 0.01],
    [0.2, 0.5, -0.25],
    [0.01, -0.25, 0.5]
])]

All_A_list_3x3 = [A_list_positive_2, A_list_positive_1, A_list_independent,
                  A_list_negative_1, A_list_mix_2, A_list_mix_3]


All_A_list_3x3_var_p = make_var_1_to_var_p(All_A_list_3x3, p)

## 5x5 Var(p) processes ##

A_list_5x5_positive_5 = [np.array([
    [0.3, 0.2, 0.4, 0.0, 0.0],
    [0.2, 0.3, 0.0, 0.2, 0.0],
    [0.2, 0.0, 0.3, 0.2, 0.2],
    [0.0, 0.2, 0.2, 0.3, 0.0],
    [0.0, 0.0, 0.2, 0.0, 0.5]
])]

A_list_5x5_mix_10 = [np.array([
    [0.3, -0.2, 0.4, -0.2, 0.2],
    [0.2, 0.3, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.3, -0.2, 0.2],
    [-0.2, 0.2, -0.2, 0.3, 0.2],
    [0.2, -0.2, 0.2, 0.2, 0.5]
])]

A_list_5x5_independent = [np.array([
    [0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5]
])]

All_A_list_5x5 = [A_list_5x5_positive_5, A_list_5x5_mix_10, A_list_5x5_independent]

All_A_list_5x5_var_p = make_var_1_to_var_p(All_A_list_5x5, p)


## 7x7 Var(p) processes ##


A_list_7x7_mix_21_1 = [np.array([
    [ 0.3, -0.0,  0.0,  0.0, -0.0,  0.0, -0.0],
    [-0.3,  0.4, -0.0, -0.0,  0.0, -0.0,  0.0],
    [ 0.2, -0.3,  0.4,  0.0, -0.0,  0.0, -0.0],
    [ 0.3, -0.3,  0.3,  0.4, -0.0,  0.0, -0.0],
    [-0.4,  0.3, -0.3, -0.3,  0.4, -0.0,  0.0],
    [ 0.2, -0.2,  0.3,  0.2, -0.3,  0.4, -0.0],
    [-0.3,  0.4, -0.2, -0.3,  0.3, -0.4,  0.3]
])]

A_list_7x7_mix_10 = [np.array([
    [ 0.3,  0.25,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.3,  0.25,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.3,  0.25,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.3,  0.25,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.3,  0.25,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.3,  0.25],
    [ 0.25, 0.0,  0.0,  0.0,  0.0,  0.0,  0.3]
])]

A_list_7x7_mix_21 = [np.array([
    [ 0.25,  0.10, -0.10,  0.05, -0.05,  0.10, -0.15],
    [ 0.10,  0.20,  0.05, -0.10,  0.05,  0.15, -0.10],
    [-0.10,  0.05,  0.30,  0.10, -0.15,  0.05,  0.10],
    [ 0.05, -0.10,  0.10,  0.25,  0.10, -0.15,  0.05],
    [-0.05,  0.05, -0.15,  0.10,  0.20,  0.10, -0.10],
    [ 0.10,  0.15,  0.05, -0.15,  0.10,  0.25,  0.05],
    [-0.15, -0.10,  0.10,  0.05, -0.10,  0.05,  0.30]
])]

A_list_7x7_independent = [np.array([
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
])]

All_A_list_7x7 = [A_list_7x7_mix_21_1, A_list_7x7_mix_10, A_list_7x7_mix_21, A_list_7x7_independent]

All_A_list_7x7_var_p = make_var_1_to_var_p(All_A_list_7x7, p)

### Helper Functions ###

def all_sdf_estimators_plot(A_list, T, x):

    tabs = st.tabs(["ALL normal scale", 'ALL log scale', "Multi-sinusoidal normal scale", 'Multi-sinusoidal log scale',
                    'Freq-averaged normal scale', 'Freq-averaged log scale'])

    # Theory sdf
    _, true_sdf = true_sdf_var_process(A_list, np.eye(A_list[0].shape[0]), T // 2)
    true_sdf *= 10

    # Basic periodogram
    frequencies, periodogram_sdf = calculate_periodogram(x, taper=None)

    # Cosine taper estimator
    p = 0.9
    cos_taper = cosine_tapering_window(T, p)
    _, cos_taper_sdf = calculate_periodogram(x, taper=cos_taper)

    # Sinusoidal multi taper estimator
    W = 0.05
    num_tapers = choose_num_tapers(T, W)
    _, milti_sine_taper_sdf = sinusoidal_multitaper_sdf_matrix(x, num_tapers=num_tapers)

    # Frequency averaged periodogram estimator
    m = 100
    _, freq_avg_periodogram_sdf, _ = calculate_freq_avg_periodogram(x, m)

    # Off the shelf estimator
    _, package_sdf = estimate_cross_sdf_matrix(x)

    # Different estimators plot
    all_estimators = [periodogram_sdf, cos_taper_sdf, milti_sine_taper_sdf, freq_avg_periodogram_sdf, package_sdf]
    all_estimators_names = ["Periodogram", f"Cosine Taper ({p})", f"Multi-Sinusoidal (W={num_tapers})",
                            f"Freq-avg Periodogram ({m})", "package"]

    with tabs[0]:
        st.subheader(body=f"Empirical SDF vs True SDF on normal scale")
        st.plotly_chart(
            figure_or_data=plot_sdf_with_theoretical(frequencies, all_estimators, all_estimators_names,
                                                     true_sdf, log_scale=False),
            use_container_width=True,
        )

    with tabs[1]:
        st.subheader(body=f"Empirical SDF vs True SDF on log scale")
        st.plotly_chart(
            figure_or_data=plot_sdf_with_theoretical(frequencies, all_estimators, all_estimators_names,
                                                     true_sdf, log_scale=True),
            use_container_width=True,
        )

    # Sinusoidal
    W_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    multi_sine_estimators = []
    multi_sine_estimators_names = []

    for W_i in W_list:
        num_tapers = choose_num_tapers(T, W_i)
        multi_sine_estimators_names.append(f'Multi-Sinusoidal (W={W_i})')
        _, milti_sine_taper_sdf_i = sinusoidal_multitaper_sdf_matrix(x, num_tapers=num_tapers)
        multi_sine_estimators.append(milti_sine_taper_sdf_i)

    with tabs[2]:
        st.subheader(body=f"Multi-Sinusoidal SDF vs True SDF on normal scale")
        st.plotly_chart(
            figure_or_data=plot_sdf_with_theoretical(frequencies, multi_sine_estimators, multi_sine_estimators_names,
                                                     true_sdf, log_scale=False),
            use_container_width=True,
        )

    with tabs[3]:
        st.subheader(body=f"Multi-Sinusoidal SDF vs True SDF on log scale")
        st.plotly_chart(
            figure_or_data=plot_sdf_with_theoretical(frequencies, multi_sine_estimators, multi_sine_estimators_names,
                                                     true_sdf, log_scale=True),
            use_container_width=True,
        )

    # Frequency-averaged periodogram
    m_list = [15, 30, 50, 100, 120, 150, 200]
    freq_avg_estimators = []
    freq_avg_estimators_names = []

    for m_i in m_list:
        freq_avg_estimators_names.append(f'Freq-averaged m={m_i}')
        _, freq_avg_periodogram_sdf_i, _ = calculate_freq_avg_periodogram(x, m_i)
        freq_avg_estimators.append(freq_avg_periodogram_sdf_i)

    with tabs[4]:
        st.subheader(body=f"Frequency-averaged SDF vs True SDF on normal scale")
        st.plotly_chart(
            figure_or_data=plot_sdf_with_theoretical(frequencies, freq_avg_estimators, freq_avg_estimators_names,
                                                     true_sdf, log_scale=False),
            use_container_width=True,
        )

    with tabs[5]:
        st.subheader(body=f"Frequency-averaged SDF vs True SDF on log scale")
        st.plotly_chart(
            figure_or_data=plot_sdf_with_theoretical(frequencies, freq_avg_estimators, freq_avg_estimators_names,
                                                     true_sdf, log_scale=True),
            use_container_width=True,
        )


def highlight_rows(df, whole_row=False, alpha=0.05):
    # Define a function to apply styles
    if whole_row:
        def highlight(row):
            # Compare values in Column A and Column B
            return ['background-color: yellow' if row['Z_i'] < row[f'C_k({alpha})'] else '' for _ in row]
    else:
        def highlight(row):
            # Compare values in 'Z_i' and 'C_i(0.05)'
            smaller_value_col = 'Z_i' if row['Z_i'] < row[f'C_k({alpha})'] else f'C_k({alpha})'
            return ['background-color: yellow' if col == smaller_value_col else '' for col in row.index]

    # Apply the style
    styled_df = df.style.apply(highlight, axis=1)
    return styled_df


def run_algorithms_given_m(x, m, alpha, T, C, D, m2, m1, num_iter, A_list):
    matsuda_algorithm_table, matsuda_edges_left = backward_stepwise_selection(x, m, T, C, D,
                                                                              m1, m2, num_iter, alpha)
    walden_algorithm_table, walden_edges_left = walden_one_step_algorithm(x, m, T, C, D,
                                                                          m1, m2, num_iter, alpha)
    st.markdown("---")
    cols_algorithm = st.columns(3)
    true_edges = true_non_missing_edges(A_list)
    with cols_algorithm[0]:
        st.subheader("Matsuda Algorithm")
        st.dataframe(matsuda_algorithm_table.style.highlight_min(axis=0))
        st.subheader("Predicted Graph Structure")
        plot_graph(x.shape[0], matsuda_edges_left, true_edges)

        st.subheader("True graph")
        plot_graph(x.shape[0], true_edges, true_edges)
    with cols_algorithm[1]:
        st.subheader("Walden Algorithm")
        st.dataframe(highlight_rows(walden_algorithm_table, alpha=0.05))
        st.subheader("Predicted Graph Structure")
        plot_graph(x.shape[0], walden_edges_left, true_edges)


def do_analysis(A_list, m_range, T = 1024, series_plot=False, sdf_plot=False, cvll_plot=True):

    ### To SET ###
    test_m = False

    # Constants
    C = 0.617
    D = 0.446
    m1 = 0
    m2 = 1
    num_iter = 10
    alpha = 0.05

    # Generate time series
    x = generate_var_process(A_list, T, 1000, seed=100)
    x = x - np.mean(x, axis=1, keepdims=True)

    if series_plot:
        st.subheader(body=f"Realisation of VAR({len(A_list)}) process")
        st.plotly_chart(
            figure_or_data=plot_time_series(x=x),
            use_container_width=True,
        )

    if sdf_plot:
        with st.expander("Plots of Different SDF Estimators"):
            all_sdf_estimators_plot(A_list, T, x)

    ### Hypothesis testing ###

    cvll_values, best_m = find_best_m(x, m_range)

    if cvll_plot:
        st.subheader(body=f"Best parameter M based on CVLL criteria defined in Y. Matsuda (2006)")
        cols_cvll = st.columns(2)
        with cols_cvll[0]:

            st.plotly_chart(
                figure_or_data=plot_cvll_criterion(m_range, cvll_values, best_m),
                use_container_width=True,
            )

        with cols_cvll[1]:
            latex_equation = r"""
            \begin{gathered}
            \operatorname{CVLL}(m):=\frac{1}{n} \sum_{j=1}^{[n / 2]} \operatorname{tr}\left\{I\left(\lambda_j\right) \hat{f}_{-j}^{-1}\left(m, \lambda_j\right)\right\}+\log \operatorname{det}\left\{\hat{f}_{-j}\left(m, \lambda_j\right)\right\}, \\
            \hat{f}_{-j}\left(m, \lambda_j\right):=\left(\sum_{k=-m / 2, k \neq 0}^{m / 2} w_k\right)^{-1} \sum_{k=-m / 2, k \neq 0}^{m / 2} w_k I\left(\lambda_{j+k}\right) \quad(j=1, \ldots,[n / 2]) .
            \end{gathered}
            """
            st.header("")
            st.markdown("<p style='font-size:20px;'>We choose m that minimises:</p>", unsafe_allow_html=True)
            st.latex(latex_equation)
            st.markdown("<p style='font-size:20px;'>We achieve that by setting w_0 = 0 and re-normalising the weights.</p>",
                        unsafe_allow_html=True)

            st.markdown(
                f"<p style='font-size:20px;'>Minimal CVLL is 1.5121 for m = {best_m}.</p>",
                unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:20px; color:red;'>Imortantly we cannot get rid of the \"edge\" frequencies "
                "as this formula requires all the positive Fourier frequencies</p>",
                unsafe_allow_html=True)

    # True M
    run_algorithms_given_m(x, best_m, alpha, T, C, D,
                           m2, m1, num_iter, A_list)


    if test_m:
        # M-50
        m_minus_50 = best_m - 50
        st.header(f"M-50 = {m_minus_50}")

        run_algorithms_given_m(x, m_minus_50, alpha, T, C, D,
                               m2, m1, num_iter, A_list)
        # M+50
        m_plus_50 = best_m + 50
        st.header(f"M+50 = {m_plus_50}")

        run_algorithms_given_m(x, m_plus_50, alpha, T, C, D,
                               m2, m1, num_iter, A_list)


def insert_page(A_list, m_range, T=1024, series_plot=False, sdf_plot=False, cvll_plot=True):
    st.subheader(f"The VAR({len(A_list)}) process is given by equation:")
    generate_latex_equation(A_list)
    check_series(A_list)
    do_analysis(A_list, m_range, T=T,  series_plot=series_plot, sdf_plot=sdf_plot, cvll_plot=cvll_plot)


def main():
    m_range = np.arange(30, 300, 2)
    m_range_high = np.arange(300, 800, 6)

    ### Page code ###

    st.set_page_config(
        page_title="Gregg's Research",
        page_icon=":material/compare_arrows:",
        layout="wide",
    )

    st.header(body="Gregg's research project")
    st.markdown("---")

    tabs_different_length = st.tabs(["T=1024", "T=2048"])

    with tabs_different_length[0]:

        tabs_different_size = st.tabs(['VAR(1) 3x3', 'VAR(3) 3x3', 'VAR(1) 5x5', 'VAR(3) 5x5', 'VAR(1) 7x7', 'VAR(3) 7x7'])

        ### 3x3 ###
        with tabs_different_size[0]:
            if series_3x3_show:
                tabs = st.tabs(["3x3: 2+", "3x3: 1+", "3x3: 0", "3x3: 1-", "3x3: 2+-", "3x3: 3+-"])
                for i, A_list in enumerate(All_A_list_3x3):
                    with tabs[i]:
                        insert_page(A_list, m_range, series_plot=False, sdf_plot=False, cvll_plot=True)


                        st.markdown("---")
                        st.header(f"T={10240}")

                        insert_page(A_list, m_range_high, T=10240)

        with tabs_different_size[1]:
            if series_3x3_show:
                tabs = st.tabs(["3x3: 2+", "3x3: 1+", "3x3: 0", "3x3: 1-", "3x3: 2+-", "3x3: 3+-"])
                for i, A_list in enumerate(All_A_list_3x3_var_p):
                    with tabs[i]:
                        insert_page(A_list, m_range)

                        st.header(f"T={10240}")

                        insert_page(A_list, m_range_high, T=10240)

        ### 5x5 ###
        with tabs_different_size[2]:
            if series_5x5_show:
                tabs = st.tabs(["5x5: 5+", "5x5: 10+-", "5x5: 0"])
                for i, A_list in enumerate(All_A_list_5x5):
                    with tabs[i]:
                        insert_page(A_list, m_range)

        with tabs_different_size[3]:
            if series_5x5_show:
                tabs = st.tabs(["5x5: 5+", "5x5: 10+-", "5x5: 0"])
                for i, A_list in enumerate(All_A_list_5x5_var_p):
                    with tabs[i]:
                        insert_page(A_list, m_range)

        ### 7x7 ###
        with tabs_different_size[4]:
            if series_5x5_show:
                tabs = st.tabs(["7x7: 21+-", "7x7: 10+-", "7x7: 21+-", "7x7: 0"])
                for i, A_list in enumerate(All_A_list_7x7):
                    with tabs[i]:
                        insert_page(A_list, m_range)

        with tabs_different_size[5]:
            if series_7x7_show:
                tabs = st.tabs(["7x7: 21+-", "7x7: 10+-", "7x7: 21+-", "7x7: 0"])
                for i, A_list in enumerate(All_A_list_7x7_var_p):
                    with tabs[i]:
                        insert_page(A_list, m_range)

    if show_t_different:
        with tabs_different_length[1]:

            tabs_different_size = st.tabs(
                ['VAR(1) 3x3', 'VAR(3) 3x3', 'VAR(1) 5x5', 'VAR(3) 5x5', 'VAR(1) 7x7', 'VAR(3) 7x7'])

            ### 3x3 ###
            with tabs_different_size[0]:
                if series_3x3_show:
                    tabs = st.tabs(["3x3: 2+", "3x3: 1+", "3x3: 0", "3x3: 1-", "3x3: 2+-", "3x3: 3+-"])
                    for i, A_list in enumerate(All_A_list_3x3):
                        with tabs[i]:
                            insert_page(A_list,m_range, T=10240)

            with tabs_different_size[1]:
                if series_3x3_show:
                    tabs = st.tabs(["3x3: 2+", "3x3: 1+", "3x3: 0", "3x3: 1-", "3x3: 2+-", "3x3: 3+-"])
                    for i, A_list in enumerate(All_A_list_3x3_var_p):
                        with tabs[i]:
                            insert_page(A_list,m_range, T=10240)


if __name__ == "__main__":
    main()

