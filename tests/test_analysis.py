import os
import numpy as np
from panqec.analysis import (
    get_subthreshold_fit_function, get_single_qubit_error_rate, Analysis
)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def test_subthreshold_cubic_fit_function():
    C_0, C_1, C_2, C_3 = 0, 1, 2, 1
    log_p_L_th, log_p_th = np.log(0.5), np.log(0.5)

    subthreshold_fit_function = get_subthreshold_fit_function(order=3)

    p = 0.5
    L = 10
    log_p_L = subthreshold_fit_function(
        (np.log(p), L),
        log_p_L_th, log_p_th, C_0, C_1, C_2, C_3
    )
    assert log_p_L == np.log(0.5)


class TestGetSingleQubitErrorRates:

    def test_trivial(self):
        effective_error_list = []
        p_est, p_se = get_single_qubit_error_rate(effective_error_list)
        assert np.isnan(p_est) and np.isnan(p_se)

    def test_simple_example(self):
        effective_error_list = [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ]

        n_results = len(effective_error_list)
        assert n_results == 7

        # Rate of any error occuring should be 3/4.
        p_est, p_se = get_single_qubit_error_rate(effective_error_list, i=0)
        assert np.isclose(p_est, (n_results - 1)/n_results)

        # Probability of each error type occuring should be based on count.
        p_x, p_x_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='X'
        )
        assert np.isclose(p_x, 1/n_results)

        p_y, p_y_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='Y'
        )
        assert np.isclose(p_y, 2/n_results)

        p_z, p_z_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='Z'
        )
        assert np.isclose(p_z, 3/n_results)

    def test_simple_example_on_2nd_qubit(self):
        effective_error_list = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
        ]

        # Rate of any error occuring should be 3/4.
        p_est, p_se = get_single_qubit_error_rate(effective_error_list, i=1)
        assert np.isclose(p_est, 6/7)
        assert np.isclose(p_se, np.sqrt((6/7)*(1 - (6/7))/(7 + 1)))

        # Probability of each error type occuring should be based on count.
        p_x, p_x_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='X'
        )
        assert np.isclose(p_x, 1/7)
        assert np.isclose(p_x_se, np.sqrt(p_x*(1 - p_x)/(7 + 1)))

        p_y, p_y_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='Y'
        )
        assert np.isclose(p_y, 2/7)

        p_z, p_z_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='Z'
        )
        assert np.isclose(p_z, 3/7)


class TestAnalysis:

    def test_analyse_toric_2d_results(self):
        results_path = os.path.join(DATA_DIR, 'toric')
        assert os.path.exists(results_path)
        analysis = Analysis(results_path)
        analysis.analyze()
        assert analysis.results.shape == (30, 21)
        assert set(analysis.results.columns) == set([
            'size', 'code', 'n', 'k', 'd', 'error_model', 'decoder',
            'probability', 'wall_time', 'n_samp', 'effective_error', 'success',
            'codespace', 'bias', 'results_file', 'p_est', 'p_se', 'p_word_est',
            'p_word_se', 'single_qubit_p_est', 'single_qubit_p_se'
        ])
