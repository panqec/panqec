import numpy as np
from panqec.analysis import (
    get_subthreshold_fit_function, get_single_qubit_error_rate
)


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
        assert p_est == (n_results - 1)/n_results

        # Probability of each error type occuring should be based on count.
        p_x, p_x_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='X'
        )
        assert p_x == 1/n_results

        p_y, p_y_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='Y'
        )
        assert p_y == 2/n_results

        p_z, p_z_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='Z'
        )
        assert p_z == 3/n_results

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
        assert p_est == 6/7
        assert p_se == np.sqrt((6/7)*(1 - (6/7))/(7 + 1))

        # Probability of each error type occuring should be based on count.
        p_x, p_x_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='X'
        )
        assert p_x == 1/7
        assert p_x_se == np.sqrt(p_x*(1 - p_x)/(7 + 1))

        p_y, p_y_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='Y'
        )
        assert p_y == 2/7

        p_z, p_z_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='Z'
        )
        assert p_z == 3/7
