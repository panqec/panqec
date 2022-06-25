import numpy as np
from panqec.analysis import get_subthreshold_fit_function


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
