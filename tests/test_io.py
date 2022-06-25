from panqec.io import serialize_results, dump_results
import numpy as np
import datetime
import os
import pytest


@pytest.fixture
def serialized_results():
    i_trial = 2
    n_trials = 10

    L_list = np.array([1, 2])
    p_list = np.array([0.1, 0.2])
    L_repeats = np.array([1, 1])

    start_time = datetime.datetime(2021, 4, 6, 20, 41, 26, 887933)
    time_elapsed = datetime.timedelta(seconds=10699, microseconds=324064)

    time_remaining = datetime.timedelta(0)
    eta = datetime.datetime(2021, 4, 6, 20, 41, 26, 887933)

    np.random.seed(0)
    p_est = np.random.rand(L_list.shape[0], p_list.shape[0])
    p_se = np.random.rand(L_list.shape[0], p_list.shape[0])

    n_fail = np.random.randint(0, n_trials + 1, size=p_est.shape)
    n_try = np.ones(p_est.shape, dtype=int)*10

    effective_errors = [
        [
            [
                np.random.randint(0, 2, size=3*L_list[i_L]**3)
                for i_run in range(len(p_list))
            ]
            for i_p in range(len(p_list))
        ]
        for i_L in range(len(L_list))
    ]

    results_dict = serialize_results(
        i_trial, n_trials, L_list, p_list, L_repeats, start_time, time_elapsed,
        time_remaining, eta, p_est, p_se, n_fail, n_try, effective_errors
    )

    assert 'parameters' in results_dict.keys()
    assert 'time' in results_dict.keys()
    assert 'statistics' in results_dict.keys()
    assert 'results' in results_dict.keys()
    return results_dict


def test_dump_results(serialized_results, tmpdir):
    results_dict = serialized_results
    p = tmpdir.mkdir('sub')
    export_json = os.path.join(p, 'example.json')
    assert not os.path.exists(export_json)

    dump_results(export_json, results_dict)
    assert os.path.exists(export_json)
