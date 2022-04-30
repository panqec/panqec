"""
Routines for input and output.

This includes routines to produce, merge and analyse data produced by
simulations.

:Author:
    Eric Huang
"""
import numpy as np
import datetime
import os
import json
from .bpauli import bvectors_to_ints
from .utils import sizeof_fmt


def serialize_results(
    i_trial: int, n_trials: int,
    L_list: np.ndarray, p_list: np.ndarray, L_repeats: np.ndarray,
    start_time: datetime.datetime, time_elapsed: datetime.timedelta,
    time_remaining: datetime.timedelta, eta: datetime.datetime,
    p_est: np.ndarray, p_se: np.ndarray,
    n_fail: np.ndarray, n_try: np.ndarray,
    effective_errors: list
) -> dict:
    """Convert results to dict."""
    return {
        'parameters': {
            'i_trial': i_trial,
            'n_trials': n_trials,
            'L_list': L_list.tolist(),
            'p_list': p_list.tolist(),
            'L_repeats': L_repeats.tolist(),
            'n_list': [int(3*L**3) for L in L_list],
        },
        'time': {
            'start_time': str(start_time),
            'time_elapsed': str(time_elapsed),
            'time_remaining': str(time_remaining),
            'eta': str(eta),
        },
        'statistics': {
            'p_est': p_est.tolist(),
            'p_se': p_se.tolist(),
        },
        'results': {
            'n_fail': n_fail.tolist(),
            'n_try': n_try.tolist(),
            'effective_errors': [
                [
                    bvectors_to_ints(effective_errors[i_L][i_p])
                    for i_p in range(len(p_list))
                ]
                for i_L in range(len(L_list))
            ],
        },
    }


def dump_results(export_json: str, results_dict: dict, verbose: bool = True):
    """Save results dict to json file."""
    with open(export_json, 'w') as f:
        json.dump(results_dict, f)
    if verbose:
        print(
            f'Results written to {export_json} '
            f'({sizeof_fmt(os.path.getsize(export_json))})'
        )
