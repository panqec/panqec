#!/usr/bin/env python3

import numpy as np
from pymatching import Matching
import os
from bn3d.bpauli import get_effective_error
from bn3d.tc3d import (
    get_vertex_Z_stabilisers,
    get_all_logicals,
)
from bn3d.io import (
    serialize_results, dump_results
)
import datetime
from bn3d.config import BN3D_DIR


# Parameters
# Change these as you wish.

# Subdirectory of BN3D_DIR to save results.
subdir = '00_tc3d'

# Description of this run
run_description = 'tc3d_no_px'

# Number of trials to repeat.
n_trials = 10000

# List of sizes to sample.
L_list = np.array([4, 6, 8, 10, 12])

# List of error rates to sample.
p_list = np.linspace(0.01, 0.04, 11)

# Number of times to repeat for each L.
L_repeats = np.ones_like(L_list).astype(int)

# Frequecy of plot update.
plot_frequency = 10

# Frequency of saving to file.
save_frequency = 100


# Make sure the output directory exists.
# The BN3D_DIR should have been set in the .env file.
output_dir = os.path.join(str(BN3D_DIR), subdir)
os.makedirs(output_dir, exist_ok=True)

file_name = (
    datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    + f'_{run_description}.json'
)
export_json = os.path.abspath(
    os.path.join(output_dir, file_name)
)

assert os.path.exists(os.path.dirname(export_json))


# Initialize result arrays.
n_fail = np.zeros((L_list.shape[0], p_list.shape[0]))
n_try = np.zeros((L_list.shape[0], p_list.shape[0]))


# Initialize list matching objects for each size.
stabilisers = []
matchers = []
for L in L_list:
    H_Z = get_vertex_Z_stabilisers(L)[:, 3*L**3:]
    stabilisers.append(H_Z)
    matchers.append(Matching(H_Z))

# Initialize storage of effective errors.
effective_errors: list = [
    [
        []
        for i_p in range(len(p_list))
    ]
    for i_L in range(len(L_list))
]

start_time = datetime.datetime.now()

# Run the trials
for i_trial in range(n_trials):
    for i_L, L in enumerate(L_list):
        H_Z = stabilisers[i_L]
        logicals = get_all_logicals(L)

        matching = matchers[i_L]
        for i_p, p in enumerate(p_list):
            for i_repeat in range(L_repeats[i_L]):

                # Initialize the total error.
                total_error = np.zeros(2*3*L**3, dtype=np.uint)

                # X noise.
                noise_X = np.random.binomial(1, p, H_Z.shape[1])

                # Decode point sector by matching Z vertex stabilisers.
                syndrome_Z = H_Z.dot(noise_X) % 2
                correction_X = matching.decode(
                    syndrome_Z, num_neighbours=None
                )
                total_error_X = (noise_X + correction_X) % 2
                total_error[:3*L**3] = total_error_X

                # Skip decoding the loop sector.

                # Compute the effective error.
                effective_error = get_effective_error(
                    logicals, total_error
                )
                effective_errors[i_L][i_p].append(effective_error)

                if np.any(effective_error == 1):
                    n_fail[i_L, i_p] += 1
                n_try[i_L, i_p] += 1

    # Plot at plot frequency and on last step.
    if i_trial % plot_frequency == 1 or i_trial == n_trials - 1:
        p_est = n_fail/n_try
        p_se = np.sqrt(p_est*(1 - p_est)/(n_try + 1))
        time_elapsed = datetime.datetime.now() - start_time
        time_remaining = (
            n_trials*time_elapsed/(i_trial + 1) - time_elapsed
        )
        eta = start_time + time_remaining

    # Save at given frequency and on last step.
    if i_trial % save_frequency == 1 or i_trial == n_trials - 1:
        results_dict = serialize_results(
            i_trial, n_trials, L_list, p_list, L_repeats,
            start_time, time_elapsed, time_remaining, eta,
            p_est, p_se, n_fail, n_try, effective_errors
        )
        dump_results(export_json, results_dict)
