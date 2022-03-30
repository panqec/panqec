"""
Routines for plotting thing .

:Author:
    Eric Huang
"""
import json


def plot_export_json(plt, export_json):
    """Plot data in previously exported .json file."""
    with open(export_json) as f:
        data = json.load(f)

    plt.figure(figsize=(16, 9))
    for i, L in enumerate(data['parameters']['L_list']):
        plt.errorbar(
            data['parameters']['p_list'],
            data['statistics']['p_est'][i],
            yerr=data['statistics']['p_se'][i],
            label=f'L={L}'
        )
    plt.xlabel('Physical Error Rate', fontsize=20)
    plt.ylabel('Logical Error Rate', fontsize=20)
    time_elapsed = data['time']['time_elapsed']
    time_remaining = data['time']['time_remaining']
    eta = data['time']['eta']
    i_trial = data['parameters']['i_trial']
    n_trials = data['parameters']['n_trials']
    plt.title(
        f'Progress {100*(i_trial + 1)/n_trials}%, '
        f'{i_trial + 1} out of {n_trials}\n'
        f'Elapsed {time_elapsed}, Remaining {time_remaining}\n'
        f'ETA {eta}'
    )
    plt.legend()
