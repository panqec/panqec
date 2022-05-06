import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def get_wilson_loop_data(estimates, max_tau=10, L=15, eps=1e-5):
    probas = estimates['p'].unique()
    temperatures = estimates['temperature'].unique()

    filt_estimates = estimates[(estimates['tau'] == max_tau) &
                               (estimates['L_x'] == L)]

    n_wilson_loops = len(filt_estimates['Wilson Loop_estimate'].to_numpy()[0])

    perimeters = 4 * (np.arange(2, n_wilson_loops+2))
    log_wl = np.zeros((len(probas), len(temperatures), n_wilson_loops))

    for i_p, p_val in enumerate(probas):
        for i_T, T_val in enumerate(temperatures):
            curr_estimates = filt_estimates[
                (filt_estimates['p'] == p_val) &
                (filt_estimates['temperature'] == T_val)
            ]
            wilson_loops = curr_estimates['Wilson Loop_estimate'].to_numpy()[0]

            log_wl[i_p, i_T] = -np.log(np.abs(wilson_loops)+eps) / perimeters

    return perimeters, log_wl


def wilson_loop_ansatz(L, a, b, c):
    return a*L + b + c*np.log(L)


def get_wilson_loop_ansatz_parameters(
    estimates, max_tau=10, L=15, eps=1e-5, start=0
):
    perimeters, log_wl = get_wilson_loop_data(
        estimates, max_tau=max_tau, L=L, eps=eps
    )
    temperatures = estimates['temperature'].unique()
    probas = estimates['p'].unique()

    list_params = np.zeros((len(probas), len(temperatures), 3))
    for i_p in range(len(probas)):
        for i_T in range(len(temperatures)):
            a, b, c = curve_fit(
                wilson_loop_ansatz, perimeters[start:],
                log_wl[i_p, i_T][start:], p0=[0.1, 0.1, 0.1]
            )[0]
            list_params[i_p, i_T] = [a, b, c]
    list_params = np.array(list_params)

    return list_params


def get_wilson_loop_critical_temperatures(
    estimates, max_tau=10, L=15, eps=1e-5, start=0, threshold=1e-4
):
    temperatures = estimates['temperature'].unique()
    probas = estimates['p'].unique()

    list_params = get_wilson_loop_ansatz_parameters(
        estimates, max_tau=max_tau, L=L, eps=eps, start=start
    )
    list_a = list_params[:, :, 0]

    critical_temperatures = np.zeros(len(probas))

    for i_p in range(len(probas)):
        critical_temperatures[i_p] = temperatures[list_a[i_p] > threshold][0]

    return critical_temperatures


def plot_wilson_loops(estimates, max_tau=10, L=15, eps=1e-5, save_file=None):
    perimeters, log_wl = get_wilson_loop_data(
        estimates, max_tau=max_tau, L=L, eps=eps
    )

    probas = estimates['p'].unique()
    temperatures = estimates['temperature'].unique()

    plt.rcParams['figure.figsize'] = (10, len(probas)*7)
    plt.rcParams['font.size'] = 15

    for i_p, p_val in enumerate(probas):
        plt.subplot(log_wl.shape[0], 1, i_p+1)
        for i_T, T_val in enumerate(temperatures):
            plt.plot(perimeters, log_wl[i_p, i_T], 'o-', label=f'T={T_val}')

            plt.title(f'p={p_val}')
            plt.ylabel(r"$-\log(WL) / P$")
            plt.xlabel("Perimeter")
            plt.xticks(perimeters)
            plt.legend()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")


def plot_ansatz_wilson_loops(estimates, max_tau=10, L=15, eps=1e-5, start=0):
    a_params = get_wilson_loop_ansatz_parameters(estimates,
                                                 max_tau=max_tau,
                                                 L=L, eps=eps,
                                                 start=start)[:, :, 0]

    probas = estimates['p'].unique()
    temperatures = estimates['temperature'].unique()

    plt.rcParams['figure.figsize'] = (8, 7*len(probas))

    for i_p in range(len(probas)):
        plt.subplot(len(probas), 1, i_p+1)
        plt.plot(temperatures, a_params[i_p, :], "-o")
        plt.xlabel("Temperature")
        plt.ylabel("a")
        plt.title(f"$p={probas[i_p]}$")


def plot_phase_diagram(probabilities, critical_temperatures, save_file=None):
    plt.rcParams['figure.figsize'] = (8, 5)

    plt.plot(probabilities, critical_temperatures, "-o")

    plt.xlabel("Probability")
    plt.ylabel("Temperature")

    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
