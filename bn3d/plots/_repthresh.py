import itertools
import numpy as np
from scipy.special import binom


def plot_repthresh(plt, pdf=None):
    p_list = np.linspace(0, 1, 51)
    n_list = [3, 9, 101]
    p_fail = np.zeros((len(n_list), len(p_list)))
    for (i_n, n), (i_p, p) in itertools.product(
        enumerate(n_list), enumerate(p_list)
    ):
        p_fail[i_n, i_p] = sum(
            binom(n, j)*p**j*(1 - p)**(n - j)
            for j in range(int((n - 1)/2) + 1, n + 1)
        )

    plt.plot(p_list, p_list, '--', label=r'$n=1$')
    markers = ['.', '+', '^']
    for (i_n, n), marker in zip(enumerate(n_list), markers):
        plt.plot(p_list, p_fail[i_n], marker=marker, label=f'$n={n}$')
    plt.axvline(0.5, color='gray', linestyle='-.', label=r'$n\to\infty$')
    plt.legend(title='Reptition Code Size', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.minorticks_on()
    plt.text(
        0.52, 0.1, r'$p_{\rm{th}}=0.5$',
        color='gray', rotation=90, fontsize=16
    )
    plt.xlabel(r'Physical Error rate $p$', fontsize=16)
    plt.ylabel(r'Logical Error rate $p_{{\rm{fail}},n}$', fontsize=16)
    if pdf is not None:
        plt.savefig(pdf, bbox_inches='tight')
    plt.show()
