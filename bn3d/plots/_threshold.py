"""
Routines for extracting plotting threshold.

:Author:
    Eric Huang
"""

import numpy as np
import pandas as pd
from ._hashing_bound import project_triangle, get_hashing_bound
from ..analysis import quadratic


def detailed_plot(plt, results_df, error_model):
    """Plot routine on loop."""
    df = results_df.copy()
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    plot_labels = [
        (0, 'p_est', 'p_se', error_model),
        (1, 'p_x', 'p_x_se', 'Point sector'),
        (2, 'p_z', 'p_z_se', 'Loop sector'),
    ]
    for (i_ax, prob, prob_se, title) in plot_labels:
        ax = axes[i_ax]
        for code_size in df['size'].unique():
            df_filtered = df[
                (df['size'] == code_size) & (df['error_model'] == error_model)
            ]
            ax.errorbar(
                df_filtered['probability'], df_filtered[prob],
                yerr=df_filtered[prob_se],
                label=df_filtered['code'].iloc[0]
            )
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e0)
        ax.set_title(title)
        ax.set_xlabel('Physical Error Rate')
        ax.legend()
    axes[0].set_ylabel('Logical Error Rate')


def update_plot(plt, results_df, error_model):
    """Plot routine on loop."""
    df = results_df.copy()

    for code_size in df['size'].unique():
        df_filtered = df[
            (df['size'] == code_size) & (df['error_model'] == error_model)
        ]
        plt.errorbar(
            df_filtered['probability'], df_filtered['p_est'],
            yerr=df_filtered['p_se'],
            label=df_filtered['code'].iloc[0]
        )
    plt.title(error_model)
    plt.xlabel('Physical Error Rate', fontsize=20)
    plt.ylabel('Logical Error Rate', fontsize=20)
    plt.legend()


def plot_data_collapse(plt, df_trunc, params_opt, params_bs):
    rescaled_p_fit = np.linspace(
        df_trunc['rescaled_p'].min(), df_trunc['rescaled_p'].max(), 101
    )
    f_fit = quadratic(rescaled_p_fit, *params_opt)

    f_fit_bs = np.array([
        quadratic(rescaled_p_fit, *params)
        for params in params_bs
    ])

    for d_val in np.sort(df_trunc['d'].unique()):
        df_trunc_filt = df_trunc[df_trunc['d'] == d_val]
        plt.errorbar(
            df_trunc_filt['rescaled_p'], df_trunc_filt['p_est'],
            yerr=df_trunc_filt['p_se'], fmt='o', capsize=5,
            label=r'$d={}$'.format(d_val)
        )
    plt.plot(
        rescaled_p_fit, f_fit, color='black', linewidth=1, label='Best fit'
    )
    plt.fill_between(
        rescaled_p_fit,
        np.quantile(f_fit_bs, 0.16, axis=0),
        np.quantile(f_fit_bs, 0.84, axis=0),
        color='gray', alpha=0.2, label=r'$1\sigma$ fit'
    )
    plt.xlabel(
        r'Rescaled error probability $(p - p_{\mathrm{th}})d^{1/\nu}$',
        fontsize=16
    )
    plt.ylabel(r'Logical failure rate $p_{\mathrm{fail}}$', fontsize=16)
    plt.title(df_trunc['error_model'].iloc[0])
    plt.legend()


def plot_threshold_fss(
    plt, df_trunc, p_th_fss, p_th_fss_left, p_th_fss_right, p_th_fss_se
):

    for d_val in np.sort(df_trunc['d'].unique()):
        df_trunc_filt = df_trunc[df_trunc['d'] == d_val]
        plt.errorbar(
            df_trunc_filt['probability'], df_trunc_filt['p_est'],
            yerr=df_trunc_filt['p_se'],
            fmt='o-', capsize=5, label=r'$d={}$'.format(d_val)
        )
    plt.axvline(
        p_th_fss, color='red', linestyle='-.',
        label=r'$p_{\mathrm{th}}=(%.2f\pm %.2f)\%%$' % (
            100*p_th_fss, 100*p_th_fss_se,
        )
    )
    plt.axvspan(
        p_th_fss_left, p_th_fss_right, alpha=0.5, color='pink'
    )
    plt.xlabel(
        r'Rescaled error probability $(p - p_{\mathrm{th}})d^{1/\nu}$',
        fontsize=16
    )
    plt.ylabel(r'Logical failure rate $p_{\mathrm{fail}}$', fontsize=16)
    plt.title(df_trunc['error_model'].iloc[0])
    plt.legend()


def draw_tick_symbol(
    plt, Line2D,
    log=False, axis='x',
    tick_height=0.03, tick_width=0.1, tick_location=2.5,
    axis_offset=0,
):
    x_points = np.array([
        -0.25,
        0,
        0,
        0.25,
    ])*tick_width + tick_location
    if log:
        x_points = 10**np.array(x_points)
    y_points = np.array([
        0,
        0.5,
        -0.5,
        0,
    ])*tick_height + axis_offset
    points = (x_points, y_points)
    if axis != 'x':
        points = (y_points, x_points)
    line = Line2D(
        *points,
        lw=1, color='k',
    )
    line.set_clip_on(False)
    plt.gca().add_line(line)


def plot_threshold_vs_bias(plt, Line2D, error_model_df, png=None):
    p_th_key = 'p_th_fss'
    p_th_se_key = 'p_th_fss_se'
    plt.figure()
    eta_keys = ['eta_x', 'eta_z', 'eta_y']
    colors = ['r', 'b', 'g']
    markers = ['x', 'o', '^']
    inf_replacement = 1000
    for eta_key, color, marker in zip(eta_keys, colors, markers):
        df_filt = error_model_df[
            error_model_df[eta_key] >= 0.5
        ].sort_values(by=eta_key)
        p_th_inf = df_filt[df_filt[eta_key] == np.inf][p_th_key].iloc[0]
        plt.errorbar(
            df_filt[eta_key], df_filt[p_th_key],
            yerr=df_filt[p_th_se_key],
            fmt='o-',
            color=color,
            label=r'${}$ bias'.format(eta_key[-1].upper()),
            marker=marker
        )
        plt.plot(
            [
                df_filt[df_filt[eta_key] != np.inf].iloc[-1][eta_key],
                inf_replacement
            ],
            [
                df_filt[df_filt[eta_key] != np.inf].iloc[-1][p_th_key],
                p_th_inf,
            ],
            '--', color=color, marker=marker, linewidth=1
        )
        plt.text(
            inf_replacement,
            p_th_inf + 0.01,
            '{:.3f}'.format(p_th_inf),
            color=color,
            ha='center'
        )

    # Show label for depolarizing data point.
    p_th_dep = error_model_df[error_model_df[eta_key] == 0.5].iloc[0][p_th_key]
    plt.text(0.5, p_th_dep + 0.01, f'{p_th_dep:.3f}', ha='center')

    eta_interp = np.logspace(np.log10(0.5), np.log10(100), 101)
    interp_points = [
        (
            1/(2*(1 + eta)),
            1/(2*(1 + eta)),
            eta/(1 + eta)
        )
        for eta in eta_interp
    ]
    hb_interp = [
        get_hashing_bound(point)
        for point in interp_points
    ]
    plt.plot(eta_interp, hb_interp, '-.', color='black', label='hashing')
    plt.plot(
        df_filt[eta_key],
        [get_hashing_bound(point) for point in df_filt['noise_direction']],
        'k.'
    )
    plt.plot(
        inf_replacement,
        get_hashing_bound((0, 0, 1)),
        '.'
    )
    plt.plot(
        [
            eta_interp[-1],
            inf_replacement,
        ],
        [
            hb_interp[-1],
            get_hashing_bound((0, 0, 1))
        ],
        '--', color='black', linewidth=1,
    )
    plt.legend()
    plt.xscale('log')
    draw_tick_symbol(plt, Line2D, log=True)
    plt.xticks(
        ticks=[0.5, 1e0, 1e1, 1e2, inf_replacement],
        labels=['0.5', '1', '10', ' '*13 + '100' + ' '*10 + '...', r'$\infty$']
    )
    plt.xlabel(r'Bias Ratio $\eta$', fontsize=16)
    plt.ylabel(r'Threshold $p_{\mathrm{th}}$', fontsize=16)
    if png is not None:
        plt.savefig(png)


def plot_thresholds_on_triangle(plt, error_model_df):
    eta_keys = ['eta_x', 'eta_z', 'eta_y']
    colors = ['r', 'b', 'g']
    markers = ['x', 'o', '^']
    label_offsets = [(0, 0.1), (0.1, 0), (0, 0.1)]

    plt.plot(*np.array([
        project_triangle(point)
        for point in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]
    ]).T, '-', color='gray', linewidth=1)

    p_th_dep = error_model_df[
        error_model_df['eta_z'] == 0.5
    ].iloc[0]['p_th_sd']
    plt.text(0.1, 0, f'{p_th_dep:.2f}', ha='center')

    for eta_key, color, marker, offset in zip(
        eta_keys, colors, markers, label_offsets
    ):
        df_filt = error_model_df[
            error_model_df[eta_key] >= 0.5
        ].sort_values(by=eta_key)
        plt.plot(
            df_filt['h'], df_filt['v'],
            marker=marker, linestyle='-', color=color,
            label=r'${}$ bias'.format(eta_key[-1].upper()),
        )
        plt.text(
            *np.array(project_triangle(df_filt.iloc[-1]['noise_direction']))
            + offset,
            '{:.2f}'.format(df_filt.iloc[-1]['p_th_sd']),
            color=color, ha='center'
        )
    plt.text(
        *np.array(project_triangle((1, 0, 0))) + [-0.1, -0.1],
        r'$X$',
        ha='center', fontsize=16, color='r'
    )
    plt.text(
        *np.array(project_triangle((0, 1, 0))) + [0.1, -0.1],
        r'$Y$',
        ha='center', fontsize=16, color='g'
    )
    plt.text(
        *np.array(project_triangle((0, 0, 1))) + [0, 0.1],
        r'$Z$',
        ha='center', fontsize=16, color='b'
    )
    plt.axis('off')
    plt.legend(title='Thresholds')
    plt.gca().set_aspect(1)
