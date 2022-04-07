"""
Routines for extracting plotting threshold.

:Author:
    Eric Huang
"""
import re
import os
import numpy as np
import pandas as pd
from scipy.special import binom
import itertools
from ._hashing_bound import project_triangle, get_hashing_bound
from ..analysis import quadratic


def detailed_plot(
    plt, results_df, error_model, x_limits=None, save_folder=None,
    yscale=None, eta_key='eta_x', min_y_axis=1e-3,
    thresholds_df=None,
):
    """Plot routine on loop.

    Parameters
    ----------
    plt : matplotlib.pyplot
        The matplotlib pyplot reference.
    results_df : pd.Dataframe
        Results table.
    error_model : str
        Name of the error model to filter to.
    x_limits : Optional[Union[List[Tuple[float, float]], str]]
        Will set limits from 0 to 0.5 if None given.
        Will not impose limits if 'auto' given.
    save_folder : str
        If given will save save figure as png to directory.
    yscale : Optional[str]
        Set to 'log' to make yscale logarithmic.
    thresholds_df : Optional[pd.DataFrame]
        Plot the estimated threshold if given.
    """
    df = results_df.copy()
    df.sort_values('probability', inplace=True)
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    plot_labels = [
        (0, 'p_est', 'p_se', 'All sectors'),
        (1, 'p_x', 'p_x_se', 'Point sector'),
        (2, 'p_z', 'p_z_se', 'Loop sector'),
    ]
    if x_limits is None:
        x_limits = [(0, 0.5), (0, 0.5), (0, 0.5)]

    eta = df[(df['error_model'] == error_model)].iloc[0][eta_key]
    fig.suptitle(f"$\\eta={eta:.1f}$")

    for (i_ax, prob, prob_se, title) in plot_labels:
        ax = axes[i_ax]
        legend_title = None
        for code_size in np.sort(df['size'].unique()):
            df_filtered = df[
                (df['size'] == code_size) & (df['error_model'] == error_model)
            ]
            ax.errorbar(
                df_filtered['probability'], df_filtered[prob],
                yerr=df_filtered[prob_se],
                label=r'$L={}$'.format(df_filtered['size'].iloc[0][0]),
                capsize=1,
                linestyle='-',
                marker='.',
            )
        if i_ax == 0 and thresholds_df is not None:
            thresholds = thresholds_df[
                thresholds_df['error_model'] == error_model
            ]
            if thresholds.shape[0] > 0:
                p_th_fss_left = thresholds['p_th_fss_left'].iloc[0]
                p_th_fss_right = thresholds['p_th_fss_right'].iloc[0]
                p_th_fss = thresholds['p_th_fss'].iloc[0]
                p_th_fss_se = thresholds['p_th_fss_se'].iloc[0]
                if not pd.isna(p_th_fss_left) and not pd.isna(p_th_fss_right):
                    ax.axvspan(
                        p_th_fss_left, p_th_fss_right,
                        alpha=0.5, color='pink'
                    )
                if not pd.isna(p_th_fss):
                    ax.axvline(
                        p_th_fss,
                        color='red',
                        linestyle='--',
                    )
                if p_th_fss_se is not None and p_th_fss is not None:
                    legend_title = r'$p_{\mathrm{th}}=(%.2f\pm %.2f)\%%$' % (
                        100*p_th_fss, 100*p_th_fss_se,
                    )
        if yscale is not None:
            ax.set_yscale(yscale)
        if x_limits != 'auto':
            ax.set_xlim(x_limits[i_ax])
            ax.set_ylim(min_y_axis, 1)

        ax.set_title(title)
        ax.locator_params(axis='x', nbins=6)

        ax.set_xlabel('Physical Error Rate')
        if legend_title is not None:
            ax.legend(loc='best', title=legend_title)
        else:
            ax.legend(loc='best')
    axes[0].set_ylabel('Logical Error Rate')

    fig.suptitle(f"$\eta={eta:.1f}$")

    if save_folder:
        filename = os.path.join(save_folder, results_df['label'][0])
        plt.savefig(f'{filename}.png')


def xyz_sector_plot(
    plt, results_df, error_model, x_limits=None, save_folder=None,
    yscale=None, eta_key='eta_x', min_y_axis=1e-3,
    thresholds_df=None,
):
    """Plot the different sectors (pure X, pure Y, pure Z) crossover plots

    Parameters
    ----------
    plt : matplotlib.pyplot
        The matplotlib pyplot reference.
    results_df : pd.Dataframe
        Results table.
    error_model : str
        Name of the error model to filter to.
    x_limits : Optional[Union[List[Tuple[float, float]], str]]
        Will set limits from 0 to 0.5 if None given.
        Will not impose limits if 'auto' given.
    save_folder : str
        If given will save save figure as png to directory.
    yscale : Optional[str]
        Set to 'log' to make yscale logarithmic.
    min_y_axis: Optional[float]
        Minimum value in the yscale (relevant in logarithmic scale)
    thresholds_df : Optional[pd.DataFrame]
        Plot the estimated threshold if given.
    """
    df = results_df.copy()
    df.sort_values('probability', inplace=True)
    fig, axes = plt.subplots(ncols=4, figsize=(16, 4))
    plot_labels = [
        (0, 'p_est', 'p_se', 'Full threshold'),
        (1, 'p_pure_x', 'p_pure_x_se', '$X_L$ sector'),
        (2, 'p_pure_y', 'p_pure_y_se', '$Y_L$ sector'),
        (3, 'p_pure_z', 'p_pure_z_se', '$Z_L$ sector'),
    ]
    if x_limits is None:
        x_limits = [(0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5)]

    eta = df[(df['error_model'] == error_model)].iloc[0][eta_key]

    for (i_ax, prob, prob_se, title) in plot_labels:
        ax = axes[i_ax]
        legend_title = None
        for code_size in np.sort(df['size'].unique()):
            df_filtered = df[
                (df['size'] == code_size) & (df['error_model'] == error_model)
            ]
            ax.errorbar(
                df_filtered['probability'], df_filtered[prob],
                yerr=df_filtered[prob_se],
                label=r'$d={}$'.format(df_filtered['size'].iloc[0][0]),
                capsize=1,
                linestyle='-',
                marker='.',
            )

        if i_ax == 0 and thresholds_df is not None:
            thresholds = thresholds_df[
                thresholds_df['error_model'] == error_model
            ]
            if thresholds.shape[0] > 0:
                p_th_fss_left = thresholds['p_th_fss_left'].iloc[0]
                p_th_fss_right = thresholds['p_th_fss_right'].iloc[0]
                p_th_fss = thresholds['p_th_fss'].iloc[0]
                p_th_fss_se = thresholds['p_th_fss_se'].iloc[0]
                if not pd.isna(p_th_fss_left) and not pd.isna(p_th_fss_right):
                    ax.axvspan(
                        p_th_fss_left, p_th_fss_right,
                        alpha=0.5, color='pink'
                    )
                if not pd.isna(p_th_fss):
                    ax.axvline(
                        p_th_fss,
                        color='red',
                        linestyle='--',
                    )
                if p_th_fss_se is not None and p_th_fss is not None:
                    legend_title = r'$p_{\mathrm{th}}=(%.2f\pm %.2f)\%%$' % (
                        100*p_th_fss, 100*p_th_fss_se,
                    )
        if yscale is not None:
            ax.set_yscale(yscale)
        if x_limits != 'auto':
            ax.set_xlim(x_limits[i_ax])
            ax.set_ylim(min_y_axis, 1e0)

        ax.set_title(title)
        ax.locator_params(axis='x', nbins=6)

        ax.set_xlabel('Physical Error Rate')
        if legend_title is not None:
            ax.legend(loc='best', title=legend_title)
        else:
            ax.legend(loc='best')
    axes[0].set_ylabel('Logical Error Rate')

    fig.suptitle(f"$\\eta={eta:.1f}$")

    if save_folder:
        filename = os.path.join(save_folder, results_df['label'][0])
        plt.savefig(f'{filename}.png')


def update_plot(plt, results_df, error_model, xlim=None, ylim=None, save_folder=None,
                yscale=None, eta_key='eta_x', min_y_axis=1e-3):
    """Plot routine on loop."""
    df = results_df.copy()
    df.sort_values('probability', inplace=True)

    if xlim is None:
        xlim = (0, 0.5)
    if ylim is None:
        ylim = (0, 1.)

    for code_size in np.sort(df['size'].unique()):
        df_filtered = df[
            (df['size'] == code_size) & (df['error_model'] == error_model)
        ]
        plt.errorbar(
            df_filtered['probability'], df_filtered['p_est'],
            yerr=df_filtered['p_se'],
            label=r'$d={}$'.format(df_filtered['size'].iloc[0][0]),
            capsize=1,
            linestyle='-',
            marker='.'
        )

    if xlim != 'auto':
        plt.xlim(xlim)
        plt.ylim(ylim)

    if yscale is not None:
        plt.yscale(yscale)

    # plt.title(error_model)
    plt.xlabel('Physical Error Rate', fontsize=16)
    plt.ylabel('Logical Error Rate', fontsize=16)
    plt.legend(prop={'size': 12})


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



    error_model = df_trunc['error_model'].iloc[0]
    title = get_error_model_format(error_model)
    plt.title(title)
    plt.legend()


def get_error_model_format(error_model: str, eta=None) -> str:
    if 'deformed' in error_model:
        fmt = 'Deformed'
    else:
        fmt = 'Undeformed'

    if eta is None:
        match = re.search(r'Pauli X(.+)Y(.+)Z(.+)', error_model)
        if match:
            r_x = np.round(float(match.group(1)), 4)
            r_y = np.round(float(match.group(2)), 4)
            r_z = np.round(float(match.group(3)), 4)
            fmt += ' $(r_X, r_Y, r_Z)=({},{},{})$'.format(r_x, r_y, r_z)
        else:
            fmt = error_model
    else:
        fmt += r' $\eta={}$'.format(eta)
    return fmt


def plot_threshold_nearest(plt, p_th_nearest):
    plt.axvline(
        p_th_nearest, color='green', linestyle='-.',
        label=r'$p_{\mathrm{th}}=(%.2f)\%%$' % (
            100*p_th_nearest
        )
    )


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
        'Error probability $p$',
        fontsize=16
    )
    plt.ylabel(r'Logical failure rate $p_{\mathrm{fail}}$', fontsize=16)
    plt.title(get_error_model_format(df_trunc['error_model'].iloc[0]))
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


def format_eta(eta, decimals=4):
    if eta == np.inf:
        return r'\infty'
    elif np.isclose(np.round(eta, decimals) % 1, 0):
        return str(int(np.round(eta, decimals)))
    else:
        return str(np.round(eta, decimals))


def plot_threshold_vs_bias(
    plt, Line2D, error_model_df, main_linestyle='-',
    eta_keys=['eta_x', 'eta_z', 'eta_y'],
    markers=['x', 'o', '^'],
    colors=['r', 'b', 'g'],
    alphas=[1, 1, 1],
    labels=None,
    depolarizing_label=False,
    hashing=True,
    png=None,
):
    p_th_key = 'p_th_fss'
    p_th_left_key = 'p_th_fss_left'
    p_th_right_key = 'p_th_fss_right'
    if labels is None:
        labels = [
            r'${}$ bias'.format(eta_key[-1].upper())
            for eta_key in eta_keys
        ]

    inf_replacement = 1000

    for eta_key, color, alpha, marker, label in zip(
        eta_keys, colors, alphas, markers, labels
    ):
        df_filt = error_model_df[
            error_model_df[eta_key] >= 0.4
        ].sort_values(by=eta_key)

        p_th_inf = df_filt[df_filt[eta_key] == np.inf][p_th_key].iloc[0]

        df_filt.replace(np.inf, inf_replacement, inplace=True)

        errors_left = df_filt[p_th_key] - df_filt[p_th_left_key]
        errors_right = df_filt[p_th_right_key] - df_filt[p_th_key]
        # errors = np.array([errors_left, errors_right])
        errors = np.zeros(df_filt[p_th_key].shape)
        plt.errorbar(
            df_filt[eta_key], df_filt[p_th_key], errors,
            linestyle=main_linestyle,
            color=color,
            alpha=alpha,
            label=label,
            marker=marker,
            markersize=8,
            linewidth=2
        )

        plt.text(
            inf_replacement - 100,
            p_th_inf + 0.03,
            '{:.2f}'.format(p_th_inf),
            color=color,
            ha='center',
            fontsize=15
        )

        # Show label for depolarizing data point.
        if depolarizing_label:
            p_th_dep = error_model_df[np.isclose(error_model_df[eta_key], 0.5)].iloc[0][p_th_key]
            plt.text(0.5 + 0.05, p_th_dep + 0.03, f'{p_th_dep:.2f}', ha='center', color=color, fontsize=15)

    # Plot the hashing bound curve.
    if hashing:
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
        eta_interp = np.append(eta_interp, [inf_replacement])
        hb_interp = np.append(hb_interp, [get_hashing_bound((0, 0, 1))])

        plt.plot(eta_interp, hb_interp, '-.', color='black', label='Hashing bound', alpha=0.5, linewidth=2)

    plt.xscale('log')

    plt.xlabel(r'$\eta$', fontsize=27)
    plt.ylabel(r'$p_{\mathrm{th}}$', fontsize=27)

    plt.ylim(0, 0.5)
    plt.legend(fontsize=16, loc='upper left')

    draw_tick_symbol(plt, Line2D, log=True)
    plt.xticks(
        ticks=[0.5, 1e0, 1e1, 1e2, inf_replacement],
        labels=['0.5', '1', '10', ' '*13 + '100' + ' '*10 + '...', r'$\infty$'],
        fontsize=17
    )
    plt.yticks(fontsize=17)


def plot_thresholds_on_triangle(
    plt, error_model_df, title='Thresholds',
    colors=['r', 'b', 'g']
):
    label_threshold = 'p_th_fss'
    eta_keys = ['eta_x', 'eta_z', 'eta_y']
    markers = ['x', 'o', '^']
    label_offsets = [(0, 0.1), (0.2, 0), (0, 0.1)]

    plt.plot(*np.array([
        project_triangle(point)
        for point in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)]
    ]).T, '-', color='gray', linewidth=1)

    p_th_dep = error_model_df[
        error_model_df['eta_z'] == 0.5
    ].iloc[0][label_threshold]
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
            '{:.2f}'.format(df_filt.iloc[-1][label_threshold]),
            color=color, ha='center',
            fontsize=16
        )
    plt.text(
        *np.array(project_triangle((1, 0, 0))) + [-0.1, -0.1],
        r'$X$',
        ha='center', fontsize=20, color='r'
    )
    plt.text(
        *np.array(project_triangle((0, 1, 0))) + [0.1, -0.1],
        r'$Y$',
        ha='center', fontsize=20, color='g'
    )
    plt.text(
        *np.array(project_triangle((0, 0, 1))) + [0, 0.1],
        r'$Z$',
        ha='center', fontsize=20, color='b'
    )
    plt.axis('off')
    plt.legend(title=title, loc='upper left', title_fontsize=16, fontsize=12)
    plt.gca().set_aspect(1)


def plot_combined_threshold_vs_bias(plt, Line2D, thresholds_df,
                                    hashing=False,
                                    eta_keys=['eta_z', 'eta_z'],
                                    labels=['3D XZZZZX', '3D CSS'],
                                    colors=['r', 'b'],
                                    alphas=[1, 1],
                                    markers=['o', 'x'],
                                    linestyles=['-', '--'],
                                    depolarizing_labels=[True, False],
                                    figsize=(5,4),
                                    pdf=None):
    n_plots = len(thresholds_df)

    for i_plot in range(n_plots):
        hashing = hashing and (i_plot == 0)
        plot_threshold_vs_bias(
            plt, Line2D, thresholds_df[i_plot],
            markers=[markers[i_plot]],
            eta_keys=[eta_keys[i_plot]],
            colors=[colors[i_plot]],
            alphas=[alphas[i_plot]],
            labels=[labels[i_plot]],
            depolarizing_label=depolarizing_labels[i_plot],
            main_linestyle=linestyles[i_plot],
            hashing=hashing
        )

    plt.rcParams['figure.figsize'] = figsize

    if pdf:
        plt.savefig(pdf, bbox_inches='tight')


def plot_crossing_collapse(
    plt, bias_direction, deformed, results_df,
    thresholds_df, trunc_results_df, params_bs_list,
    pdf=None
):
    params_bs_dict = dict(zip(
        thresholds_df['error_model'].values,
        params_bs_list
    ))
    eta_key = f'eta_{bias_direction}'
    plot_thresholds_df = thresholds_df[
        ~(thresholds_df['error_model'].str.contains('Deformed') ^ deformed)
        & (thresholds_df[eta_key] >= 0.5)
    ].sort_values(by=eta_key)
    fig, axes = plt.subplots(
        plot_thresholds_df.shape[0], 2, figsize=(10, 15),
        gridspec_kw={'width_ratios': [1.2, 1]}
    )
    for i, (_, row) in enumerate(plot_thresholds_df.iterrows()):
        plt.sca(axes[i, 0])
        df_trunc = trunc_results_df[
            (trunc_results_df['error_model'] == row['error_model'])
        ]
        df_no_trunc = results_df[
            results_df['error_model'] == row['error_model']
        ].copy()
        df_no_trunc['d'] = results_df['d']
        plot_threshold_fss(
            plt, df_no_trunc, row['p_th_fss'], row['p_th_fss_left'],
            row['p_th_fss_right'], row['p_th_fss_se']
        )
        plot_threshold_nearest(
            plt, row['p_th_nearest']
        )
        plt.title(None)
        plt.ylabel(r'$p_{\mathrm{fail}}$')
        plt.gca().tick_params(direction='in')
        if i < plot_thresholds_df.shape[0] - 1:
            plt.gca().tick_params(labelbottom=False, direction='in')
            plt.xlabel(None)
        else:
            plt.xlabel(r'Error Rate $p$')
        plt.gca().minorticks_on()
        plt.gca().tick_params(direction='in', which='minor')
        plt.ylim(0, 0.9)

        proba_min = min(df_no_trunc['probability'].unique())
        proba_max = max(df_no_trunc['probability'].unique())

        # proba_min = 0
        # proba_max = 0.5

        plt.xlim(proba_min, proba_max)
        plt.gca().get_legend().remove()
        plt.gca().legend(
            plt.gca().get_legend_handles_labels()[1][:1],
            title=r'$\eta_%s=%s$' % (
                bias_direction.upper(),
                format_eta(row[eta_key])
            ),
            title_fontsize=12,
            loc='lower right'
        )

        plt.sca(axes[i, 1])
        plot_data_collapse(
            plt, df_trunc, row['fss_params'],
            params_bs_dict[row['error_model']]
        )
        plt.title(None)
        plt.ylabel(None)
        rescaled_p_vals = trunc_results_df[
            trunc_results_df['error_model'] == row['error_model']
        ]['rescaled_p']
        plt.xlim(
            np.min(rescaled_p_vals) - 0.05,
            np.max(rescaled_p_vals) + 0.05
        )
        plt.gca().tick_params(labelleft=False, axis='y', direction='in')
        if i < plot_thresholds_df.shape[0] - 1:
            plt.gca().tick_params(labelbottom=False, direction='in')
            plt.xlabel(None)
        else:
            plt.xlabel(r'Rescaled Error rate $(p-p_{\mathrm{th}})d^{1/\nu}$')
        plt.gca().get_legend().remove()
        plt.legend(
            ncol=3
        )
        plt.ylim(0, 0.9)
    fig.subplots_adjust(wspace=0.01, hspace=0)
    if pdf:
        plt.savefig(pdf, bbox_inches='tight')


def plot_deformedxps(plt, results_df, pdf=None):
    detailed_plot(
        plt, results_df, 'Deformed Pauli X1.0Y0.0Z0.0',
    )
    if pdf:
        plt.savefig(pdf, bbox_inches='tight')


def plot_deformedzps(plt, results_df, pdf=None):
    detailed_plot(
        plt, results_df, 'Deformed Pauli X0.0Y0.0Z1.0',
    )
    if pdf:
        plt.savefig(pdf, bbox_inches='tight')


def plot_combined_triangles(plt, thresholds_df, pdf=None):
    thres_df_filt = thresholds_df[
        thresholds_df['error_model'].str.contains('Deformed')
    ]
    fig, axes = plt.subplots(nrows=2, figsize=(4, 8))
    plt.sca(axes[0])
    plot_thresholds_on_triangle(
        plt, thres_df_filt, title='Deformed', colors=['r', 'b', 'g']
    )
    thres_df_filt = thresholds_df[
        ~thresholds_df['error_model'].str.contains('Deformed')
    ]
    plt.sca(axes[1])
    plot_thresholds_on_triangle(
        plt, thres_df_filt, title='Undeform.',
        colors=['#ff9999', '#9999ff', '#55aa55']
    )
    if pdf:
        plt.savefig(pdf, bbox_inches='tight')


def plot_crossing_example(
    plt, results_df, thresholds_df, params_bs_list, pdf=None
):
    error_model = 'Deformed Pauli X0.0Y0.0Z1.0'
    row = thresholds_df[thresholds_df['error_model'] == error_model].iloc[0]
    df_no_trunc = results_df[
        results_df['error_model'] == row['error_model']
    ].copy()
    df_no_trunc['d'] = results_df['d']
    plot_threshold_fss(
        plt, df_no_trunc, row['p_th_fss'], row['p_th_fss_left'],
        row['p_th_fss_right'], row['p_th_fss_se']
    )
    if pdf:
        plt.savefig(pdf, bbox_inches='tight')


def plot_collapse_example(
    plt, thresholds_df, trunc_results_df, params_bs_list,
    verbose=True, pdf=None
):
    error_model = 'Deformed Pauli X0.0Y0.0Z1.0'
    row = thresholds_df[thresholds_df['error_model'] == error_model].iloc[0]
    df_trunc = trunc_results_df[trunc_results_df['error_model'] == error_model]
    i = thresholds_df['error_model'].tolist().index(error_model)
    plot_data_collapse(plt, df_trunc, row['fss_params'], params_bs_list[i])
    if verbose:
        print(pd.DataFrame(
            {
                'value': row['fss_params'],
                'se': np.std(params_bs_list[i], axis=0)
            },
            index=['p_th', 'nu', 'A', 'B', 'C']
        ))
    if pdf:
        plt.savefig(pdf, bbox_inches='tight')


def plot_repetition_code_threshold(plt, pdf=None):
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

