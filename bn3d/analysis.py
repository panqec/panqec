"""
Routines for analysing output data.

:Author:
    Eric Huang
"""

import os
import warnings
from typing import List, Optional, Tuple
import itertools
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from .config import SLURM_DIR
from .app import read_input_json
from .plots._hashing_bound import project_triangle
from .utils import fmt_uncertainty


def get_results_df_from_batch(batch_sim, batch_label):
    batch_results = batch_sim.get_results()
    # print(
    #     'wall_time =',
    #     str(datetime.timedelta(seconds=batch_sim.wall_time))
    # )
    # print('n_trials = ', min(sim.n_results for sim in batch_sim))
    for sim, batch_result in zip(batch_sim, batch_results):
        n_logicals = batch_result['n_k_d'][1]

        # Small fix for the current situation. TO REMOVE in later versions
        if n_logicals == -1:
            n_logicals = 1

        batch_result['label'] = batch_label
        batch_result['noise_direction'] = sim.error_model.direction
        if len(sim.results['effective_error']) > 0:
            batch_result['p_x'] = np.array(
                sim.results['effective_error']
            )[:, :n_logicals].any(axis=1).mean()
            batch_result['p_x_se'] = np.sqrt(
                batch_result['p_x']*(1 - batch_result['p_x'])
                / (sim.n_results + 1)
            )
            batch_result['p_z'] = np.array(
                sim.results['effective_error']
            )[:, n_logicals:].any(axis=1).mean()
            batch_result['p_z_se'] = np.sqrt(
                batch_result['p_z']*(1 - batch_result['p_z'])
                / (sim.n_results + 1)
            )
        else:
            batch_result['p_x'] = np.nan
            batch_result['p_x_se'] = np.nan
            batch_result['p_z'] = np.nan
            batch_result['p_z_se'] = np.nan

    results = batch_results

    results_df = pd.DataFrame(results)
    return results_df


def get_results_df(
    job_list: List[str],
    output_dir: str,
    input_dir: str = None
) -> pd.DataFrame:
    """Get raw results in DataFrame."""
    if input_dir is None:
        input_dir = os.path.join(SLURM_DIR, 'inputs')

    input_files = [
        os.path.join(input_dir, f'{name}.json')
        for name in job_list
    ]
    output_dirs = [
        os.path.join(output_dir, name)
        for name in job_list
    ]
    batches = {}
    for i in range(len(input_files)):
        batch_sim = read_input_json(input_files[i])
        for sim in batch_sim:
            sim.load_results(output_dirs[i])
        batches[batch_sim.label] = batch_sim

    results = []
    for batch_label, batch_sim in batches.items():
        batch_results = batch_sim.get_results()
        # print(
        #     'wall_time =',
        #     str(datetime.timedelta(seconds=batch_sim.wall_time))
        # )
        # print('n_trials = ', min(sim.n_results for sim in batch_sim))
        for sim, batch_result in zip(batch_sim, batch_results):
            n_logicals = batch_result['n_k_d'][1]

            # Small fix for the current situation. TO REMOVE in later versions
            if n_logicals == -1:
                n_logicals = 1

            batch_result['label'] = batch_label
            batch_result['noise_direction'] = sim.error_model.direction
            if len(sim.results['effective_error']) > 0:
                batch_result['p_x'] = np.array(
                    sim.results['effective_error']
                )[:, :n_logicals].any(axis=1).mean()
                batch_result['p_x_se'] = np.sqrt(
                    batch_result['p_x']*(1 - batch_result['p_x'])
                    / (sim.n_results + 1)
                )
                batch_result['p_z'] = np.array(
                    sim.results['effective_error']
                )[:, n_logicals:].any(axis=1).mean()
                batch_result['p_z_se'] = np.sqrt(
                    batch_result['p_z']*(1 - batch_result['p_z'])
                    / (sim.n_results + 1)
                )
            else:
                batch_result['p_x'] = np.nan
                batch_result['p_x_se'] = np.nan
                batch_result['p_z'] = np.nan
                batch_result['p_z_se'] = np.nan
        results += batch_results

    results_df = pd.DataFrame(results)
    return results_df


def get_p_th_sd_interp(
    df_filt: pd.DataFrame,
    p_nearest: Optional[float] = None
) -> Tuple[float, float, float]:
    """Estimate threshold by where SD of p_est is local min."""

    # Points to interpolate at.
    interp_res = 0.001
    p_min = df_filt['probability'].min()
    p_max = df_filt['probability'].max()
    if p_nearest is not None:
        p_max = min(p_max, p_nearest*2)
    p_interp = np.arange(p_min, p_max + interp_res, interp_res)

    # Initialize to extents by default.
    p_left = p_min
    p_right = p_max

    # Interpolate lines.
    curves = {}
    for code in df_filt['code'].unique():
        df_filt_code = df_filt[df_filt['code'] == code].sort_values(
            by='probability'
        )
        interpolator = interp1d(
            df_filt_code['probability'], df_filt_code['p_est'],
            fill_value="extrapolate"
        )
        curves[code] = interpolator(p_interp)
    interp_df = pd.DataFrame(curves)
    interp_df.index = p_interp

    # SD of p_est interpolated.
    interp_std = interp_df.std(axis=1)

    # Local minima and local maxima indices.
    i_minima = argrelextrema(interp_std.values, np.less)[0]
    i_maxima = argrelextrema(interp_std.values, np.greater)[0]

    if len(i_minima) == 0:
        i_minima = argrelextrema(interp_std.values, np.less_equal)[0]

    if len(i_maxima) == 0:
        i_maxima = argrelextrema(interp_std.values, np.greater_equal)[0]

    # Also include the end points in the maxima.
    i_maxima = np.array([0] + i_maxima.tolist() + [len(p_interp) - 1])

    std_peak_heights = []
    for i_minimum in i_minima:
        i_left_maxima = i_maxima[i_maxima < i_minimum]
        i_right_maxima = i_maxima[i_maxima > i_minimum]

        left_height = 0
        if len(i_left_maxima) > 0:
            i_left_maximum = i_left_maxima.max()
            left_height = (
                interp_std.iloc[i_left_maximum] - interp_std.iloc[i_minimum]
            )

        right_height = 0
        if len(i_right_maxima) > 0:
            i_right_maximum = i_right_maxima.min()
            right_height = (
                interp_std.iloc[i_right_maximum] - interp_std.iloc[i_minimum]
            )

        std_peak_heights.append(left_height + right_height)

    # Find the local minimum surrounded by highest peaks.
    try:
        i_crossover = i_minima[np.argmax(std_peak_heights)]
    except ValueError as err:
        print(std_peak_heights, i_minima)
        print(interp_std.values)
        raise ValueError(err)

    # Left and right peak SD locations.
    p_left = p_interp[i_maxima[i_maxima < i_crossover].max()]
    p_right = p_interp[i_maxima[i_crossover < i_maxima].min()]

    # Crossover is the error rate value for that point.
    p_crossover = p_interp[i_crossover]
    return p_crossover, p_left, p_right


def get_code_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame of codes available."""
    code_df = results_df[['code', 'n_k_d']].copy()
    code_df = code_df.drop_duplicates().reset_index(drop=True)
    code_df = pd.concat([
        code_df,
        pd.DataFrame(
            code_df['n_k_d'].tolist(),
            index=code_df.index,
            columns=['n', 'k', 'd']
        )
    ], axis=1)
    code_df = code_df.sort_values(by='n').reset_index(drop=True)
    return code_df


def longest_sequence(arr, char):
    curr_seq_start = 0
    curr_seq_stop = 0
    best_seq = (curr_seq_start, curr_seq_stop)
    for i in range(len(arr)):
        if arr[i] == char:
            curr_seq_stop += 1
            if curr_seq_stop - curr_seq_start > best_seq[1] - best_seq[0]:
                best_seq = (curr_seq_start, curr_seq_stop)
        else:
            curr_seq_start = i+1
            curr_seq_stop = i+1

    return best_seq


def get_p_th_nearest(df_filt: pd.DataFrame) -> float:
    code_df = get_code_df(df_filt)
    # Estimate the threshold by where the order of the lines change.
    p_est_df = pd.DataFrame({
        code: dict(df_filt[df_filt['code'] == code][[
            'probability', 'p_est'
        ]].values)
        for code in code_df['code']
    })
    p_est_df = p_est_df.sort_index()

    # Where the ordering changes the most.
    # orders = np.argsort(p_est_df.values, axis=1)
    # orders = np.apply_along_axis(
    #     lambda x: 'A' if np.all(np.diff(x) < 0)
    # else ('B' if np.all(np.diff(x) > 0) else '0'), 1, orders
    # )

    # cond = np.all(
    #     np.isclose(p_est_df.values, np.zeros(p_est_df.shape[1])), axis=1
    # )
    # orders[cond] = 'A'

    # longest_A_seq = longest_sequence(orders, 'A')
    # longest_B_seq = longest_sequence(orders, 'B')
    # if longest_A_seq[1] > longest_B_seq[0]:
    #     print(p_est_df)
    #     print(p_est_df.values)
    #     print(np.argsort(p_est_df.values, axis=1))
    #     print(orders)
    #     raise RuntimeError(
    #         "Problem with finding p_th_nearest: "
    #         f"{longest_A_seq} > {longest_B_seq}"
    #     )

    # p_th_nearest = p_est_df.index[(longest_B_seq[0] + longest_A_seq[1]) // 2]

    i_order_change = np.diff(
        np.diff(
            np.argsort(p_est_df.values, axis=1)
        ).sum(axis=1)
    ).argmax()
    p_th_nearest = p_est_df.index[i_order_change]

    return p_th_nearest


# def fit_function(x_data, *params):
#     p, d = x_data
#     p_th, nu, A, B, C = params
#     x = (p - p_th)*d**nu
#     return A + B*x + C*x**2

def fit_function(x_data, *params):
    p, d = x_data
    p_th, nu, A, B, C = params
    x = (p - p_th)*d**nu
    return A + B*x + C*x**2


def grad_fit_function(x_data, *params):
    p, d = x_data
    p_th, nu, A, B, C = params
    x = (p - p_th)*d**nu

    grad_p_th = - B * d**nu - 2*C*(p - p_th) * d**(2*nu)
    grad_nu = x * np.log(d) * (B + 2*C*x)
    grad_A = 1 * np.ones(grad_nu.shape)
    grad_B = x * np.ones(grad_nu.shape)
    grad_C = x**2 * np.ones(grad_nu.shape)

    jac = np.vstack([grad_p_th, grad_nu, grad_A, grad_B, grad_C]).T
    return jac


def quadratic(x, *params):
    _, _, A, B, C = params
    return A + B*x + C*x**2


def rescale_prob(x_data, *params):
    p, d = x_data
    p_th, nu, A, B, C = params
    x = (p - p_th)*d**nu
    return x


def get_fit_params(
    p_list, d_list, f_list, params_0=None, ftol=1e-5, maxfev=2000
) -> np.ndarray:
    """Get fitting params."""
    # Curve fitting inputs.
    x_data = np.array([
        p_list,
        d_list
    ])

    # Target outputs.
    y_data = f_list

    # Curve fit.
    lower_bound = 0
    if params_0[0] < lower_bound:
        params_0[0] = np.random.uniform(lower_bound, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params_opt, _ = curve_fit(
            fit_function, x_data, y_data, jac=grad_fit_function, method='trf',
            p0=params_0, ftol=ftol, maxfev=maxfev,
            bounds=(
                [lower_bound] + [-np.inf]*4,  # type: ignore
                [0.5] + [np.inf]*4
            )
        )

    return params_opt


def fit_fss_params(
    df_filt: pd.DataFrame,
    p_left_val: float,
    p_right_val: float,
    p_nearest: float,
    n_bs: int = 100,
    ftol_est: float = 1e-5,
    ftol_std: float = 1e-5,
    maxfev: int = 2000
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Get optimized parameters and data table."""
    # Truncate error probability between values.
    df_trunc = df_filt[
        (p_left_val <= df_filt['probability'])
        & (df_filt['probability'] <= p_right_val)
    ].copy()
    df_trunc = df_trunc.dropna(subset=['p_est'])
    df_trunc['d'] = df_trunc['n_k_d'].apply(lambda x: x[2])

    d_list = df_trunc['d'].values
    p_list = df_trunc['probability'].values
    f_list = df_trunc['p_est'].values

    # Initial parameters to optimize.
    f_0 = df_trunc[df_trunc['probability'] == p_nearest]['p_est'].mean()
    if pd.isna(f_0):
        f_0 = df_trunc['p_est'].mean()
    params_0 = [p_nearest, 2, f_0, 1, 1]

    try:
        params_opt = get_fit_params(
            p_list, d_list, f_list, params_0=params_0, ftol=ftol_est,
            maxfev=maxfev
        )
    except RuntimeError as err:
        print('fitting failed')
        print(err)
        params_opt = np.array([np.nan]*5)

    df_trunc['rescaled_p'] = rescale_prob([p_list, d_list], *params_opt)

    # Bootstrap resampling parameters.
    rng = np.random.default_rng(0)
    params_bs_list = []
    for i_bs in range(n_bs):

        # Sample from Beta distribution.
        f_list_bs = []
        for i in range(df_trunc.shape[0]):
            n_trials = int(df_trunc['n_trials'].iloc[i])
            n_fail = int(df_trunc['n_fail'].iloc[i])
            if n_fail == 0:
                n_fail = 1
            f_list_bs.append(
                rng.beta(n_fail, n_trials - n_fail)
            )
        f_bs = np.array(f_list_bs)

        try:
            params_bs_list.append(
                get_fit_params(
                    p_list, d_list, f_bs, params_0=params_opt, ftol=ftol_std,
                    maxfev=maxfev
                )
            )
        except RuntimeError:
            print('bootstrap fitting failed')
            params_bs_list.append(np.array([np.nan]*5))
    params_bs = np.array(params_bs_list)
    return params_opt, params_bs, df_trunc


def get_error_model_df(results_df):
    """Get error models."""
    error_model_df = results_df[[
        'error_model', 'noise_direction'
    ]].drop_duplicates().sort_values(by='noise_direction')

    r_xyz = pd.DataFrame(
        error_model_df['noise_direction'].tolist(),
        index=error_model_df.index,
        columns=['r_x', 'r_y', 'r_z']
    )

    projected_triangle = pd.DataFrame(
        error_model_df['noise_direction'].apply(project_triangle).tolist(),
        index=error_model_df.index,
        columns=['h', 'v']
    )

    error_model_df = pd.concat([
        error_model_df, r_xyz, projected_triangle
    ], axis=1)

    error_model_df['eta_x'] = error_model_df['r_x']/(
        error_model_df['r_y'] + error_model_df['r_z']
    )
    error_model_df['eta_y'] = error_model_df['r_y']/(
        error_model_df['r_x'] + error_model_df['r_z']
    )
    error_model_df['eta_z'] = error_model_df['r_z']/(
        error_model_df['r_x'] + error_model_df['r_y']
    )
    return error_model_df


def get_thresholds_df(results_df, ftol_est=1e-5, ftol_std=1e-5, maxfev=2000):
    thresholds_df = get_error_model_df(results_df)
    p_th_sd = []
    p_th_nearest = []
    p_left = []
    p_right = []
    fss_params = []
    p_th_fss_left = []
    p_th_fss_right = []
    p_th_fss_se = []
    df_trunc_list = []
    params_bs_list = []
    for error_model in thresholds_df['error_model']:
        df_filt = results_df[results_df['error_model'] == error_model]

        # Find nearest value where crossover changes.
        p_th_nearest_val = get_p_th_nearest(df_filt)
        p_th_nearest.append(p_th_nearest_val)

        # More refined crossover using standard deviation heuristic.
        p_th_sd_val, p_left_val, p_right_val = get_p_th_sd_interp(
            df_filt, p_nearest=p_th_nearest_val
        )
        p_th_sd.append(p_th_sd_val)

        # Left and right bounds to truncate.
        p_left.append(p_left_val)
        p_right.append(p_right_val)

        # Finite-size scaling fitting.
        params_opt, params_bs, df_trunc = fit_fss_params(
            df_filt, p_left_val, p_right_val, p_th_nearest_val,
            ftol_est=ftol_est, ftol_std=ftol_std, maxfev=maxfev
        )
        fss_params.append(params_opt)

        # 1-sigma error bar bounds.
        p_th_fss_left.append(np.quantile(params_bs[:, 0], 0.16))
        p_th_fss_right.append(np.quantile(params_bs[:, 0], 0.84))

        # Standard error.
        p_th_fss_se.append(params_bs[:, 0].std())

        # Trucated data.
        df_trunc_list.append(df_trunc)

        # Bootstrap parameters sample list.
        params_bs_list.append(params_bs)

    thresholds_df['p_th_sd'] = p_th_sd
    thresholds_df['p_th_nearest'] = p_th_nearest
    thresholds_df['p_left'] = p_left
    thresholds_df['p_right'] = p_right
    thresholds_df['p_th_fss'] = np.array(fss_params)[:, 0]
    thresholds_df['p_th_fss_left'] = p_th_fss_left
    thresholds_df['p_th_fss_right'] = p_th_fss_right
    thresholds_df['p_th_fss_se'] = p_th_fss_se
    thresholds_df['fss_params'] = list(map(tuple, fss_params))

    trunc_results_df = pd.concat(df_trunc_list, axis=0)
    return thresholds_df, trunc_results_df, params_bs_list


def export_summary_table_latex(
    bias_direction, deformed, summary_df_part, table_dir=None
):
    latex_lines = (
        summary_df_part.drop('Bias', axis=1)
        .to_latex(index=False, escape=False).split('\n')
    )
    if latex_lines[-1] == '':
        latex_lines.pop(-1)
    latex_lines.append(
        r'\caption{Thresholds for $%s$ bias in %s 3D toric code.}%%' % (
            bias_direction.upper(),
            {True: 'deformed', False: 'undeformed'}[deformed]
        )
    )
    table_name = 'thresh%s%s' % (
        bias_direction,
        {True: 'deformed', False: 'undeformed'}[deformed]
    )
    latex_lines.append(
        r'\label{tab:%s}' % table_name
    )
    latex_str = '\n'.join(latex_lines)
    if table_dir is not None:
        table_tex = os.path.join(table_dir, f'{table_name}.tex')
        with open(table_tex, 'w') as f:
            f.write(latex_str)
    return latex_str


def export_summary_tables(thresholds_df, table_dir=None, verbose=True):
    summary_df_list = []
    summary_df_latex = []
    for bias_direction, deformed in itertools.product(
        ['x', 'y', 'z'], [True, False]
    ):
        eta_key = f'eta_{bias_direction}'
        deform_filter = thresholds_df['error_model'].str.contains('Deformed')
        if not deformed:
            deform_filter = ~deform_filter
        summary_df_part = thresholds_df[
            deform_filter
            & (thresholds_df[eta_key] >= 0.5)
        ].sort_values(by=eta_key).copy()
        summary_df_part['Bias'] = r'$%s$' % bias_direction.upper()
        summary_df_part[eta_key] = summary_df_part[eta_key].map(
            '{:.1f}'.format
        ).replace('inf', r'$\infty$', regex=False)
        # str.replace('inf', r'$\infty$', regex=False)
        summary_df_part['p_th_latex'] = summary_df_part[[
            'p_th_fss', 'p_th_fss_se'
        ]].apply(
            lambda x: fmt_uncertainty(x[0]*100, x[1]*100, unit=r'\%'), axis=1
        )
        summary_df_part = summary_df_part[[
            'Bias', eta_key, 'r_x', 'r_y', 'r_z', 'p_th_latex',
        ]]
        for k in ['r_x', 'r_y', 'r_z']:
            summary_df_part[k] = summary_df_part[k].round(4)
        summary_df_part.columns = [
            'Bias', r'$\eta$', r'$r_X$', r'$r_Y$', r'$r_Z$',
            r'$p_{\mathrm{th}}$ ($\%$)'
        ]
        summary_df_list.append(summary_df_part)
        summary_df_latex.append(
            export_summary_table_latex(
                bias_direction, deformed,
                summary_df_part, table_dir=table_dir
            )
        )
        if verbose:
            print(summary_df_latex[-1])
    return summary_df_list, summary_df_latex
