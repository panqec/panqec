"""
Routines for analysing output data.

:Author:
    Eric Huang
"""

import os
import warnings
from typing import List, Optional, Tuple, Union, Callable, Iterable, Any, Dict
import json
import re
import gzip
from zipfile import ZipFile
import itertools
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from .config import SLURM_DIR
from .simulation import read_input_json, BatchSimulation
from .utils import fmt_uncertainty, NumpyEncoder, identity
from .bpauli import int_to_bvector, bvector_to_pauli_string

Numerical = Union[Iterable, float, int]


# TODO make this the legit way of doing things.
class Analysis:
    """Analysis on large collections of results files.

    This is the preferred method because it does not require reading the input
    files since the input parameters are saved to the results files anwyay.
    It also does not create Simulation objects for every data point, which
    could be slow.

    Parameters
    ----------
    results : Union[str, List[str]]
        Path to directory or .zip containing .json.gz or .json results.
        Can also accept list of paths.

    Attributes
    ----------
    results : pd.DataFrame
        Results for each set of (code, error model and decoder).
    thresholds : pd.DataFrame
        Thresholds for each (code family, error_model and decoder).
    """

    # List of paths where results are to be analyzed.
    results_paths: List[str] = []

    # Locations of individual results files.
    file_locations: List[Union[str, Tuple[str, str]]] = []

    # Raw data extracted from files.
    raw: pd.DataFrame

    # Results of each code, error_modle and decoder.
    results: pd.DataFrame

    # Thresholds for each code family, error modle and decoder.
    thresholds: pd.DataFrame

    # Prints more details if true.
    verbose: bool = True

    # Set of parameters that are unique to each input.
    INPUT_KEYS: List[str] = [
        'size', 'code', 'n', 'k', 'd', 'error_model', 'decoder',
        'probability'
    ]

    def __init__(
        self, results: Union[str, List[str]] = [], verbose: bool = True
    ):
        self.results_paths = []
        if isinstance(results, list):
            self.results_paths += results
        else:
            self.results_paths.append(results)

    def log(self, message: str):
        """Display a message.

        Parameters
        ----------
        message : str
            Message to be displayed.
        """
        if self.verbose:
            print(message)

    def analyze(self, progress: Optional[Callable] = identity):
        """Perform the full analysis.

        Parameters
        ----------
        progress : Optional[Callable]
            Progress bar indicator such as tqdm or its notebook variant.
        """
        self.find_files()
        self.count_files()
        self.read_files(progress=progress)
        self.aggregate()
        self.calculate_total_error_rates()
        self.calculate_word_error_rates()
        self.calculate_single_qubit_error_rates()
        self.calculate_total_thresholds()
        self.calculate_single_qubit_sector_thresholds()

    def find_files(self):
        """Find where the results files are."""

        self.log('Finding files')

        # List of paths to json files or tuples of zip file and json.
        file_locations: List[str, Tuple[str, str]] = []

        for results_path in self.results_paths:

            # Look for .zip files that may contain .json.gz or .json files
            # inside.
            zip_files: List[Any] = []

            # Also look for standalone .json or .json.gz results files.
            json_files: List[Any] = []

            # Recursively look for .zip, .json.gz and .json files.
            if os.path.isdir(results_path):
                results_dir = results_path
                for path in Path(results_dir).rglob('*.zip'):
                    zip_files.append(path)
                for path in Path(results_dir).rglob('*.json.gz'):
                    json_files.append(path)
                for path in Path(results_dir).rglob('*.json'):
                    json_files.append(path)

            # results_path may also be a results file or .zip.
            elif '.zip' in results_path:
                zip_files.append(results_path)
            elif '.json' in results_path:
                json_files.append(results_path)

            # Look for .json and .json.gz files inside .zip archives.
            for zip_file in zip_files:
                zf = ZipFile(zip_file)
                for zip_path in zf.filelist:
                    if '.json' in zip_path.filename:
                        file_locations.append((zip_file, zip_path.filename))

            # Add json paths directly.
            file_locations += json_files

        self.file_locations = file_locations

        self.log(f'Found {len(file_locations)} files')

        return self.file_locations

    def count_files(self):
        """Count how many files were found."""
        return len(self.file_locations)

    def read_files(self, progress=identity):
        """Read raw data from the files that were found."""

        self.log('Reading files')
        entries = []
        for file_location in progress(self.file_locations):
            if isinstance(file_location, tuple):
                zip_file, results_file = file_location
                nominal_path = os.path.join(
                    os.path.abspath(zip_file), results_file
                )
                zf = ZipFile(zip_file)
                if '.json.gz' in results_file:
                    with zf.open(results_file) as f:
                        with gzip.open(f, 'rb') as g:
                            data = json.loads(g.read().decode('utf-8'))
                else:
                    with zf.open(results_file) as f:
                        data = json.load(f)
            elif '.json.gz' in str(file_location):
                nominal_path = os.path.abspath(file_location)
                with gzip.open(file_location, 'rb') as g:
                    data = json.loads(g.read().decode('utf-8'))
            else:
                nominal_path = os.path.abspath(file_location)
                with open(file_location) as jf:
                    data = json.load(jf)
            entries += read_entry(data, results_file=nominal_path)
        self.raw = pd.DataFrame(entries)

        # Round probability to six digits, because anything smaller than that
        # is likely numerical error.
        self.raw['probability'] = self.raw['probability'].round(6)

    def aggregate(self):
        """Aggregate the raw data into results attribute."""
        self.log('Aggregating data')

        # Input keys by which data is to be grouped.
        grouped_df = self.raw.groupby(self.INPUT_KEYS)

        # Columns for which grouped values are to be summed.
        added_columns = grouped_df[[
            'wall_time', 'n_trials'
        ]].sum()

        # Columns for which grouped entries are to be concantenated np arrays.
        concat_columns = grouped_df[[
            'effective_error', 'success', 'codespace'
        ]].aggregate(lambda x: np.concatenate(x.values))

        # Columns for which the any value out of the grouped values can be
        # taken.
        take_first_columns = grouped_df[['bias']].first()

        # Columns to be grouped and turned into lists.
        list_columns = grouped_df[['results_file']].aggregate(list)

        # Stack the columns together side by side to form a big table.
        self.results = pd.concat([
            added_columns, concat_columns, take_first_columns, list_columns
        ], axis=1).reset_index()

        self.results['n_fail'] = self.results['success'].apply(sum)
        self.results['code_family'] = self.results['code'].apply(
            infer_code_family
        )

    def calculate_total_error_rates(self):
        """Calculate the total error rate.

        And add it as a column to the results attribute of this class.
        """
        self.log('Calculating total error rates')
        estimates_list = []
        uncertainties_list = []
        for i_entry, entry in self.results.iterrows():
            estimator = 1 - entry['success'].mean()
            uncertainty = get_standard_error(estimator, entry['n_trials'])
            estimates_list.append(estimator)
            uncertainties_list.append(uncertainty)
        self.results['p_est'] = estimates_list
        self.results['p_se'] = uncertainties_list

    def calculate_word_error_rates(self):
        """Calculate the word error rate using the formula.

        The formula assumes a uniform error rate across all logical qubits.
        """
        self.log('Calculating word error rates')
        p_est = self.results['p_est']
        p_se = self.results['p_se']
        k = self.results['k']
        p_word_est, p_word_se = get_word_error_rate(p_est, p_se, k)
        self.results['p_word_est'] = p_word_est
        self.results['p_word_se'] = p_word_se

    def calculate_single_qubit_error_rates(self):
        """Add single error rate estimates and uncertainties to results.

        Adds 'single_qubit_p_est' and 'single_qubit_p_se' as array-valued
        columns to the results attribute of this class.

        Each entry is an array of shape (k, 4),
        with the i-th row corresponding to the value for i-th logical qubit
        and the column 0 containing values for the total error rate,
        column 1 for X, column 2 for Y, and column 3 for Y errors.

        Note that it is not checked whether or not the code is in the code
        space.
        """
        self.log('Calculating single qubit error rates')

        # Calculate single-qubit error rates.
        estimates_list = []
        uncertainties_list = []
        for i_entry, entry in self.results.iterrows():
            estimates = np.zeros((entry['k'], 4))
            uncertainties = np.zeros((entry['k'], 4))
            for i in range(entry['k']):
                for i_pauli, pauli in enumerate([None, 'X', 'Y', 'Z']):
                    estimate, uncertainty = get_single_qubit_error_rate(
                        entry['effective_error'], i=i, error_type=pauli,
                    )
                    estimates[i, i_pauli] = estimate
                    uncertainties[i, i_pauli] = uncertainty
            estimates_list.append(estimates)
            uncertainties_list.append(estimates)
        self.results['single_qubit_p_est'] = estimates_list
        self.results['single_qubit_p_se'] = uncertainties_list

    def calculate_total_thresholds(self, **kwargs):
        """Calculate thresholds for total error rate."""
        self.log('Calculating total thresholds')
        thresholds_df, trunc_results_df, params_bs_list = get_thresholds_df(
            self.results, **kwargs
        )
        self.thresholds_df = thresholds_df
        self.trunc_results_df = trunc_results_df
        self.params_bs_list = params_bs_list

    def calculate_single_qubit_sector_thresholds(self, **kwargs):
        """Calculate single-qubit thresholds for each sector."""
        self.log('Calculating single-qubit sector thresholds thresholds')
        pass


def infer_code_family(label: str) -> str:
    """Infer the code family from the code label.

    Parameters
    ----------
    label : str
        The code label.

    Returns
    -------
    family : str
        The code family.
        If the code family cannot be inferred,
        the original code label is returned.

    Examples
    --------
    >>> infer_code_family('Toric 4x4')
    'Toric'
    >>> infer_code_family('Rhombic 10x10x10')
    'Rhombic'
    """
    family = label
    dimension_pattern = r'\d+x\d+(x\d+)?'
    family = re.sub(dimension_pattern, '', family).strip()
    return family


def get_standard_error(estimator, n_samples):
    """Get the standard error of mean estimator.

    Parameters
    ----------
    estimator : float
        Number of hits divided by number of samples.
    n_samples : int
        Number of samples.

    Returns
    -------
    standard_error : float
        The standard error taken as the standard deviation of a Beta
        distribution.
    """
    return np.sqrt(estimator*(1 - estimator)/(n_samples + 1))


def get_results_df_from_batch(
    batch_sim: BatchSimulation, batch_label: str
) -> pd.DataFrame:
    """Get results DataFrame directly from a BatchSimulation.
    (As opposed to from a file saved to disk.)

    Parameters
    ----------
    batch_sim : BatchSimulation
        The object to extract data from.
    batch_label : str
        The label to put in the table.

    Returns
    ------
    results_df : pd.DataFrame
        The results for each (code, error_model, decoder).
    """
    batch_results = batch_sim.get_results()
    # print(
    #     'wall_time =',
    #     str(datetime.timedelta(seconds=batch_sim.wall_time))
    # )
    # print('n_trials = ', min(sim.n_results for sim in batch_sim))
    for sim, batch_result in zip(batch_sim, batch_results):
        n_logicals = batch_result['k']

        # Small fix for the current situation. TO REMOVE in later versions
        if n_logicals == -1:
            n_logicals = 1

        batch_result['label'] = batch_label
        batch_result['noise_direction'] = sim.error_model.direction
        if len(sim.results['effective_error']) > 0:
            batch_result['p_x'] = np.array(
                sim.results['effective_error']
            )[:, :n_logicals].any(axis=1).mean()
            batch_result['p_x_se'] = get_standard_error(
                batch_result['p_x'], sim.n_results
            )
            batch_result['p_z'] = np.array(
                sim.results['effective_error']
            )[:, n_logicals:].any(axis=1).mean()
            batch_result['p_z_se'] = get_standard_error(
                batch_result['p_z'], sim.n_results
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
    input_dir: str = None,
) -> pd.DataFrame:
    """Get raw results in DataFrame from list of jobs in output dir.

    This is an old legacy way of doing things,
    because it requires the input files as well.
    Use the Analysis object instead if possible.

    Parameters
    ----------
    job_list : List[str]
        List of folder names in output_dir to extract results from.
    output_dir : str
        The path to the output directory containing the directories with names
        as listed in job_list.
    input_dir : str
        The directory where the input json files are stored.
    """

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

        # Determine whether or not it's a onefile format.
        onefile = False
        output_file_list = os.listdir(output_dirs[i])
        if len(output_file_list) == 1:
            if '.json.gz' in output_file_list[0]:
                results_onefile = os.path.join(
                    output_dirs[i], output_file_list[0]
                )
                with gzip.open(results_onefile, 'rb') as g:
                    data = json.loads(g.read().decode('utf-8'))
                if isinstance(data, list):
                    onefile = True

        if onefile:
            batch_sim = read_input_json(input_files[i])
            input_jsons = [json.dumps(element['inputs']) for element in data]
            for sim in batch_sim:
                sim_input_json = json.dumps(
                    sim.get_results_to_save()['inputs'],
                    cls=NumpyEncoder
                )
                matching_indices = [
                    i
                    for i, value in enumerate(input_jsons)
                    if value == sim_input_json
                ]
                if matching_indices:
                    sim.load_results_from_dict(data[matching_indices[0]])
            batches[batch_sim.label] = batch_sim

        else:
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
            n_logicals = batch_result['k']

            # Small fix for the current situation. TO REMOVE in later versions
            if n_logicals == -1:
                n_logicals = 1

            batch_result['label'] = batch_label
            batch_result['noise_direction'] = sim.error_model.direction

            eta_x, eta_y, eta_z = get_bias_ratios(sim.error_model.direction)
            batch_result['eta_x'] = eta_x
            batch_result['eta_y'] = eta_y
            batch_result['eta_z'] = eta_z
            batch_result['wall_time'] = sim._results['wall_time']

            if len(sim.results['effective_error']) > 0:
                codespace = np.array(sim.results['codespace'])
                x_errors = np.array(
                    sim.results['effective_error']
                )[:, :n_logicals].any(axis=1)
                batch_result['p_x'] = x_errors.mean()
                batch_result['p_x_se'] = get_standard_error(
                    batch_result['p_x'], sim.n_results
                )

                z_errors = np.array(
                    sim.results['effective_error']
                )[:, n_logicals:].any(axis=1)
                batch_result['p_z'] = z_errors.mean()
                batch_result['p_z_se'] = get_standard_error(
                    batch_result['p_z'], sim.n_results
                )
                batch_result['p_undecodable'] = (~codespace).mean()

                i_logical = 0

                # Single-qubit rate estimates and uncertainties.
                single_qubit_results = dict()
                p_est, p_se = get_single_qubit_error_rate(
                    sim.results['effective_error'], i=i_logical,
                    error_type=None
                )
                single_qubit_results.update({
                    f'p_{i_logical}_est': p_est,
                    f'p_{i_logical}_se': p_se,
                })
                for pauli in 'XYZ':
                    p_pauli_est, p_pauli_se = get_single_qubit_error_rate(
                        sim.results['effective_error'], i=i_logical,
                        error_type=pauli
                    )
                    single_qubit_results.update({
                        f'p_{i_logical}_{pauli}_est': p_pauli_est,
                        f'p_{i_logical}_{pauli}_se': p_pauli_se,
                    })
                batch_result.update(single_qubit_results)
                assert np.isclose(
                    single_qubit_results[f'p_{i_logical}_X_est']
                    + single_qubit_results[f'p_{i_logical}_Y_est']
                    + single_qubit_results[f'p_{i_logical}_Z_est'],
                    single_qubit_results[f'p_{i_logical}_est']
                )

            else:
                batch_result['p_x'] = np.nan
                batch_result['p_x_se'] = np.nan
                batch_result['p_z'] = np.nan
                batch_result['p_z_se'] = np.nan
                batch_result['p_undecodable'] = np.nan
        results += batch_results

    results_df = pd.DataFrame(results)

    # Calculate the word error rate and standard error.
    error_rate_labels = [
        ('p_est', 'p_se'),
        ('p_x', 'p_x_se'),
        ('p_z', 'p_z_se'),
    ]
    for p_L, p_L_se in error_rate_labels:
        (
            results_df[f'{p_L}_word'], results_df[f'{p_L_se}_word']
        ) = get_word_error_rate(
            results_df[p_L], results_df[p_L_se], results_df['k']
        )

    return results_df


def get_single_qubit_error_rate(
    effective_error_list: Union[List[List[int]], np.ndarray],
    i: int = 0,
    error_type: Optional[str] = None,
) -> Tuple[float, float]:
    """Estimate single-qubit error rate of i-th qubit and its standard error.

    This is the probability of getting an error on the i-th logical qubit,
    marginalized over the other logical qubits.

    Parameters
    ----------
    effective_error_list :
        List of many logical effective errors produced by simulation,
        each of which is given in bsf format.
    n : int
        The index of the logical qubit on which estimation is to be done.
    error_type :
        Type of Pauli error to calculate error for, i.e. 'X', 'Y' or 'Z'
        If None is given, then rate for any error is estimated.

    Returns
    -------
    p_i_est : float
        Single-qubit error rate estimator.
    p_i_se : float
        Standard error for the estimator.
    """
    p_est = np.nan
    p_se = np.nan

    # Convert to numpy array.
    effective_errors = np.array(effective_error_list)

    # Return nan if wrong shape given.
    if len(effective_errors.shape) != 2:
        return p_est, p_se

    # Number of logical qubits and sample size.
    k = int(effective_errors.shape[1]/2)
    n_results = effective_errors.shape[0]

    # Errors on the single logical qubit of interest.
    qubit_errors = np.array(
        [effective_errors[:, i], effective_errors[:, k + i]]
    ).T

    # Calculate error rate based on error type.
    if error_type is None:
        p_est = 1 - (qubit_errors == [0, 0]).all(axis=1).mean()
    elif error_type == 'X':
        p_est = (qubit_errors == [1, 0]).all(axis=1).mean()
    elif error_type == 'Y':
        p_est = (qubit_errors == [1, 1]).all(axis=1).mean()
    elif error_type == 'Z':
        p_est = (qubit_errors == [0, 1]).all(axis=1).mean()

    # Beta distribution assumed.
    p_se = get_standard_error(p_est, n_results)

    return p_est, p_se


def get_word_error_rate(p_est, p_se, k) -> Tuple:
    """Calculate the word error rate and its standard error.

    Parameters
    ----------
    p_est : Numerical
        Value or array of estimated logical error rate.
    p_se : Numerical
        Value or array of standard error on logical error rate.
    k : Numerical
        Number of logical qubits, as value or array.

    Returns
    -------
    p_est_word : Numerical
        Value or array of estimated word error rate.
    p_se_word : Numerical
        Value or array of standard error of word error rate.
    """
    p_est_word = 1 - (1 - p_est)**(1/k)
    p_se_word = (1/k)*(1 - p_est)**(1/k - 1)*p_se
    return p_est_word, p_se_word


def get_logical_rates_df(
    job_list, input_dir, output_dir, progress: Optional[Callable] = None
):
    """Get DataFrame of logical error rates for each logical error.

    This is superseded by the Analysis class.
    """
    if progress is None:
        def progress_func(x, total: int = 0):
            return x
    else:
        progress_func = progress

    input_files = [
        os.path.join(input_dir, f'{name}.json')
        for name in job_list
    ]
    output_dirs = [
        os.path.join(output_dir, name)
        for name in job_list
    ]

    arguments = list(zip(input_files, output_dirs))

    with Pool(cpu_count()) as pool:
        data = pool.starmap(
            extract_logical_rates,
            progress_func(arguments, total=len(arguments))
        )

    data = [entry for entries in data for entry in entries]
    df = pd.DataFrame(data)
    return df


def extract_logical_rates(input_file, output_dir):
    """Extract logical error rates from results.
    Superseded by Analysis class.
    """
    batch_sim = read_input_json(input_file)

    data = []
    for sim in batch_sim:
        sim.load_results(output_dir)
        batch_result = sim.get_results()
        entry = {
            'label': batch_sim.label,
            'noise_direction': sim.error_model.direction,
            'probability': batch_result['probability'],
            'size': batch_result['size'],
            'n': batch_result['n'],
            'k': batch_result['k'],
            'd': batch_result['d'],
        }

        n_logicals = batch_result['k']

        # Small fix for the current situation. TO REMOVE in later versions
        if n_logicals == -1:
            n_logicals = 1

        # All possible logical errors.
        possible_logical_errors = [
            int_to_bvector(int_rep, n_logicals)
            for int_rep in range(1, 2**(2*n_logicals))
        ]

        for logical_error in possible_logical_errors:
            pauli_string = bvector_to_pauli_string(logical_error)
            p_est_label = f'p_est_{pauli_string}'
            p_se_label = f'p_se_{pauli_string}'
            p_est_logical = (
                np.array(sim.results['effective_error'])
                == logical_error
            ).all(axis=1).mean()
            entry[p_est_label] = p_est_logical
            entry[p_se_label] = get_standard_error(
                p_est_logical, sim.n_results
            )
        data.append(entry)
    return data


def get_p_th_sd_interp(
    df_filt: pd.DataFrame,
    p_nearest: Optional[float] = None,
    p_est: str = 'p_est',
) -> Tuple[float, float, float]:
    """Estimate threshold by where SD of p_est is local min.

    This is a very coarse heuristic to estimate roughly where the crossover
    point is, if there is one and the left and right limits where one should
    truncate the data.
    This can be used as a starting point for something more precise,
    finite-size scaling.
    The rationale is that away from the crossing point,
    the lines of p_est vs p should get wider and wider vertically,
    but if they start getting narrower again,
    then that's a sign that we're moving to a different regime and finite-size
    scaling is not likely to work, so we should throw away those data points.

    Parameters
    ----------
    df_filt : pd.DataFrame
        Results with columns: 'probability', 'code', `p_est`.
        The 'probability' column is the physical error rate p.
        The 'code' column is the code label.
        The `p_est` column is the logical error rate.
    p_nearest : Optional[float]
        A hint for the nearest 'probability' value that is close to the
        threshold, to be used as a starting point for searching.
    p_est : str
        The column in the `df_filt` DataFrame that is to be used as the logical
        error rate to estimate the threshold.

    Returns
    -------
    p_crossover : float
        The apparent crossover point where the plot of p_est vs p for each code
        becomes narrowest vertically.
    p_left : float
        The left limit where when moving left from p_crossover,
        the spread of the p_est vs p for all the codes becomes widest.
        Data to the left of this point is recommended to be truncated out.
    p_right : float
        The right limit where when moving right from p_crossover,
        the spread of the p_est vs p for all the codes becomes widest.
        Data to the right of this point is recommended to be truncated out.
    """

    # Points to interpolate at.
    interp_res = 0.001
    p_min = df_filt['probability'].min()
    p_max = df_filt['probability'].max()
    if p_nearest is not None:
        if p_nearest > p_min:
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
        if df_filt_code.shape[0] > 1:
            interpolator = interp1d(
                df_filt_code['probability'], df_filt_code[p_est],
                fill_value="extrapolate"
            )
            curves[code] = interpolator(p_interp)

        # Use a straight horizontal line if only one point available.
        else:
            curves[code] = np.ones_like(p_interp)*df_filt_code[p_est].iloc[0]

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
    if not any(i_maxima < i_crossover):
        p_left = p_interp[i_maxima[0]]
    else:
        p_left = p_interp[i_maxima[i_maxima < i_crossover].max()]
    if not any(i_crossover < i_maxima):
        p_right = p_interp[i_maxima[-1]]
    else:
        p_right = p_interp[i_maxima[i_crossover < i_maxima].min()]

    # Crossover is the error rate value for that point.
    p_crossover = p_interp[i_crossover]
    return p_crossover, p_left, p_right


def get_code_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame of all codes available.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table with columns 'code', 'n', 'k', 'd',
        plus possibly other columns.

    Returns
    -------
    code_df : pd.DataFrame
        DataFrame with only only ['code', 'n', 'k', 'd'] as columns,
        with no duplicates.
    """
    code_df = results_df[['code', 'n', 'k', 'd']].copy()
    code_df = code_df.drop_duplicates().reset_index(drop=True)
    code_df = code_df.sort_values(by='n').reset_index(drop=True)
    return code_df


def longest_sequence(arr, char):
    """Find longest continuous sequence of chars in an array.

    Parameters
    ----------
    arr : Iterable
        An array possibly containing `char`.
    char : Any
        We are looking for sequences of this char.

    Returns
    -------
    best_seq_start : int
        Where the longest sequence of `char` starts.
    best_seq_end : int
        Where the longest sequence of `char` ends.
    """
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


def get_p_th_nearest(df_filt: pd.DataFrame, p_est: str = 'p_est') -> float:
    """Estimate which p in the results is nearest to the threshold.

    This is very very rough heuristic by going along each value of the physical
    error rate p in the plots of p_est vs p for different code sizes and seeing
    when the order of the lines changes.
    The point where it starts to change is deemed p_th_nearest.
    This is super rough, so a more refined method such as finite-sized scaling
    will be required to make it more precise and put uncertainties around it.

    Parameters
    ----------
    df_filt : pd.DataFrame
        Results with columns: 'probability', 'code', `p_est`.
        The 'probability' column is the physical error rate p.
        The 'code' column is the code label.
        The `p_est` column is the logical error rate.
    p_est : str
        The column in the `df_filt` DataFrame that is to be used as the logical
        error rate to estimate the threshold.

    Returns
    -------
    p_th_nearest : float
        The value in the `probability` that is apparently the closes to the
        threshold.
    """
    code_df = get_code_df(df_filt)

    # Estimate the threshold by where the order of the lines change.
    p_est_df = pd.DataFrame({
        code: dict(df_filt[df_filt['code'] == code][[
            'probability', p_est
        ]].values)
        for code in code_df['code']
    })
    p_est_df = p_est_df.sort_index()

    try:
        i_order_change = np.diff(
            np.diff(
                np.argsort(p_est_df.values, axis=1)
            ).sum(axis=1)
        ).argmax()
        p_th_nearest = p_est_df.index[i_order_change]
    except ValueError:
        p_th_nearest = p_est_df.index[0]

    return p_th_nearest


def fit_function(x_data, *params):
    """Quadratic fit function for finite-size scaling. """
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
    """Rescaled physical error rate."""
    p, d = x_data
    p_th, nu, A, B, C = params
    x = (p - p_th)*d**nu
    return x


def get_fit_params(
    p_list: np.ndarray, d_list: np.ndarray, f_list: np.ndarray,
    params_0: Optional[Union[np.ndarray, List]] = None,
    ftol: float = 1e-5, maxfev: int = 2000
) -> np.ndarray:
    """Get fitting params.

    Parameters
    ----------
    p_list : np.ndarray
        List of physical error rates.
    d_list : np.ndarray
        List of code distances.
    f_list : np.ndarray
        List of logical error rates.
    params_0 : Optional[Union[np.ndarray, List]]
        Hint parameters for the optimizer about where to start minimizing cost
        function.
    ftol : float
        Tolerance for the optimizer.
    maxfev : int
        Maximum number of iterations for the optimizer.

    Returns
    -------
    params_opt : Tuple[float]
        The optimized parameters that fits the data best.
    """

    # Curve fitting inputs.
    x_data = np.array([
        p_list,
        d_list
    ])

    # Target outputs.
    y_data = f_list

    # Curve fit.
    bounds = [min(x_data[0]), max(x_data[0])]

    if params_0 is not None and params_0[0] not in bounds:
        params_0[0] = (bounds[0] + bounds[1]) / 2

    # print("Bounds", bounds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params_opt, _ = curve_fit(
            fit_function, x_data, y_data,
            p0=params_0, ftol=ftol, maxfev=maxfev
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
    maxfev: int = 2000,
    p_est: str = 'p_est',
    n_fail_label: str = 'n_fail',
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Get optimized parameters and data table tweaked with heuristics.

    Parameters
    ----------
    df_filt : pd.DataFrame
        Results with columns: 'probability', 'code', `p_est`, 'n_trials',
        `n_fail_label`.
        The 'probability' column is the physical error rate p.
        The 'code' column is the code label.
        The `p_est` column is the logical error rate.
    p_left_val : float
        The left left value of 'probability' to truncate.
    p_right_val : float
        The left right value of 'probability' to truncate.
    p_nearest : float
        The nearest value of 'probability' to what was previously roughly
        estimated to be the threshold.
    n_bs : int
        The number of bootstrap samples to take.
    ftol_est : float
        Tolerance for the best fit.
    ftol_std : float
        Tolerance for the bootstrapped fits.
    maxfev : int
        Maximum iterations for curve fitting optimizer.
    p_est : str
        Label for the logical error rate to use.
    n_fail_label : str
        Label for the number of logical fails to use.

    Returns
    -------
    params_opt : np.ndarray
        Array of optimized parameters
    params_bs : np.ndarray
        Array with each row being arrays of optimized parameters for each
        bootstrap resample.
    df_trunc : pd.DataFrame
        The truncated DataFrame used for performing the curve fitting.
    """
    # Truncate error probability between values.
    df_trunc = df_filt[
        (p_left_val <= df_filt['probability'])
        & (df_filt['probability'] <= p_right_val)
    ].copy()
    df_trunc = df_trunc.dropna(subset=[p_est])

    d_list = df_trunc['d'].values
    p_list = df_trunc['probability'].values
    f_list = df_trunc[p_est].values

    # Initial parameters to optimize.
    f_0 = df_trunc[df_trunc['probability'] == p_nearest][p_est].mean()
    if pd.isna(f_0):
        f_0 = df_trunc[p_est].mean()
    params_0 = [p_nearest, 2, f_0, 1, 1]

    try:
        params_opt = get_fit_params(
            p_list, d_list, f_list, params_0=params_0, ftol=ftol_est,
            maxfev=maxfev
        )
    except (RuntimeError, TypeError) as err:
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
            n_fail = int(df_trunc[n_fail_label].iloc[i])
            if n_fail == 0:
                n_fail = 1
            if n_fail == n_trials:
                f_list_bs.append(0.5)
            else:
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
        except (RuntimeError, TypeError):
            params_bs_list.append(np.array([np.nan]*5))
    params_bs = np.array(params_bs_list)
    return params_opt, params_bs, df_trunc


def get_bias_ratios(noise_direction):
    """Get the bias ratios in each direction given the noise direction.

    Parameters
    ----------
    noise_direction : (float, float, float)
        The (r_x, r_y, r_z) parameters of the Pauli channel.

    Returns
    -------
    eta_x : float
        The X bias ratio.
    eta_y : float
        The Y bias ratio.
    eta_z : float
        The Z bias ratio.
    """
    r_x, r_y, r_z = noise_direction

    if r_y + r_z != 0:
        eta_x = r_x / (r_y + r_z)
    else:
        eta_x = np.inf

    if r_x + r_z != 0:
        eta_y = r_y / (r_x + r_z)
    else:
        eta_y = np.inf

    if r_x + r_y != 0:
        eta_z = r_z / (r_x + r_y)
    else:
        eta_z = np.inf

    return eta_x, eta_y, eta_z


def deduce_noise_direction(error_model: str) -> Tuple[float, float, float]:
    """Deduce the noise direction given the error model label.

    Parameters
    ----------
    error_model : str
        Label of the error model.

    Returns
    -------
    r_x : float
        The component in the Pauli X direction.
    r_y : float
        The component in the Pauli Y direction.
    r_z : float
        The component in the Pauli Z direction.
    """
    direction = (0.0, 0.0, 0.0)
    match = re.search(r'Pauli X([\d\.]+)Y([\d\.]+)Z([\d\.]+)', error_model)
    if match:
        direction = (
            float(match.group(1)), float(match.group(2)), float(match.group(3))
        )
    return direction


def get_error_model_df(results_df):
    """Get DataFrame error models and noise parameters.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns 'error_model'

    Returns
    -------
    error_model_df : pd.DataFrame
        DataFrame with columns:
        'code_family', 'error_model', 'decoder',
        'noise_direction',
        'r_x', 'r_y', 'r_z',
        'eta_x', 'eta_y', 'eta_z'
    """
    if 'noise_direction' not in results_df.columns:
        results_df['noise_direction'] = results_df['error_model'].apply(
            deduce_noise_direction
        )
    error_model_df = results_df[[
        'code_family', 'error_model', 'decoder'
    ]].drop_duplicates()
    error_model_df['noise_direction'] = error_model_df['error_model'].apply(
        deduce_noise_direction
    )
    error_model_df = error_model_df.sort_values(by='noise_direction')

    r_xyz = pd.DataFrame(
        error_model_df['noise_direction'].tolist(),
        index=error_model_df.index,
        columns=['r_x', 'r_y', 'r_z']
    )

    error_model_df = pd.concat([
        error_model_df, r_xyz
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


def get_thresholds_df(
    results_df: pd.DataFrame,
    ftol_est: float = 1e-5,
    ftol_std: float = 1e-5,
    maxfev: int = 2000,
    logical_type: str = 'total',
    n_fail_label: str = 'n_fail',
):
    """Extract thresholds from table of results using heuristics.

    Parameters
    ----------
    results_df : pd.DataFrame
        The results for each (code, error_model, decoder).
        Should have at least the columns:
        'code', 'error_model', 'decoder', 'n', 'k', 'd',
        'n_fail', 'n_trials'.
        If the `logical_type` keyword argument is given,
        then then either 'p_0_est' and 'p_est_word' should be columns too.
    ftol_est : float
        Tolerance for the best fit.
    ftol_std : float
        Tolerance for the bootstrap fits.
    maxfev : int
        Maximum number of iterations for the curve fitting.
    logical_type : str
        Pick from 'total', 'single', or 'word',
        which will take `p_est` to be 'p_est', 'p_0_est', 'p_est_word'
        respectively.
        This is used to adjust which error rate is used as 'the' logical error
        rate for purposes of extracting thresholds with finite-size scaling.
    n_fail_label : str
        The column that is 'n_fail'.
    """

    # Initialize with unique error models and their parameters.
    thresholds_df = get_error_model_df(results_df)

    # Intialize the lists.
    p_th_sd = []
    p_th_nearest = []
    p_left = []
    p_right = []
    fss_params = []
    p_th_fss = []
    p_th_fss_left = []
    p_th_fss_right = []
    p_th_fss_se = []
    df_trunc_list = []
    params_bs_list = []

    p_est = 'p_est'
    if logical_type == 'single':
        p_est = 'p_0_est'
    elif logical_type == 'word':
        p_est = 'p_est_word'

    parameter_sets = thresholds_df[[
        'code_family', 'error_model', 'decoder'
    ]].values
    for code_family, error_model, decoder in parameter_sets:
        df_filt = results_df[
            (results_df['code_family'] == code_family)
            & (results_df['error_model'] == error_model)
            & (results_df['decoder'] == decoder)
        ]

        # Find nearest value where crossover changes.
        p_th_nearest_val = get_p_th_nearest(df_filt, p_est=p_est)
        p_th_nearest.append(p_th_nearest_val)

        # More refined crossover using standard deviation heuristic.
        p_th_sd_val, p_left_val, p_right_val = get_p_th_sd_interp(
            df_filt, p_nearest=p_th_nearest_val, p_est=p_est
        )
        p_th_sd.append(p_th_sd_val)

        # Left and right bounds to truncate.
        p_left.append(p_left_val)
        p_right.append(p_right_val)

        # Finite-size scaling fitting.
        params_opt, params_bs, df_trunc = fit_fss_params(
            df_filt, p_left_val, p_right_val, p_th_nearest_val,
            ftol_est=ftol_est, ftol_std=ftol_std, maxfev=maxfev,
            p_est=p_est, n_fail_label=n_fail_label,
        )
        fss_params.append(params_opt)

        # 1-sigma error bar bounds.
        p_th_fss_left.append(np.quantile(params_bs[:, 0], 0.16))
        p_th_fss_right.append(np.quantile(params_bs[:, 0], 0.84))

        # Standard error.
        p_th_fss_se.append(params_bs[:, 0].std())

        # Use the median as the estimator.
        p_th_fss.append(np.median(params_bs[:, 0]))

        # Trucated data.
        df_trunc_list.append(df_trunc)

        # Bootstrap parameters sample list.
        params_bs_list.append(params_bs)

    thresholds_df['p_th_sd'] = p_th_sd
    thresholds_df['p_th_nearest'] = p_th_nearest
    thresholds_df['p_left'] = p_left
    thresholds_df['p_right'] = p_right
    # thresholds_df['p_th_fss'] = np.array(fss_params)[:, 0]
    thresholds_df['p_th_fss'] = p_th_fss
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


def subthreshold_scaling(results_df, chosen_probabilities=None):
    """Do subthreshold scaling analysis.

    This was a legacy method where we tried many different fitting ansatzs.
    """
    if chosen_probabilities is None:
        chosen_probabilities = np.sort(results_df['probability'].unique())
    sts_properties = []
    for probability in chosen_probabilities:
        df_filt = results_df[
            np.isclose(results_df['probability'], probability)
        ].copy()
        df_filt['d'] = df_filt['size'].apply(lambda x: min(x))
        df_filt = df_filt[df_filt['d'] > 2]

        d_values = df_filt['d'].values
        p_est_values = df_filt['p_est'].values
        p_se_values = df_filt['p_se'].values
        log_p_est_values = np.log(p_est_values)
        log_p_se_values = p_se_values/p_est_values
        w = 1/log_p_se_values

        # Fit to linear ansatz log(p_est) = c_0 + c_1*d
        linear_coefficients = polyfit(
            d_values, log_p_est_values,
            deg=1,
            w=w
        )

        # Fit to quadratic ansatz log(p_est) = c_0 + c_2*d**2
        # fit_coefficients, _, _, _ = np.linalg.lstsq(
        #     np.vstack([w*np.ones_like(d_values), w*d_values**2]).T,
        #     log_p_est_values*w,
        #     rcond=None
        # )
        # quadratic_coefficients = np.array([
        #     fit_coefficients[0],
        #     0.0,
        #     fit_coefficients[1]
        # ])

        # Fit to ansatz log(p_est) = c_0 + c_1*d + c_2*d**2
        quadratic_coefficients = polyfit(
            d_values, log_p_est_values,
            deg=2,
            w=w
        )

        # Fit to ansatz log(p_est) = c_0 + c_3*d**3
        cubic_coefficients = polyfit(
            d_values, log_p_est_values,
            deg=3,
            w=w
        )

        # The slope of the linear fit.
        linear_fit_gradient = linear_coefficients[-1]
        sts_properties.append({
            'probability': probability,
            'd': d_values,
            'p_est': p_est_values,
            'p_se': p_se_values,
            'linear_coefficients': linear_coefficients,
            'quadratic_coefficients': quadratic_coefficients,
            'cubic_coefficients': cubic_coefficients,
            'linear_fit_gradient': linear_fit_gradient,
            'log_p_on_1_minus_p': np.log(probability/(1 - probability)),
        })

    gradient_coefficients = polyfit(
        [props['log_p_on_1_minus_p'] for props in sts_properties],
        [props['linear_fit_gradient'] for props in sts_properties],
        1
    )

    return sts_properties, gradient_coefficients


def fit_subthreshold_scaling_cubic(results_df, order=3, ansatz='poly'):
    """Get fit parameters for subthreshold scaling ansatz."""
    log_p_L = np.log(results_df['p_est'].values)
    log_p = np.log(results_df['probability'].values)
    L = results_df['size'].apply(lambda x: min(x))

    if ansatz == 'free_power':
        params_0 = [max(log_p_L), max(log_p), 1, 1]
    elif ansatz == 'simple':
        params_0 = [max(log_p_L), max(log_p), 1]
    else:
        params_0 = tuple(
            [max(log_p_L), max(log_p)] + np.ones(order + 1).tolist()
        )

    x_data = np.array([log_p, L])
    y_data = log_p_L
    maxfev: int = 2000
    ftol: float = 1e-5

    subthreshold_fit_function = get_subthreshold_fit_function(
        order=order, ansatz=ansatz
    )

    params_opt, _ = curve_fit(
        subthreshold_fit_function, x_data, y_data,
        p0=params_0, ftol=ftol, maxfev=maxfev
    )
    y_fit = subthreshold_fit_function(x_data, *params_opt)
    return y_fit, y_data, params_opt


def get_subthreshold_fit_function(order=3, ansatz='poly'):

    def free_power_sts_fit_function(x_data, *params):
        """Subthreshold scaling ansatz fit function log_p_L(log_p, L)."""
        log_p, L = x_data
        log_p_L_th, log_p_th = params[:2]
        const, power = params[2:]

        # Z-distance of code, weight of lowest-weight Z-only logical operator.
        d = const*L**power

        # The log of the logical error rate according to the ansatz.
        log_p_L = log_p_L_th + (d + 1)/2*(log_p - log_p_th)
        return log_p_L

    def simple_sts_fit_function(x_data, *params):
        """Subthreshold scaling ansatz fit function log_p_L(log_p, L)."""
        log_p, L = x_data
        log_p_L_th, log_p_th = params[:2]
        const = params[2]

        # Z-distance of code, weight of lowest-weight Z-only logical operator.
        d = const*L**order

        # The log of the logical error rate according to the ansatz.
        log_p_L = log_p_L_th + (d + 1)/2*(log_p - log_p_th)
        return log_p_L

    def sts_fit_function(x_data, *params):
        """Subthreshold scaling ansatz fit function log_p_L(log_p, L)."""
        log_p, L = x_data
        log_p_L_th, log_p_th = params[:2]
        d_coefficients = np.array(params[2:])

        # Z-distance of code, weight of lowest-weight Z-only logical operator.
        d = d_coefficients.dot([L**n for n in range(order + 1)])

        # The log of the logical error rate according to the ansatz.
        log_p_L = log_p_L_th + (d + 1)/2*(log_p - log_p_th)
        return log_p_L

    if ansatz == 'free_power':
        return free_power_sts_fit_function
    elif ansatz == 'simple':
        return simple_sts_fit_function
    else:
        return sts_fit_function


def read_entry(
    data: Union[List, Dict], results_file: Optional[str] = None
) -> List[Dict]:
    """List of entries from data in list or dict format.

    Returns an empty list if it is not a valid results dict.

    Parameters
    ----------
    data : Union[List, Dict]
        The data that is parsed raw from a results file.
    results_path : str
        The path to the results directory or .zip file.

    Returns
    -------
    entries : List[Dict]
        The list of entries, each of which corresponds to a results file.
    """
    entries = []
    if isinstance(data, list):
        for sub_data in data:
            entries += read_entry(sub_data, results_file=results_file)

    elif isinstance(data, dict):

        # If requires keys are not there, return empty list.
        if 'inputs' not in data or 'results' not in data:
            return []

        # Add the inputs.
        entry = data['inputs']
        entry['size'] = tuple(entry['size'])

        # Add the results, converting to np arrays where possible.
        entry.update(data['results'])
        for key in ['codespace', 'success']:
            if key in entry:
                entry[key] = np.array(entry[key], dtype=bool)
        entry['effective_error'] = np.array(
            entry['effective_error'], dtype=np.uint8
        )

        # Record the path of the results file if given.
        if results_file:
            entry['results_file'] = results_file

        # Count the number of samples
        entry['n_trials'] = len(entry['effective_error'])

        # Deal with legacy names for things.
        if 'n_k_d' in entry:
            n_k_d = entry.pop('n_k_d')
            entry['n'], entry['k'], entry['d'] = n_k_d
        if 'error_probability' in entry:
            entry['probability'] = entry.pop('error_probability')

        entry['bias'] = deduce_bias(entry['error_model'])
        entries.append(entry)
    return entries


def deduce_bias(
    error_model: str, rtol: float = 0.1,
    results_path: Optional[str] = None
) -> Union[str, float, int]:
    """Deduce the eta ratio from the noise model label.

    Parameters
    ----------
    noise_model : str
        The noise model.
    rtol : float
        Relative tolearnce to consider rounding eta value to int.

    Returns
    -------
    eta : Union[str, float, int]
        The eta value. If it's infinite then the string 'inf' is returned.
    """
    eta: Union[str, float, int] = 0

    # Infer the bias from the file path if possible.
    file_path_match = None
    if results_path:
        file_path_match = re.search(r'bias-([\d\.]+|inf)-', results_path)

    if file_path_match:
        if file_path_match.group(1) == 'inf':
            eta = 'inf'
        else:
            eta = float(file_path_match.group(1))
            if np.isclose(eta % 1, 0):
                eta = int(np.round(eta))
        return eta

    # Commonly occuring eta values to snap to.
    common_eta_values = [0.5, 3, 10, 30, 100, 300, 1000]
    error_model_match = re.search(
        r'Pauli X([\d\.]+)Y([\d\.]+)Z([\d\.]+)', error_model
    )
    if error_model_match:
        direction = np.array([
            float(error_model_match.group(i))
            for i in [1, 2, 3]
        ])
        r_max = np.max(direction)
        if r_max == 1:
            eta = 'inf'
        else:
            eta = r_max/(1 - r_max)
            common_matches = np.isclose(eta, common_eta_values, rtol=rtol)
            if any(common_matches):
                eta = common_eta_values[np.argwhere(common_matches).flat[0]]
            elif np.isclose(eta, np.round(eta), rtol=rtol):
                eta = int(np.round(eta))
            else:
                eta = np.round(r_max/(1 - r_max), 3)
    return eta
