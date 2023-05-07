"""
Routines for analyzing output data.
"""

import os
import warnings
from typing import List, Optional, Tuple, Union, Callable, Iterable, Any, Dict
import json
import re
import gzip
from zipfile import ZipFile
from itertools import product
from pathlib import Path
from pprint import pformat
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, fsolve
from scipy.signal import argrelextrema
from .config import SHORT_NAMES, LONG_NAMES
from .utils import (
    fmt_uncertainty, identity,
    rescale_prob, fmt_confidence_interval,
    load_json, save_json, get_label,
    quadratic
)

Numerical = Union[Iterable, float, int]


class Analysis:
    """Analysis on large collections of results files.

    This is the preferred method because it does not require reading the input
    files since the input parameters are saved to the results files anyway.
    It also does not create Simulation objects for every data point, which
    could be slow.

    Parameters
    ----------
    results : Union[str, List[str]]
        Path to directory or .zip containing .json.gz or .json results.
        Can also accept list of paths.
    overrides : Optional[Union[str, dict]]
        Path to json file that gives specifications on what to override,
        for instance when to truncate.
        Can also be a dictionary with the same contents as a json file.
        As well as replacements values for known analytical threshold values.
        See `scripts/overrides.json` for an example.
    verbose : bool
        If True, logs will be printed at various steps of the analysis.

    Attributes
    ----------
    results : pd.DataFrame
        Results for each set of (code, error model and decoder).
    thresholds : pd.DataFrame
        Thresholds for each (code family, error_model and decoder).
    trunc_results : pd.DataFrames
        Truncated results that were used for threshold calculation.
    sectors : Dict[str, Any]
        Dict containing thresholds and trunc_results DataFrames for each
        sector.
    """

    # List of paths where results are to be analyzed.
    results_paths: List[str] = []

    # Locations of individual results files.
    file_locations: List[Union[str, Tuple[str, str]]] = []

    # Raw data extracted from files.
    raw: pd.DataFrame

    # Results of each code, error_model and decoder.
    results: pd.DataFrame

    # Prints more details if true.
    verbose: bool = False

    # Set of parameters that are unique to each input.
    INPUT_KEYS: List[str] = [
        'code_str', 'error_model_str', 'error_rate', 'decoder_str',
        'method_str'
    ]

    # Set of parameters that identifies a family for the threshold vs bias
    # plot.
    FAMILY_KEYS: List[str] = [
        'decoder', 'code_family', 'error_model_family', 'bias'
    ]

    # Set of input parameter that identifies a threshold.
    THRESHOLD_KEYS: List[str] = [
        'code_family', 'error_model', 'decoder'
    ]

    # Set of parameter keys that identify a simulation,
    # which is visualized as a point with error bars in the logical vs physical
    # error rate crossing plots to extract thresholds from.
    POINT_KEYS: List[str] = [
        'code', 'error_model', 'decoder', 'error_rate'
    ]

    SECTOR_KEYS: List[str] = ['total', 'X', 'Z']

    # Quality targets that are to be met.
    targets: Dict[str, Any] = {
        'n_error_rates': 8,
        'n_trials': 10000,
    }

    # Specification of manual overrides such as truncation ranges.
    overrides_spec: Dict

    # Processed manual overrides.
    overrides: Dict[str, Dict[Tuple, Any]]

    # Manual values to be replace parsed from the overrides json file.
    replaces: Dict[Tuple, Any]

    # Parameter values to be skipped, also from the overrides json file.
    skips: List[Tuple]

    # Extra replace overrides that are not found in the results but get
    # appended to thresholds DataFrame in the end.
    extra_thresholds: List[Dict]

    # Threshold data for each sector ('total', 'X', 'Z').
    sectors: Dict[str, Any]

    def __init__(
        self, results: Union[str, List[str]] = [], verbose: bool = False,
        overrides: Optional[Union[str, dict]] = None,
        progress: Optional[Callable] = identity
    ):
        self.verbose = verbose
        self.results_paths = []
        if isinstance(results, list):
            self.results_paths += results
        else:
            self.results_paths.append(results)

        for results_path in self.results_paths:
            if not os.path.exists(results_path):
                raise ValueError(f"The path {results_path} does not exist")

        self.overrides_spec = dict()
        if overrides:
            if isinstance(overrides, str):
                with open(overrides) as f:
                    self.overrides_spec = json.load(f)
            elif isinstance(overrides, dict):
                self.overrides_spec = overrides

        # Manual specifications are intialized as empty until apply_overrides
        # is called after results are aggregated.
        self.overrides = {sector: dict() for sector in self.SECTOR_KEYS}
        self.replaces = dict()
        self.skips = []
        self.extra_thresholds = []

        self._thresholds: Optional[pd.DataFrame] = None
        self._sector_thresholds: Optional[Dict[str, pd.DataFrame]] = None
        self._min_thresholds: Optional[pd.DataFrame] = None
        self._trunc_results: Optional[Dict[str, pd.DataFrame]] = None

        self.find_files()
        self.count_files()
        self.read_files(progress=progress)
        self.aggregate()
        self.apply_overrides()
        self.calculate_total_error_rates()
        self.calculate_word_error_rates()
        self.calculate_single_qubit_error_rates()
        self.assign_labels()
        self.reorder_columns()

    @property
    def thresholds(self):
        if (self.sector_thresholds is None or
                'total' not in self.sector_thresholds):
            self.calculate_thresholds()

        return self.sector_thresholds['total']

    @property
    def sector_thresholds(self):
        if self._sector_thresholds is None:
            self.calculate_sector_thresholds()

        return self._sector_thresholds

    @property
    def min_thresholds(self):
        if self._min_thresholds is None:
            self.calculate_min_thresholds()

        return self._min_thresholds

    @property
    def trunc_results(self):
        if self._trunc_results is None:
            # TODO: replace by something more fine-grained
            self.calculate_thresholds()
            self.calculate_sector_thresholds()

        return self._trunc_results

    def get_results(self, progress: Optional[Callable] = identity):
        return self._results

    def _load_results_df(self, data: dict) -> pd.DataFrame:
        """Load dict into results DataFrame."""
        results = pd.DataFrame(data)
        if 'size' in results:
            results['size'] = results['size'].apply(tuple)
        return results

    def load(self, path):
        """Load previous analysis from .json.gz file."""
        data = load_json(path)
        self._results = self._load_results_df(data['results'])
        self.trunc_results = self._load_results_df(data['trunc_results'])
        self._thresholds = pd.DataFrame(data['thresholds'])
        for k in ['fss_params', 'params_bs']:
            self._thresholds[k] = self._thresholds[k].apply(np.array)
        self._sector_thresholds = {
            sector: pd.DataFrame(sector_data['thresholds'])
            for sector, sector_data in data['sector_thresholds'].items()
        }
        self._trunc_results = {
            sector: self._load_results_df(sector_data)
            for sector, sector_data in data['trunc_results'].items()
        }

    def save(self, path):
        """Save analysis to a .json.gz file."""
        self.log(f'Saving analysis to {path}')
        drop_columns = [
            'effective_error', 'codespace', 'success', 'results_file'
        ]
        data = {
            'trunc_results': {
                sector: self.trunc_results[sector].drop(
                    drop_columns, axis=1
                ).to_dict()
                for sector in self.trunc_results.keys()
            },
            'results': self._results.drop(drop_columns, axis=1).to_dict(),
            'thresholds': self.thresholds.to_dict(),
            'sector_thresholds': {
                sector: sector_data.to_dict()
                for sector, sector_data in self.sector_thresholds.items()
            }
        }
        save_json(data, path)

    def apply_overrides(self):
        """Read manual overrides from .json file."""

        # Do nothing is there is no given overrides json.
        if not self.overrides_spec:
            return

        # Otherwise find the parameter sets which need to be truncated and save
        # them to the overrides attribute.
        data = self.overrides_spec
        if 'overrides' in data and isinstance(data['overrides'], list):

            self.overrides = {sector: dict() for sector in self.SECTOR_KEYS}
            self.replaces = dict()
            self.skips = []
            self.extra_thresholds = []

            # Keep track of which overrides got used and not used so the user
            # does not write them in vain in case they make a mistake.
            overrides_used = np.zeros(len(data['overrides']), dtype=bool)

            for i_override, override in enumerate(data['overrides']):
                if 'filters' in override and override['filters']:
                    filter_index = None
                    for key, value in override['filters'].items():
                        extra_index = self._results[key] == value
                        if filter_index is None:
                            filter_index = extra_index
                        else:
                            filter_index = filter_index & extra_index
                    parameter_sets = self._results[filter_index][[
                        'code', 'error_model', 'decoder'
                    ]].drop_duplicates().values
                    for triple in parameter_sets:
                        if 'truncate' in override:
                            sector_key = 'total'
                            if 'sector' in override:
                                sector_key = override['sector']
                            self.overrides[sector_key][tuple(triple)] = \
                                override['truncate']
                            overrides_used[i_override] = True
                        if 'replace' in override:
                            self.replaces[tuple(triple)] = override['replace']
                            overrides_used[i_override] = True
                        if 'skip' in override and override['skip']:
                            self.skips.append(tuple(triple))
                            overrides_used[i_override] = True

                    if len(parameter_sets) == 0 and 'replace' in override:
                        entry = dict(override['filters'])
                        entry.update(
                            self.replace_threshold(override['replace'])
                        )
                        self.extra_thresholds.append(entry)
                        overrides_used[i_override] = True

            n_overrides = len(data['overrides'])
            used_overrides = sum(overrides_used)
            unused_overrides = n_overrides - used_overrides
            self.log(f'Using {used_overrides} out of {n_overrides} filters')

            # Printout the unused overrides so the user can find them.
            if unused_overrides > 0:
                self.log(f'{unused_overrides} unused overrides:')
                counter = 0
                for used, entry in zip(overrides_used, data['overrides']):
                    if not used:
                        self.log(pformat(entry, indent=2))
                    counter += 1

                    # If more than 3, truncate the logs and just say how many
                    # more there are that haven't been printed.
                    if counter > 3:
                        self.log(f'... and {unused_overrides - 3} more')
                        break

            # Apply total sector truncations to X and Z  as well if not given
            # explicitly already.
            for triple in self.overrides['total'].keys():
                for sector_key in ['X', 'Z']:
                    if triple not in self.overrides[sector_key]:
                        self.overrides[sector_key][triple] = (
                            self.overrides['total'][triple]
                        )

    def log(self, message: str):
        """Display a message, if the verbose attribute is True.

        Parameters
        ----------
        message : str
            Message to be displayed.
        """
        if self.verbose:
            print(message, flush=True)

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
            else:
                nominal_path = os.path.abspath(file_location)
                data = load_json(nominal_path)

            entries += read_entry(data, results_file=nominal_path)

        # Convert to DataFrame to conserve memory.
        self.raw = pd.DataFrame(entries)

        # Round error_rate to six digits, because anything smaller than that
        # is likely numerical error.
        self.raw['error_rate'] = self.raw['error_rate'].round(6)

    def aggregate(self):
        """Aggregate the raw data into results attribute."""
        self.log('Aggregating data')

        # Input keys by which data is to be grouped.
        grouped_df = self.raw.assign(
            code_str=self.raw['code'].astype('str'),
            decoder_str=self.raw['decoder'].astype('str'),
            error_model_str=self.raw['error_model'].astype('str'),
            method_str=self.raw['method'].astype('str'),
        ).groupby(
            self.INPUT_KEYS
        )
        # grouped_df = self.raw.groupby(self.INPUT_KEYS)

        # Columns for which grouped values are to be summed.
        added_columns = grouped_df[[
            'wall_time', 'n_trials'
        ]].sum()

        # Columns for which grouped entries are to be concantenated np arrays.
        concat_columns = grouped_df[[
            'effective_error', 'success', 'codespace'
        ]].aggregate(lambda x: np.concatenate(x.values))

        # Columns to be grouped and turned into lists.
        list_columns = grouped_df[['results_file']].aggregate(list)

        remaining_columns = grouped_df[
            ['code', 'error_model', 'decoder', 'method']
        ].first()

        # Stack the columns together side by side to form a big table.
        self._results = pd.concat([
            added_columns, concat_columns, list_columns,
            remaining_columns
        ], axis=1).reset_index()

        # Count the number of fails, later used for error bars.
        self._results['n_fail'] = (
            self._results['n_trials'] - self._results['success'].apply(sum)
        )

        self._results.drop(
            ['code_str', 'decoder_str', 'error_model_str', 'method_str'],
            axis=1, inplace=True
        )

        # Deduce the bias.
        self._results['bias'] = self._results['error_model'].apply(deduce_bias)

        # Put useful parameters as their own columns
        for col in ['n', 'k', 'd']:
            self._results[col] = self._results['code'].apply(lambda x: x[col])

        # Separate model names from parameters
        for s in ['code', 'decoder', 'error_model', 'method']:
            self._results[f'{s}_params'] = self._results[s].apply(
                lambda x: x['parameters']
            )
            self._results[s] = self._results[s].apply(lambda x: x['name'])

    def reorder_columns(self):
        """Reorder the columns of the results dataframe for a more
        relevant display when printing it"""

        ordered_columns = [
            'code', 'n', 'k', 'd', 'bias',
            'error_rate', 'n_trials', 'p_est', 'p_se', 'p_word_est',
            'p_word_se', 'single_qubit_p_est', 'single_qubit_p_se',
            'success', 'codespace', 'code_params', 'decoder',
            'decoder_params', 'error_model', 'error_model_params',
            'method', 'method_params', 'results_file', 'wall_time'
        ]

        # Remove ordered columns that don't exist in results
        for col in ordered_columns:
            if col not in self._results.columns:
                print("test", col)
                ordered_columns.remove(col)

        # Add ordered columns that do exist in results
        for col in self._results.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)

        self._results = self._results[ordered_columns]

    def calculate_total_error_rates(self):
        """Calculate the total error rate.

        And add it as a column to the results attribute of this class.
        """
        self.log('Calculating total error rates')
        estimates_list = []
        uncertainties_list = []
        for i_entry, entry in self._results.iterrows():
            estimator = 1 - entry['success'].mean()
            uncertainty = get_standard_error(estimator, entry['n_trials'])
            estimates_list.append(estimator)
            uncertainties_list.append(uncertainty)
        self._results['p_est'] = estimates_list
        self._results['p_se'] = uncertainties_list

    def calculate_word_error_rates(self):
        """Calculate the word error rate using the formula.

        The formula assumes a uniform error rate across all logical qubits.
        """
        self.log('Calculating word error rates')
        p_est = self._results['p_est']
        p_se = self._results['p_se']
        k = self._results['k']
        p_word_est, p_word_se = get_word_error_rate(p_est, p_se, k)
        self._results['p_word_est'] = p_word_est
        self._results['p_word_se'] = p_word_se

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
        for i_entry, entry in self._results.iterrows():
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
        self._results['single_qubit_p_est'] = estimates_list
        self._results['single_qubit_p_se'] = uncertainties_list

    def assign_labels(self):
        """Assign labels to each entry for filtering."""
        self._results['code_label'] = (
            self._results[['code', 'code_params']].apply(
                lambda x: get_label(*x), axis=1
            )
        )
        self._results['error_model_label'] = (
            self._results[['error_model', 'error_model_params']].apply(
                lambda x: get_label(*x), axis=1
            )
        )
        self._results['decoder_label'] = (
            self._results[['decoder', 'decoder_params']].apply(
                lambda x: get_label(*x), axis=1
            )
        )

    def calculate_thresholds(
        self,
        ftol_est: float = 1e-5,
        ftol_std: float = 1e-5,
        maxfev: int = 2000,
        p_est: str = 'p_est',
        n_trials_label: str = 'n_trials',
        n_fail_label: str = 'n_fail',
        sector: Optional[str] = None,
        autotruncate: bool = False,
    ):
        """Extract thresholds using heuristics or manual override limits.

        Records thresholds in the `sectors` attribute for the given sector.
        If sector is None, then the total logical error rate is used to
        calculate the thresholds and the threshold information is saved to the
        `sectors['total']` part.

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
            This is used to adjust which error rate is used as 'the' logical
            error rate for purposes of extracting thresholds with finite-size
            scaling.
        n_fail_label : str
            The column that is 'n_fail'.
        sector : Optional[str]
            The Pauli sector use to calculate the threshold for.
            Will use overall error if None given.
        autotruncate : bool
            Automatically truncate the p range so that only the regime where
            finite-size scaling is likely to work is used for the fitting.
            It uses a heuristic algorithm that may or may not be appropriate
            for the situation.
            By default it is set false so all data is used unless manually
            specified otherwise.
        """
        sector = 'total' if sector is None else sector

        results_df = self._results

        # Intialize the lists.
        entries = []
        df_trunc_list = []

        # List of triples that identifies each parameter set,
        param_keys = ['code', 'error_model_label', 'decoder_label']
        parameter_sets: List[Tuple] = [
            tuple(param_set)
            for param_set in self._results[
                param_keys
            ].drop_duplicates().sort_values(by=param_keys).values
        ]

        skipped_param_sets = [
            param_set for param_set in parameter_sets
            if param_set in self.skips
        ]
        if skipped_param_sets:
            self.log(f'Skipping {len(skipped_param_sets)} parameter sets')
            for skipped_param_set in skipped_param_sets:
                self.log(
                    pformat(
                        results_df[(
                            results_df[param_keys] == skipped_param_set
                        ).all(axis=1)][
                            param_keys + ['bias']
                        ].drop_duplicates().iloc[0].to_dict(),
                        indent=2
                    )
                )

        # Skip the parameter sets manually specified to be skipped.
        parameter_sets = [
            param_set for param_set in parameter_sets
            if param_set not in self.skips
        ]

        for param_set in parameter_sets:
            code_family, error_model, decoder = param_set

            entry = {
                'code': code_family,
                'error_model_label': error_model,
                'decoder_label': decoder,
            }

            df_filt = results_df[
                (results_df[param_keys] == param_set).all(axis=1)
            ]
            entry['error_model'] = df_filt['error_model'].iloc[0]
            entry['bias'] = df_filt['bias'].iloc[0]

            # Use the replacement values if there are any manually given.
            if param_set in self.replaces:
                if 'p_th_fss' in self.replaces[param_set]:

                    self.log('Using given threshold values')
                    self.log(pformat({
                        **entry,
                        **self.replaces[param_set]
                    }, indent=2))
                    entry.update(
                        self.replace_threshold(self.replaces[param_set])
                    )

            # Find nearest value where crossover changes.
            entry['p_th_nearest'] = get_p_th_nearest(df_filt, p_est=p_est)

            # Using the manual override to get truncation limits.
            if param_set in self.overrides[sector]:

                # Initialize to max limits and reuse crossover.
                entry.update({
                    'p_left': df_filt['error_rate'].min(),
                    'p_right': df_filt['error_rate'].max(),
                    'p_th_sd': entry['p_th_nearest'],
                })

                # Use the overrides to refine limits if available.
                if 'error_rate' in self.overrides[sector][param_set]:
                    tolerance = 1e-9
                    p_trunc = self.overrides[sector][param_set]['error_rate']
                    if 'min' in p_trunc and p_trunc['min'] is not None:
                        entry['p_left'] = p_trunc['min'] - tolerance
                    if 'max' in p_trunc and p_trunc['max'] is not None:
                        entry['p_right'] = p_trunc['max'] + tolerance

                # Truncate the sizes if told to do so.
                if 'd' in self.overrides[sector][param_set].keys():
                    if 'min' in self.overrides[sector][param_set]['d']:
                        df_filt = df_filt[
                            df_filt['d']
                            >= self.overrides[sector][param_set]['d']['min']
                        ]
                    if 'max' in self.overrides[sector][param_set]['d']:
                        df_filt = df_filt[
                            df_filt['d']
                            <= self.overrides[sector][param_set]['d']['max']
                        ]

                # Just use the crossover point if it's in between the limits.
                if (
                    entry['p_left'] < entry['p_th_nearest']
                    and entry['p_th_nearest'] < entry['p_right']
                ):
                    entry['p_th_sd'] = entry['p_th_nearest']

                # Otherwise take the average.
                else:
                    entry['p_th_sd'] = (entry['p_left'] + entry['p_right'])/2

            # Use auto heuristic for getting limits of p to truncate.
            elif autotruncate:
                (
                    entry['p_th_sd'], entry['p_left'], entry['p_right']
                ) = get_p_th_sd_interp(
                    df_filt, p_nearest=entry['p_th_nearest'], p_est=p_est
                )

            else:
                # Actually, override, use all data if no manual trucation.
                entry['p_th_sd'] = entry['p_th_nearest']
                entry['p_left'] = df_filt['error_rate'].min()
                entry['p_right'] = df_filt['error_rate'].max()

            if param_set not in self.replaces:

                # Finite-size scaling fitting.
                params_opt, params_bs, df_trunc = fit_fss_params(
                    df_filt, entry['p_left'], entry['p_right'],
                    entry['p_th_nearest'],
                    ftol_est=ftol_est, ftol_std=ftol_std, maxfev=maxfev,
                    p_est=p_est, n_trials_label=n_trials_label,
                    n_fail_label=n_fail_label,
                )
                df_trunc_list.append(df_trunc)

                # Use the median as the estimator.
                # and use 1-sigma error bar left and right bounds,
                # but also record the standard deviation anyway.
                entry.update({
                    'fss_params': params_opt,
                    'params_bs': params_bs,
                    'p_th_fss': np.median(params_bs[:, 0]),
                    'p_th_fss_left': np.quantile(params_bs[:, 0], 0.16),
                    'p_th_fss_right': np.quantile(params_bs[:, 0], 0.84),
                    'p_th_fss_se': params_bs[:, 0].std(),
                })

                # Record whether the fit was found or it was bad.
                fit_status = self.get_fit_status(entry)
                entry.update({
                    'fit_status': fit_status,
                    'fit_found': fit_status == 'success',
                })

            entries.append(entry)

        thresholds = pd.DataFrame(entries)

        if self.extra_thresholds:
            thresholds = pd.concat(
                [thresholds, pd.DataFrame(self.extra_thresholds)]
            )

        trunc_results = pd.concat(df_trunc_list, axis=0)

        if self._sector_thresholds is None:
            self._sector_thresholds = dict()

        if self._trunc_results is None:
            self._trunc_results = dict()

        sector_key = 'total' if sector is None else sector

        thresholds['deformation'] = thresholds['error_model_label'].apply(
            get_deformation
        )
        trunc_results['deformation'] = \
            trunc_results['error_model_label'].apply(get_deformation)

        self._sector_thresholds[sector_key] = thresholds
        self._trunc_results[sector_key] = trunc_results

    def get_fit_status(self, entry: Dict[str, Any]) -> str:
        """Status of the fit. 'success' if successful, comment if not.

        Parameters
        ----------
        entry : Dict[str, Any]

        Returns
        -------
        status : str
            'success' of the fit succeeded and is valid otherwise a reason for
            why it is not a success is given.
        """

        # Return False if any nan.
        if np.any(pd.isna(entry['fss_params'])):
            return 'Curve fitting failed.'
        keys = ['p_th_fss', 'p_th_fss_left', 'p_th_fss_right', 'p_th_fss_se']
        for key in keys:
            if pd.isna(entry[key]):
                return 'NaN threshold estimate or uncertainty.'

        # If the uncertainty is zero, something bad probably happened.
        if np.isclose(entry['p_th_fss_left'], entry['p_th_fss_right']):
            return 'Zero CI uncertainty.'
        if np.isclose(entry['p_th_fss_se'], 0):
            return 'Zero SE uncertainty.'

        # If anything is outside the interval [0, 1] then return False.
        for key in keys:
            if entry[key] < 0 or entry[key] > 1:
                return 'Invalid threshold value.'

        # If the fit parameters are implausible then return False.
        p_th, nu, A, B, C = entry['fss_params']

        # A is the logical error rate at threshold, which should be a
        # probability between 0 and 1.
        if A < 0 or A > 1:
            return 'Invalid logical error rate at threshold.'

        # If threshold is outside bounds then return False.
        # Should take more data or manually override with wider truncation.
        if entry['p_th_fss'] < entry['p_left']:
            return 'Threshold left of leftmost data point used.'
        if entry['p_th_fss'] > entry['p_right']:
            return 'Threshold right of rightmost data point used.'

        # If the fit is a flat line, it is likely that all the data was was
        # zero, or very close to zero, which means that there is either not
        # enough data or the physical error rate range has been chosen too low.
        if np.allclose([A, B, C], 0):
            return 'Zero logical error rate fit'

        # Return 'success' if everything passed.
        return 'success'

    def calculate_sector_thresholds(self):
        """Calculate thresholds of each single-qubit logical error type.

        When thresholds cannot be calculated,
        at least check whether we are above or below threshold
        by giving upper or lower bounds on the threshold.
        """
        self.log('Calculating single-qubit sector thresholds')

        # Note that in the case the final state is not in the code space,
        # it is impossible to determine what the effective error is,
        # so these cases are removed from the analysis.

        for sector in ['X', 'Z']:

            # The column names to use.
            p_est_label = f'p_est_{sector}'
            p_se_label = f'p_se_{sector}'
            n_trials_label = f'n_trials_{sector}'
            n_fail_label = f'n_fail_{sector}'

            # The number of trials is the number of entries in the binary
            # symplectic matrix in the corresponding sector,
            # that is the number of valid trials times the number of logical
            # qubits.
            self._results[n_trials_label] = self._results['k']*(
                self._results['codespace'].apply(sum)
            )

            # Count the number of fails.
            self._results[n_fail_label] = self._results[[
                'effective_error', 'codespace'
            ]].apply(lambda row: count_fails(*row, sector), axis=1)

            # Use the mean as best estimator.
            self._results[p_est_label] = self._results[n_fail_label]/(
                self._results[n_trials_label]
            )

            # Get the standard error as well.
            self._results[p_se_label] = get_standard_error(
                self._results[p_est_label], self._results[n_trials_label]
            )

            # Calculate the thresholds for this sector only.
            self.calculate_thresholds(
                p_est=p_est_label,
                n_trials_label=n_trials_label,
                n_fail_label=n_fail_label,
                sector=sector
            )

    def calculate_min_thresholds(self):
        """Use the minimum thresholds for `min_thresholds` DataFrame attribute.

        Go through all the sector thresholds in the `sectors` attribute and
        take the minimum for each parameter set.
        """

        # Stack all the sector threshold DataFrames with a 'sector' column.
        all_thresholds = []
        for sector in self.sector_thresholds:
            df = self.sector_thresholds[sector].copy()
            df['sector'] = sector
            all_thresholds.append(df)
        thresholds = pd.concat(all_thresholds, axis=0)

        # Only take the thresholds for which fit was found.
        thresholds = thresholds[thresholds['fit_found']]

        # Reset the index.
        thresholds = thresholds.reset_index(drop=True)

        # Group by input keys, taking the minimum sectors.
        self._min_thresholds = thresholds.loc[
            thresholds.groupby(self.THRESHOLD_KEYS)['p_th_fss_left']
            .idxmin().values
        ].reset_index(drop=True)

        # Stack all the truncated results together and reset index.
        used_points = list(map(tuple, pd.concat([
            self.trunc_results[sector][self.POINT_KEYS]
            for sector in self.sector_thresholds
        ], axis=0).drop_duplicates().values))

        self.trunc_results['min'] = self._results.groupby(
            self.POINT_KEYS
        ).first().loc[used_points].reset_index().sort_values(
            by=self.POINT_KEYS
        )

    def replace_threshold(self, replacement):
        """Format override replace specification for threshold df."""
        estimate = replacement['p_th_fss']
        uncertainty = 0
        if 'p_th_fss_se' in replacement:
            uncertainty = replacement['p_th_fss_se']
        return {
            "p_th_sd":  estimate,
            "p_th_nearest":  estimate,
            "p_th_fss_left":  estimate - uncertainty,
            "p_th_fss_right":  estimate + uncertainty,
            "p_th_fss":  estimate,
            'p_th_fss_se': uncertainty,
            'fit_found': True,
            'fit_status': 'success',
        }

    def get_quality_metrics(self):
        """Table of quality metrics of data used for analysis.

        Returns
        -------
        quality : pd.DataFrame
            Summary of data quality metric for each input family as index,
            in particular the minimum number of trials for any data point in
            the input family and the number of error_rate values for that
            input family that actually got used in the analysis.
        """

        # Quality of data as measured by number of trials and number
        # of unique error_rate values for this family of inputs.
        quality = pd.concat([
            self.trunc_results.groupby(self.FAMILY_KEYS)['n_trials'].min(),
            self.trunc_results.groupby(self.FAMILY_KEYS)[[
                'error_rate'
            ]].aggregate(pd.Series.nunique).rename(
                columns={'error_rates': 'n_error_rates'}
            ),
        ], axis=1)

        # Check whether the quality targets are met for each input.
        quality['pass'] = (
            (quality['n_trials'] >= self.targets['n_trials'])
            & (quality['n_error_rates'] >= self.targets['n_error_rates'])
        )
        return quality

    def make_plots(self, plot_dir: str, include_date=True):
        """Make and display the plots while saving to directory."""
        date = ''
        if include_date:
            date = pd.Timestamp.now().strftime('%Y-%m-%d') + '_'

        os.makedirs(plot_dir, exist_ok=True)
        self.log('Making collapse plots.')
        self.make_collapse_plots(
            pdf=os.path.join(plot_dir, f'{date}collapse.pdf')
        )
        self.log('Making threshold vs bias plots.')
        self.make_threshold_vs_bias_plots(
            pdf=os.path.join(plot_dir, f'{date}thresholds-vs-bias.pdf')
        )

    def plot_thresholds(
        self,
        sector=None,
        pdf=None,
        show=False,
        include_threshold_estimate=True,
        include_main_title=True,
        include_sector_title=True
    ):
        """Make all the threshold plots."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        if pdf is not None:
            pdf_writer = PdfPages(pdf)

        plt.subplots_adjust(top=0.8)

        sector_key = 'total' if sector is None else sector

        plot_parameters = self.trunc_results[sector_key].sort_values(
            by=['code', 'error_model_label', 'bias']
        )[[
            'code', 'error_model_label', 'decoder_label'
        ]].drop_duplicates().values

        for code_family, error_model, decoder in plot_parameters:
            self._plot_thresholds(
                code_family, error_model, decoder, sector,
                include_threshold_estimate, include_main_title,
                include_sector_title
            )
            if pdf is not None:
                pdf_writer.savefig(plt.gcf(), bbox_inches='tight')
            if show:
                plt.show()

        if pdf is not None:
            pdf_writer.close()

    def _plot_thresholds(
        self, code_family: str, error_model: str, decoder: str,
        sector: Optional[str] = None, include_threshold_estimate: bool = True,
        include_main_title: bool = True, include_sector_title: bool = True
    ):
        """Plot the threshold for a given parameter set for both sectors.

        Parameters
        ----------
        code_family : str
            The code family.
        error_model : str
            The error model label.
        decoder : str
            The decoder label.
        """
        import matplotlib.pyplot as plt

        sector_key = 'total' if sector is None else sector
        self.log(
            f'Plotting {code_family}, {error_model}, {decoder}, {sector_key}'
        )

        df = self._results
        df_filt_1 = df[
            (df['code'] == code_family)
            & (df['error_model_label'] == error_model)
            & (df['decoder_label'] == decoder)
        ].sort_values(by='error_rate')
        bias = df_filt_1['bias'].iloc[0]
        decoder_family = df_filt_1['decoder'].iloc[0]

        if sector_key == 'total':
            p_est_label = 'p_est'
            p_se_label = 'p_se'
        else:
            p_est_label = f'p_est_{sector}'
            p_se_label = f'p_se_{sector}'

        cmap = plt.get_cmap('tab10')
        code_labels = df_filt_1.sort_values(by='d')['code_label'].unique()
        print(code_labels)
        for i, code_label in enumerate(code_labels):
            df_filt = df_filt_1[
                df_filt_1['code_label'] == code_label
            ].sort_values(by='error_rate')
            trunc_results = self.trunc_results[sector_key]
            df_filt_trunc = trunc_results[
                (trunc_results['code_label'] == code_label)
                & (trunc_results['code'] == code_family)
                & (trunc_results['error_model_label'] == error_model)
                & (trunc_results['decoder_label'] == decoder)
            ].sort_values(by='error_rate')

            # Draw gray circle underneath plots which are actually used for
            # the finite-size scaling.
            plt.plot(
                df_filt_trunc['error_rate'],
                df_filt_trunc[p_est_label], 'o',
                color=cmap(i)
            )
            L = df_filt.iloc[0]['code_params']['L_x']
            plt.errorbar(
                df_filt['error_rate'], df_filt[p_est_label],
                yerr=df_filt[p_se_label],
                capsize=5,
                label=f'$L={L}$',
                color=cmap(i)
            )

        # Find the threshold estimate and uncertainty.
        fit_title = ''
        if include_threshold_estimate:
            thresh = self.sector_thresholds[sector_key]
            thresh_match = thresh[
                (thresh['code'] == code_family)
                & (thresh['error_model_label'] == error_model)
                & (thresh['decoder_label'] == decoder)
            ]
            if thresh_match.shape[0] == 0:
                return
            thresh_data = thresh_match.iloc[0]

            fit_title = '' if thresh_data['fit_found'] else ' (fit failed)'

            # Draw the threshold and uncertainty bounds.
            plt.axvline(thresh_data['p_th_fss'], color='red', linestyle='--')
            plt.axvspan(
                thresh_data['p_th_fss_left'], thresh_data['p_th_fss_right'],
                alpha=0.5, color='pink'
            )
            plt.legend(title=r'$p_{\mathrm{th}}=(%.2f\pm %.2f)\%%$' % (
                100*thresh_data['p_th_fss'], 100*thresh_data['p_th_fss_se'],
            ))
        else:
            plt.legend()

        plt.yscale('linear')
        plt.xlabel('Physical error rate $p$', fontsize=16)
        plt.ylabel('Logical error rate $p_L$', fontsize=16)

        if include_main_title:
            deformation = df_filt_1['error_model'].iloc[0]
            bias_label = str(bias).replace('inf', '\\infty')
            plt.suptitle(
                f'{shorten(deformation)} {lengthen(code_family, caps=False)}\n'
                f'$\\eta_Z={bias_label}$, {shorten(decoder_family)}'
                f'{fit_title}\n',
                fontsize=16
            )

        if include_sector_title:
            sector_name = ('All errors' if sector is None
                           else f'${sector}$ errors')
            plt.title(f'{sector_name}')

        if fit_title != '':
            plt.text(
                0.5, 0.5, 'INVALID', transform=plt.gca().transAxes,
                fontsize=40, color='gray', alpha=0.5, ha='center',
                va='center', rotation=30
            )

    def make_threshold_vs_bias_plots(self, pdf=None):
        """Make and save threshold vs bias plots."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        if pdf is not None:
            pdf_writer = PdfPages(pdf)
        for code in self.thresholds['code'].unique():
            self.plot_threshold_vs_bias(code)
            if pdf is not None:
                pdf_writer.savefig(plt.gcf(), bbox_inches='tight')
            plt.show()
        if pdf is not None:
            pdf_writer.close()

    def make_collapse_plots(self, pdf=None, sector=None):
        """Make all the collapse plots."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        if pdf is not None:
            pdf_writer = PdfPages(pdf)

        plot_parameters = self.trunc_results['total'].sort_values(
            by=['code', 'error_model_label', 'bias']
        )[[
            'code', 'error_model_label', 'decoder_label'
        ]].drop_duplicates().values

        if sector is None:
            sector_list = [None, 'X', 'Z']
        elif sector == 'total':
            sector_list = [None]

        for code_family, error_model, decoder in plot_parameters:
            for sector in sector_list:
                self.plot_sector_collapse(
                    code_family, error_model, decoder, sector
                )
                if pdf is not None:
                    pdf_writer.savefig(plt.gcf(), bbox_inches='tight')
                plt.show()

        if pdf is not None:
            pdf_writer.close()

    def plot_threshold_vs_bias(
        self, code: str, inf_bias_replacement: float = 1e3,
        hashing: bool = True,
    ):
        """Plot the threshold vs bias plots.

        Parameters
        ----------
        code : str
            The code family to plot threshold vs bias for.
        inf_bias_replacement : float
            The fintite value of bias at which to plot the infinite bias points
            at.
        hashing : bool
            Will also plot hashing bound if True.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.figure(figsize=(10, 4))
        line_params = self.thresholds[
            self.thresholds['code'] == code
        ][
            ['decoder_label', 'error_model', 'deformation']
        ].drop_duplicates().sort_values(
            by=['decoder_label', 'error_model'], ascending=False
        ).values
        for decoder, error_model, deformation in line_params:
            thresh_df_filt = self.thresholds[(
                self.thresholds[[
                    'code', 'error_model', 'decoder_label', 'deformation'
                ]]
                == (code, error_model, decoder, deformation)
            ).all(axis=1)].copy()
            thresh_df_filt['bias_plot'] = thresh_df_filt['bias'].replace({
                'inf': inf_bias_replacement
            }).astype(float)
            thresh_df_filt = thresh_df_filt[[
                'bias', 'bias_plot', 'p_th_fss',
                'p_th_fss_left', 'p_th_fss_right'
            ]]
            thresh_df_filt = thresh_df_filt.sort_values(
                by='bias_plot'
            ).reset_index(drop=True)
            plt.errorbar(
                thresh_df_filt['bias_plot'],
                thresh_df_filt['p_th_fss'],
                yerr=[
                    thresh_df_filt['p_th_fss']
                    - thresh_df_filt['p_th_fss_left'],
                    thresh_df_filt['p_th_fss_right']
                    - thresh_df_filt['p_th_fss']
                ],
                fmt='o',
                capsize=5,
                label=f'{shorten(error_model)}, {shorten(decoder)}'
            )
            plt.fill_between(
                thresh_df_filt['bias_plot'],
                thresh_df_filt['p_th_fss_left'],
                thresh_df_filt['p_th_fss_right'],
                alpha=0.3
            )

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
            hb_interp = np.array([
                get_hashing_bound(point)
                for point in interp_points
            ])
            eta_interp = np.append(eta_interp, [inf_bias_replacement])
            hb_interp = np.append(hb_interp, [get_hashing_bound((0, 0, 1))])

            plt.plot(eta_interp, hb_interp, '-.', color='black',
                     label='Hashing bound', alpha=0.5, linewidth=2)

        plt.legend()
        plt.title(lengthen(code, caps=True), fontsize=16)
        plt.xscale('log')
        plt.xlabel('Bias ratio $\\eta_Z$', fontsize=16)
        plt.ylabel('Threshold error rate $p_{\\rm{th}}$', fontsize=16)
        plt.ylim(0, 0.5)
        plt.xticks(
            ticks=[0.5, 1e0, 1e1, 1e2, 1e3],
            labels=['0.5', '1', '10', '100', r'$\infty$']
        )
        draw_tick_symbol(plt, Line2D, log=True)

    def plot_sector_collapse(
        self, code_family: str, error_model: str, decoder: str,
        sector: Optional[str] = None
    ):
        """Plot the data collapse for a given parameter set for both sectors.

        Parameters
        ----------
        code_family : str
            The code family.
        error_model : str
            The error model label.
        decoder : str
            The decoder label.
        """
        import matplotlib.pyplot as plt

        sector_key = 'total' if sector is None else sector
        self.log(
            f'Plotting {code_family}, {error_model}, {decoder}, {sector_key}'
        )

        df = self._results
        df_filt_1 = df[
            (df['code'] == code_family)
            & (df['error_model_label'] == error_model)
            & (df['decoder_label'] == decoder)
        ].sort_values(by='error_rate')
        bias = df_filt_1['bias'].iloc[0]
        decoder_family = df_filt_1['decoder'].iloc[0]

        if sector_key == 'total':
            p_est_label = 'p_est'
            p_se_label = 'p_se'
        else:
            p_est_label = f'p_est_{sector}'
            p_se_label = f'p_se_{sector}'

        fig, ax = plt.subplots(ncols=2, figsize=(15, 5))

        plt.subplots_adjust(top=0.8)

        plt.sca(ax[0])
        for code_label in np.sort(df_filt_1['code_label'].unique()):
            df_filt = df_filt_1[
                df_filt_1['code_label'] == code_label
            ].sort_values(by='error_rate')
            trunc_results = self.trunc_results[sector_key]
            df_filt_trunc = trunc_results[
                (trunc_results['code_label'] == code_label)
                & (trunc_results['code'] == code_family)
                & (trunc_results['error_model_label'] == error_model)
                & (trunc_results['decoder_label'] == decoder)
            ].sort_values(by='error_rate')

            # Draw gray circle underneath plots which are actually used for
            # the finite-size scaling.
            plt.plot(
                df_filt_trunc['error_rate'],
                df_filt_trunc[p_est_label], 'o', color='gray'
            )
            L = df_filt.iloc[0]['code_params']['L_x']
            plt.errorbar(
                df_filt['error_rate'], df_filt[p_est_label],
                yerr=df_filt[p_se_label],
                capsize=5,
                label=f'$L={L}$'
            )

        # Find the threshold estimate and uncertainty.
        thresh = self.sector_thresholds[sector_key]
        thresh_match = thresh[
            (thresh['code'] == code_family)
            & (thresh['error_model_label'] == error_model)
            & (thresh['decoder_label'] == decoder)
        ]
        if thresh_match.shape[0] == 0:
            return
        thresh_data = thresh_match.iloc[0]

        fit_title = '' if thresh_data['fit_found'] else ' (fit failed)'

        # Draw the threshold and uncertainty bounds.
        plt.axvline(thresh_data['p_th_fss'], color='red', linestyle='--')
        plt.axvspan(
            thresh_data['p_th_fss_left'], thresh_data['p_th_fss_right'],
            alpha=0.5, color='pink'
        )
        plt.legend(title=r'$p_{\mathrm{th}}=(%.2f\pm %.2f)\%%$' % (
            100*thresh_data['p_th_fss'], 100*thresh_data['p_th_fss_se'],
        ))
        plt.yscale('linear')
        plt.xlabel('Physical error rate $p$', fontsize=16)
        plt.ylabel('Logical error rate $p_L$', fontsize=16)
        deformation = get_deformation(df_filt_1['error_model_label'].iloc[0])
        bias_label = str(bias).replace('inf', '\\infty')
        sector_name = 'All errors' if sector is None else f'${sector}$ errors'
        plt.suptitle(
            f'{shorten(deformation)} {lengthen(code_family, caps=False)}\n'
            f'$\\eta_Z={bias_label}$, {shorten(decoder_family)} '
            f'{fit_title}',
            fontsize=16
        )
        plt.title(f'{sector_name}')
        plt.sca(ax[1])

        df_trunc = trunc_results[
            (trunc_results['code'] == code_family)
            & (trunc_results['error_model_label'] == error_model)
            & (trunc_results['decoder_label'] == decoder)
        ]
        params_opt = thresh_data['fss_params']
        params_bs = thresh_data['params_bs']
        plot_data_collapse(
            plt, df_trunc, params_opt, params_bs, title='',
            y_label='',
            x_label='Rescaled physical error rate',
            p_est_label=p_est_label
        )

        if not thresh_data['fit_found']:
            for i in range(2):
                plt.sca(ax[i])
                plt.text(
                    0.5, 0.5, 'INVALID', transform=ax[i].transAxes,
                    fontsize=40, color='gray', alpha=0.5, ha='center',
                    va='center', rotation=30
                )

    def _plot_collapse(
        self, code_family: str, error_model: str, decoder: str
    ):
        """Plot the data collapse for a given parameter set.

        Parameters
        ----------
        code_family : str
            The code family.
        error_model : str
            The error model label.
        decoder : str
            The decoder label.
        """
        import matplotlib.pyplot as plt

        df = self._results
        df_filt_1 = df[
            (df['code_family'] == code_family)
            & (df['error_model'] == error_model)
            & (df['decoder'] == decoder)
        ].sort_values(by='error_rate')
        bias = df_filt_1['bias'].iloc[0]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        plt.sca(ax[0])
        for size in np.sort(df_filt_1['size'].unique()):
            df_filt = df_filt_1[
                df_filt_1['size'] == size
            ].sort_values(by='error_rate')
            df_filt_trunc = self.trunc_results[
                (self.trunc_results['size'] == size)
                & (self.trunc_results['code_family'] == code_family)
                & (self.trunc_results['error_model'] == error_model)
                & (self.trunc_results['decoder'] == decoder)
            ].sort_values(by='error_rate')

            # Draw gray circle underneath plots which are actually used for the
            # finite-size scaling.
            plt.plot(
                df_filt_trunc['error_rate'],
                df_filt_trunc['p_est'], 'o', color='gray'
            )
            plt.errorbar(
                df_filt['error_rate'], df_filt['p_est'],
                yerr=df_filt['p_se'],
                capsize=5,
                label=f'$L={size[0]}$'
            )

        # Find the threshold estimate and uncertainty.
        thresh = self.thresholds
        thresh_data = thresh[
            (thresh['code_family'] == code_family)
            & (thresh['error_model'] == error_model)
            & (thresh['decoder'] == decoder)
        ].iloc[0]

        # Draw the threshold and uncertainty bounds.
        plt.axvline(thresh_data['p_th_fss'], color='red', linestyle='--')
        plt.axvspan(
            thresh_data['p_th_fss_left'], thresh_data['p_th_fss_right'],
            alpha=0.5, color='pink'
        )
        plt.legend(title=r'$p_{\mathrm{th}}=(%.2f\pm %.2f)\%%$' % (
            100*thresh_data['p_th_fss'], 100*thresh_data['p_th_fss_se'],
        ))
        plt.yscale('linear')
        plt.xlabel('Physical error rate $p$', fontsize=16)
        plt.ylabel('Logical error rate $p_L$', fontsize=16)
        deformation = df_filt_1['error_model_family'].iloc[0]
        bias_label = str(bias).replace('inf', '\\infty')
        plt.suptitle(
            f'{code_family}, {shorten(deformation)} '
            f'$\\eta={bias_label}$, {shorten(decoder)}',
            fontsize=16
        )
        plt.sca(ax[1])

        df_trunc = self.trunc_results[
            (self.trunc_results['code_family'] == code_family)
            & (self.trunc_results['error_model'] == error_model)
            & (self.trunc_results['decoder'] == decoder)
        ]
        params_opt = thresh_data['fss_params']
        params_bs = thresh_data['params_bs']
        plot_data_collapse(
            plt, df_trunc, params_opt, params_bs, title='',
            y_label='',
            x_label='Rescaled physical error rate'
        )

    def export_latex_tables(self):
        """Print out LaTeX tables of thresholds."""
        error_model_family_rename = {
            'Deformed XZZX Pauli': 'Deformed',
            'Deformed Rhombic Pauli': 'Deformed',
            'Pauli': 'CSS',
        }
        for code_family in self.thresholds['code_family'].unique():
            df = self.thresholds.copy()
            df['threshold'] = df[[
                'p_th_fss', 'p_th_fss_left', 'p_th_fss_right'
            ]].apply(lambda x: fmt_confidence_interval(
                    x[0]*100, x[1]*100, x[2]*100, unit=r'\%'
            ), axis=1)
            df = df[df['code_family'] == code_family]
            df2 = df.groupby([
                'error_model_family', 'decoder', 'bias'
            ])[['threshold']].first().reset_index()

            df2['error_model_family'] = df2['error_model_family'].map(
                error_model_family_rename
            )
            df2['bias_value'] = df2['bias'].replace({'inf': np.inf})
            df2['decoder'] = df2['decoder'].apply(shorten)
            df2['bias'] = df2['bias'].replace({'inf': r'$\infty$'})
            index = pd.Index(
                df2.sort_values(by='bias_value')['bias']
                .replace({'inf': r'$\infty$'}).unique(),
                name=r'Bias $\eta_Z$'
            )
            columns = df2.groupby([
                'error_model_family', 'decoder'
            ]).first().index
            columns.names = ['', '']

            data = []
            for bias in index:
                row = []
                for error_model_family, decoder in columns:
                    values = df2[
                        (df2['bias'] == bias)
                        & (df2['error_model_family'] == error_model_family)
                        & (df2['decoder'] == decoder)
                    ]
                    if values.shape[0] > 0:
                        row.append(values['threshold'].iloc[0])
                    else:
                        row.append(np.nan)
                data.append(row)

            df3 = pd.DataFrame(data, columns=columns, index=index)
            df3 = df3.fillna('-')

            self.log(df3.style.to_latex(
                caption='Caption', position='H', multicol_align='c',
                label='tab:' + code_family.lower(), position_float='centering',
                hrules=True, column_format='c'*(df3.shape[1] + 1)
            ))


def count_fails(
    effective_error: np.ndarray, codespace: np.ndarray, sector: str
) -> int:
    """Count the number of sector fails given effective errors as BSFs.

    Parmeters
    ---------
    effective_error : np.ndarray
        Size (2*k, n_trials) array of ints where each row is a binary
        symplectic form representing the effective logical error on the
        codespace of one trial.
    codespace : np.ndarray
        Size (n_trials,) array of booleans for each trial, where True denotes
        the the decoding successfully returned the state to the code space for
        that trial, while False denotes final state was not in the code space.
    sector: str
        The sector whose errors are to be counted, either 'X' or 'Z'.
    """
    n_fails: int = 0

    # The number of logical qubits.
    k = int(effective_error.shape[1]/2)

    # Filter out the trials where decoding failed.
    codespace_effective_errors = effective_error[codespace, :]

    # Get the block corresponding to the sector.
    if sector == 'X':
        block = codespace_effective_errors[:, :k]
    else:
        block = codespace_effective_errors[:, k:]

    # Count the number of fails in the block.
    n_fails = block.sum()

    return n_fails


def shorten(long_name):
    if long_name in SHORT_NAMES:
        return SHORT_NAMES[long_name]
    pattern = r'\(.*\)'
    if re.search(pattern, long_name):
        return re.sub(pattern, '', long_name)
    return long_name


def lengthen(name, caps=True):
    long_name = name
    if name in LONG_NAMES:
        long_name = LONG_NAMES[name]
    if caps:
        if len(name) > 1:
            long_name = long_name[0].upper() + long_name[1:]
        elif len(name) == 1:
            long_name = long_name[0].upper()
    return long_name


def infer_error_model_family(label: str) -> str:
    """Infer the error_model family from the error_model label.

    Parameters
    ----------
    label : str
        The error_model label.

    Returns
    -------
    family : str
        The error_model family.
        If the family cannot be inferred,
        the original code label is returned.

    Examples
    --------
    >>> infer_error_model_family('Deformed XZZX Pauli X0.0161Y0.0161Z0.9677')
    'Deformed XZZX Pauli'
    >>> infer_error_model_family('Pauli X0.0000Y0.0000Z1.0000')
    'Pauli'
    """
    family = label
    direction_pattern = r'X[\d.]+Y[\d.]+Z[\d.]+'
    family = re.sub(direction_pattern, '', family).strip()
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
        Results with columns: 'error_rate', 'code', `p_est`.
        The 'error_rate' column is the physical error rate p.
        The 'code' column is the code label.
        The `p_est` column is the logical error rate.
    p_nearest : Optional[float]
        A hint for the nearest 'error_rate' value that is close to the
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
    p_min = df_filt['error_rate'].min()
    p_max = df_filt['error_rate'].max()
    if p_nearest is not None:
        if p_nearest > p_min:
            p_max = min(p_max, p_nearest*2)
    p_interp = np.arange(p_min, p_max + interp_res, interp_res)

    # Initialize to extents by default.
    p_left = p_min
    p_right = p_max

    # Interpolate lines.
    curves = {}
    for code in df_filt['code_label'].unique():
        df_filt_code = df_filt[df_filt['code_label'] == code].sort_values(
            by='error_rate'
        )
        if df_filt_code.shape[0] > 1:
            interpolator = interp1d(
                df_filt_code['error_rate'], df_filt_code[p_est],
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
        Results with columns: 'error_rate', 'code', `p_est`.
        The 'error_rate' column is the physical error rate p.
        The 'code' column is the code label.
        The `p_est` column is the logical error rate.
    p_est : str
        The column in the `df_filt` DataFrame that is to be used as the logical
        error rate to estimate the threshold.

    Returns
    -------
    p_th_nearest : float
        The value in the `error_rate` that is apparently the closes to the
        threshold.
    """
    code_df = get_code_df(df_filt)

    # Estimate the threshold by where the order of the lines change.
    p_est_df = pd.DataFrame({
        code: dict(df_filt[df_filt['code'] == code][[
            'error_rate', p_est
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

    if params_0 is not None and not (
        bounds[0] <= params_0[0] and params_0[0] <= bounds[1]
    ):
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
    p_nearest: Optional[float] = None,
    n_bs: int = 100,
    ftol_est: float = 1e-5,
    ftol_std: float = 1e-5,
    maxfev: int = 2000,
    p_est: str = 'p_est',
    n_trials_label: str = 'n_trials',
    n_fail_label: str = 'n_fail',
    resample_points: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Get optimized parameters and data table tweaked with heuristics.

    Parameters
    ----------
    df_filt : pd.DataFrame
        Results with columns: 'error_rate', 'code', `p_est`, `n_trials_label`,
        `n_fail_label`.
        The 'error_rate' column is the physical error rate p.
        The 'code' column is the code label.
        The `p_est` column is the logical error rate.
    p_left_val : float
        The left left value of 'error_rate' to truncate.
    p_right_val : float
        The left right value of 'error_rate' to truncate.
    p_nearest : float
        The nearest value of 'error_rate' to what was previously roughly
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
    n_trials_label : str
        Column label to use for the number of trials.
    n_fail_label : str
        Column label to use for the number of logical fails.

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

    # Truncate error rate between values.
    df_trunc = df_filt[
        (p_left_val <= df_filt['error_rate'])
        & (df_filt['error_rate'] <= p_right_val)
    ].copy()
    df_trunc = df_trunc.dropna(subset=[p_est])

    d_list = df_trunc['d'].values
    p_list = df_trunc['error_rate'].values
    f_list = df_trunc[p_est].values

    # Initial parameters to optimize.
    f_0 = df_trunc[df_trunc['error_rate'] == p_nearest][p_est].mean()
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

        # Sample from Beta distribution over error bar for each data point.
        f_list_bs = []
        for i in range(df_trunc.shape[0]):
            n_trials = int(df_trunc[n_trials_label].iloc[i])
            n_fail = int(df_trunc[n_fail_label].iloc[i])

            # Posterior distribution starting from uniform prior.
            f_list_bs.append(
                rng.beta(n_fail + 1, n_trials - n_fail + 1)
            )
        f_bs = np.array(f_list_bs)

        # Resample over set of all data points if told to do so.
        if resample_points:
            resample_index = np.sort(rng.choice(
                np.arange(len(p_list), dtype=int), size=len(p_list)
            ))

        # Otherwise don't do it.
        else:
            resample_index = np.arange(len(p_list), dtype=int)

        try:
            params_bs_list.append(
                get_fit_params(
                    p_list[resample_index],
                    d_list[resample_index],
                    f_bs[resample_index],
                    params_0=params_opt, ftol=ftol_std,
                    maxfev=maxfev
                )
            )
        except (RuntimeError, TypeError):
            params_bs_list.append(np.array([np.nan]*5))
    params_bs = np.array(params_bs_list)

    # If less than 50% of rows has nan, then remove the NaN rows.
    if pd.isna(params_bs).any(axis=1).mean() < 0.5:
        params_bs = params_bs[~np.isnan(params_bs).any(axis=1)]

    # mean_params_bs = params_bs.mean(axis=0).round(2)
    # rel_diff = np.abs(params_opt - mean_params_bs)/np.abs(params_opt)
    # assert np.all(rel_diff < 0.5)
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
    for bias_direction, deformed in product(
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
        chosen_probabilities = np.sort(results_df['error_rate'].unique())
    sts_properties = []
    for error_rate in chosen_probabilities:
        df_filt = results_df[
            np.isclose(results_df['error_rate'], error_rate)
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
            'error_rate': error_rate,
            'd': d_values,
            'p_est': p_est_values,
            'p_se': p_se_values,
            'linear_coefficients': linear_coefficients,
            'quadratic_coefficients': quadratic_coefficients,
            'cubic_coefficients': cubic_coefficients,
            'linear_fit_gradient': linear_fit_gradient,
            'log_p_on_1_minus_p': np.log(error_rate/(1 - error_rate)),
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
    log_p = np.log(results_df['error_rate'].values)
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
        if 'inputs' not in data:
            raise ValueError('Inputs missing in data')
        if 'results' not in data:
            raise ValueError('Results missing in data')

        # Add the inputs.
        entry = data['inputs']

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

        entries.append(entry)
    return entries


def deduce_bias(
    error_model: dict, rtol: float = 0.1
) -> Union[str, float, int]:
    """Deduce the eta ratio from the noise model label.

    Parameters
    ----------
    noise_model : str
        The noise model.
    rtol : float
        Relative tolerance to consider rounding eta value to int.

    Returns
    -------
    eta : Union[str, float, int]
        The eta value. If it's infinite then the string 'inf' is returned.
    """
    eta: Union[str, float, int] = 0

    # Commonly occuring eta values to snap to.
    common_eta_values = [0.5, 3, 10, 30, 100, 300, 1000]

    direction = (
        error_model['parameters']['r_x'],
        error_model['parameters']['r_y'],
        error_model['parameters']['r_z'],
    )
    r_max = np.max(direction)
    if r_max == 1:
        eta = 'inf'
    else:
        eta_f: float = r_max/(1 - r_max)
        common_matches = np.isclose(eta_f, common_eta_values, rtol=rtol)
        if any(common_matches):
            eta_f = common_eta_values[
                int(np.argwhere(common_matches).flat[0])
            ]
        elif np.isclose(eta_f, np.round(eta_f), rtol=rtol):
            eta_f = int(np.round(eta_f))
        else:
            eta_f = np.round(eta_f, 3)
        eta = eta_f

    return eta


def plot_data_collapse(
    plt, df_trunc, params_opt, params_bs, title=None,
    x_label=None, y_label=None,
    p_est_label='p_est',
):
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
            df_trunc_filt['rescaled_p'], df_trunc_filt[p_est_label],
            yerr=df_trunc_filt['p_se'], fmt='o', capsize=5,
            label=r'$L={}$'.format(d_val)
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
    if x_label is None:
        x_label = (
            r'Rescaled error probability $(p - p_{\mathrm{th}})d^{1/\nu}$'
        )
    if y_label is None:
        y_label = r'Logical failure rate $p_{\mathrm{fail}}$'
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    error_model = df_trunc['error_model'].iloc[0]
    if title is None:
        title = get_error_model_format(error_model)
    plt.title(title)
    plt.legend()


def get_error_model_format(error_model: str, eta=None) -> str:
    if 'deformed' in error_model.lower():
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


def draw_tick_symbol(
    plt, Line2D,
    log=False, axis='x',
    tick_height=0.03, tick_width=0.1, tick_location=2.5,
    axis_offset=0,
):
    """Draw a section cut tick symbol on the x axis."""

    # The actual line.
    x_points = np.array([
        -0.5,
        -0.25,
        0.25,
        0.5,
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


def get_hashing_bound(point):
    r_x, r_y, r_z = point

    def max_rate(p):
        p_array = np.array([1 - p, p*r_x, p*r_y, p*r_z])
        h_array = np.zeros(4)
        for i in range(4):
            if p_array[i] != 0:
                h_array[i] = -p_array[i]*np.log2(p_array[i])
        entropy = h_array.sum()
        return 1 - entropy

    solutions = fsolve(max_rate, 0)
    return solutions[0]


def get_deformation(error_model_label):
    """Get the deformation given the error model label.

    Parameters
    ----------
    error_model_label : str
        The label specifying how the error model object is intialized.

    Returns
    -------
    deformation : str
        The name of the deformation.

    Examples
    --------
    >>> error_model_label = "PauliErrorModel()"
    >>> get_deformation(error_model_label)
    'undeformed'

    >>> error_model_label = "PauliErrorModel(r_x=0.00495, r_y=0.00495, " \
            "r_z=0.990099, deformation_name=None, deformation_kwargs={})"
    >>> get_deformation(error_model_label)
    'undeformed'

    >>> error_model_label = "PauliErrorModel(r_x=0.333333, r_y=0.333333, " \
            "r_z=0.333333, deformation_name='XZZX', deformation_kwargs={})"
    >>> get_deformation(error_model_label)
    'XZZX'

    >>> error_model_label = "PauliErrorModel(r_x=0, r_y=1, r_z=0, " \
            "deformation_name='XY')"
    >>> get_deformation(error_model_label)
    'XY'
    """
    deformation = 'undeformed'
    matches = re.search(r"deformation_name='([\w\d]+)'", error_model_label)
    if matches:
        deformation = matches.group(1)
    return deformation
