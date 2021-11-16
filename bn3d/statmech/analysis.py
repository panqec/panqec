"""
Classes for analysis of data produced by MCMC.

:Author:
    Eric Huang
"""
import os
import json
from typing import List, Callable
import numpy as np
import pandas as pd
from .controllers import DataManager


def heat_capacity(energy, energy_2, temperature):
    r"""Heat capacity.

    Using
    $C = \frac{\langle E^2\rangle - \langle E\rangle^2}{k_B T^2}$
    """
    return (energy_2 - energy**2)/temperature**2


def count_spins(row):
    if row['spin_model'] == 'LoopModel2D':
        return str(2*row['L_x']*row['L_y'])
    else:
        return row['L_x']*row['L_y']


class SimpleAnalysis:
    """Wrapper for a simple analysis."""

    data_dir: str
    data_manager: DataManager
    observable_names: List[str]
    independent_variables: List[str]
    results_df: pd.DataFrame
    inputs_df: pd.DataFrame
    estimates: pd.DataFrame
    run_time_constants: dict

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_manager = DataManager(data_dir)

    def analyse(self):
        """Perform the analysis."""
        print('Combining inputs')
        self.combine_inputs()
        print('Combining results')
        self.combine_results()
        print('Estimating observables')
        self.estimate_observables()
        print('Estimating heat capacity')
        self.estimate_heat_capacity()
        print('Estimating correlation length')
        self.estimate_correlation_length()
        print('Estimating Binder cumulant')
        self.estimate_binder()
        print('Calculating runtime stats')
        self.calculate_run_time_stats()

    def combine_results(self):
        """Combine all raw results in as single DataFrame."""
        summary = self.data_manager.load('results')
        entries = []
        for result in summary:
            entry = {
                k: v for k, v in result.items()
                if k in ['hash', 'seed', 'tau']
            }
            for name, values in result['observables'].items():
                if isinstance(values['total'], list):
                    values['total'] = np.array(values['total'])
                    values['total_2'] = np.array(values['total_2'])
                    values['total_4'] = np.array(values['total_4'])

                entry[name] = values['total'] / values['count']
                entry[name + '_2'] = values['total_2'] / values['count']
                entry[name + '_4'] = values['total_4'] / values['count']
            entry.update(result['sweep_stats'])
            entries.append(entry)
        results_df = pd.DataFrame(entries)
        self.results_df = results_df

        self.observable_names = list(summary[0]['observables'].keys())

    def combine_inputs(self):
        """Combine all input files in a single DataFrame."""
        inputs_df = pd.DataFrame(self.data_manager.load('inputs'))
        inputs_df = inputs_df.drop('disorder', axis=1)
        for k in ['disorder_model_params', 'spin_model_params']:
            inputs_df = pd.concat([
                inputs_df[k].apply(pd.Series),
                inputs_df.drop(k, axis=1)
            ], axis=1)

        self.independent_variables = list(
            inputs_df.columns.drop(['hash'])
        ) + ['tau']

        self.inputs_df = inputs_df

    def estimate_observables(self):
        """Calculate estimators and uncertainties for observables."""
        df = self.inputs_df.merge(self.results_df)

        # Uncertainties and estimates for each observable.
        estimates = pd.DataFrame()
        labels = []
        for name in self.observable_names:
            labels += [name, f'{name}_2', f'{name}_4']

        for label in labels:
            if label in df.columns:
                estimates = pd.concat([
                    estimates,
                    pd.DataFrame({
                        f'{label}_estimate': df.groupby(
                            self.independent_variables
                        )[label].apply(np.mean),
                        f'{label}_uncertainty': df.groupby(
                            self.independent_variables
                        )[label].apply(
                            lambda x: np.std(np.vstack(x.to_numpy()), axis=0)
                        ) / np.sqrt(
                            df.groupby(
                                self.independent_variables
                            )[label].count()
                        ),
                    })
                ], axis=1)
        estimates = pd.concat([
            estimates, pd.DataFrame({
                'n_disorder': df.groupby(
                    self.independent_variables
                )[self.observable_names[0]].count(),
            })
        ], axis=1)
        self.estimates = estimates.reset_index()

        self.estimates['n_spins'] = self.estimates.apply(count_spins, axis=1)

    def bootstrap_uncertainty(
        self, function: Callable, columns: List[str], n_resamp: int = 10
    ):
        """Calculate uncertainty by bootstrap resampling."""
        estimates = self.estimates

        # Calculate uncertainty by bootstrapping.
        uncertainty = np.zeros(estimates.shape[0])

        # Generator for bootstrapping.
        bs_rng = np.random.default_rng(0)

        for i_row, row in self.estimates.iterrows():
            parameters = row[self.independent_variables].drop('tau')

            # Hashes for disorder configurations matching row parameters.
            hashes = self.inputs_df[
                (self.inputs_df[parameters.index] == parameters).all(axis=1)
            ]['hash'].values

            # Raw results filtered by matching disorders and tau.
            filtered_results = self.results_df[
                self.results_df['hash'].isin(hashes)
                & (self.results_df['tau'] == row['tau'])
            ].set_index('hash')
            hashes = filtered_results.index.unique()

            resampled_values = np.zeros(n_resamp, dtype=complex)

            # Perform resampling n_resamp times and calculate correlation
            # length using resampled results.
            for i_resamp in range(n_resamp):
                resampled_hashes = bs_rng.choice(hashes, size=hashes.size)
                resampled_results = filtered_results.loc[resampled_hashes]

                variables = []
                for label in columns:
                    if label in filtered_results.columns:
                        variables.append(resampled_results[label].mean())
                    elif label in row:
                        variables.append(row[label])

                resampled_values[i_resamp] = function(*variables)

            uncertainty[i_row] = resampled_values.std()

        return uncertainty

    def estimate_heat_capacity(self):
        estimates = self.estimates
        estimates['HeatCapacity_estimate'] = heat_capacity(
            estimates['Energy_estimate'], estimates['Energy_2_estimate'],
            estimates['temperature']
        )
        estimates['HeatCapacity_uncertainty'] = self.bootstrap_uncertainty(
            heat_capacity,
            ['Energy', 'Energy_2', 'temperature']
        )

        estimates['SpecificHeat_estimate'] = (
            estimates['HeatCapacity_estimate']/estimates['n_spins']
        )
        estimates['SpecificHeat_uncertainty'] = (
            estimates['HeatCapacity_uncertainty']/estimates['n_spins']
        )

    def estimate_binder(self, n_resamp: int = 10):
        """Estimate Binder cumulant."""
        estimates = self.estimates
        estimates['Binder_estimate'] = (
            1 - estimates['Magnetization_4_estimate']/(
                3*estimates['Magnetization_2_estimate']**2
            )
        )

        # Calculate uncertainty by bootstrapping.
        uncertainty = np.zeros(estimates.shape[0])

        # Generator for bootstrapping.
        bs_rng = np.random.default_rng(0)

        for i_row, row in self.estimates.iterrows():
            parameters = row[self.independent_variables].drop('tau')

            # Hashes for disorder configurations matching row parameters.
            hashes = self.inputs_df[
                (self.inputs_df[parameters.index] == parameters).all(axis=1)
            ]['hash'].values

            # Raw results filtered by matching disorders and tau.
            filtered_results = self.results_df[
                self.results_df['hash'].isin(hashes)
                & (self.results_df['tau'] == row['tau'])
            ].set_index('hash')
            hashes = filtered_results.index.unique()

            resampled_binder = np.zeros(n_resamp, dtype=complex)

            # Perform resampling n_resamp times and calculate correlation
            # length using resampled results.
            for i_resamp in range(n_resamp):
                resampled_hashes = bs_rng.choice(hashes, size=hashes.size)
                mag_4 = filtered_results.loc[
                    resampled_hashes
                ]['Magnetization_4'].mean()
                mag_2 = filtered_results.loc[
                    resampled_hashes
                ]['Magnetization_2'].mean()
                resampled_binder[i_resamp] = 1 - mag_4/(3*mag_2**2)

            uncertainty[i_row] = resampled_binder.std()

        estimates['Binder_uncertainty'] = uncertainty

    def estimate_correlation_length(self, n_resamp: int = 10):
        """Estimate the correlation length and ratio."""

        estimates = self.estimates
        k_min = 2*np.pi/estimates[['L_x', 'L_y']].max(axis=1)
        estimates['CorrelationLength_estimate'] = 1/(
            2*np.sin(k_min/2)
        )*np.sqrt(
            estimates['Susceptibility0_estimate'].astype(complex)
            / estimates['Susceptibilitykmin_estimate']
            - 1
        )

        # Calculate uncertainty by bootstrapping.
        uncertainty = np.zeros(estimates.shape[0])

        # Generator for bootstrapping.
        bs_rng = np.random.default_rng(0)

        for i_row, row in self.estimates.iterrows():
            parameters = row[self.independent_variables].drop('tau')

            # Hashes for disorder configurations matching row parameters.
            hashes = self.inputs_df[
                (self.inputs_df[parameters.index] == parameters).all(axis=1)
            ]['hash'].values

            # Raw results filtered by matching disorders and tau.
            filtered_results = self.results_df[
                self.results_df['hash'].isin(hashes)
                & (self.results_df['tau'] == row['tau'])
            ].set_index('hash')
            hashes = filtered_results.index.unique()

            resampled_correlation_lengths = np.zeros(n_resamp, dtype=complex)

            # Perform resampling n_resamp times and calculate correlation
            # length using resampled results.
            for i_resamp in range(n_resamp):
                resampled_hashes = bs_rng.choice(hashes, size=hashes.size)
                susceptibility0_disorder_mean = filtered_results.loc[
                    resampled_hashes
                ]['Susceptibility0'].mean()
                susceptibilitykmin_disorder_mean = filtered_results.loc[
                    resampled_hashes
                ]['Susceptibilitykmin'].mean()
                resampled_correlation_lengths[i_resamp] = 1/(
                    2*np.sin(k_min[i_row]/2)
                )*np.sqrt(
                    susceptibility0_disorder_mean.astype(complex)
                    / susceptibilitykmin_disorder_mean
                    - 1
                )

            uncertainty[i_row] = resampled_correlation_lengths.std()

        estimates['CorrelationLength_uncertainty'] = uncertainty

        # Calculate the correlation ratio xi/L by dividing by size.
        L_max = estimates[['L_x', 'L_y']].max(axis=1)
        estimates['CorrelationRatio_estimate'] = (
            estimates['CorrelationLength_estimate']/L_max
        )
        estimates['CorrelationRatio_uncertainty'] = (
            estimates['CorrelationLength_uncertainty']/L_max
        )

    def calculate_run_time_stats(self):
        """Calculate run time stats."""
        self.run_time_df = self.inputs_df.merge(
            self.results_df
        )[self.independent_variables + ['run_time']]

        rtgroup = self.run_time_df.groupby(
            self.independent_variables
        )['run_time']
        self.run_time_stats = pd.DataFrame({
            'count': rtgroup.count(),
            'mean_time': rtgroup.mean(),
            'max_time': rtgroup.max(),
            'total_time': rtgroup.sum(),
        }).reset_index()

        self.run_time_constants = dict()
        for spin_model in self.run_time_stats['spin_model'].unique():
            run_times = self.run_time_stats[
                self.run_time_stats['spin_model'] == spin_model
            ]
            self.run_time_constants[spin_model] = np.sum(
                run_times['total_time']
            )/np.sum(
                count_updates(spin_model, run_times)*run_times['count']
            )

    def estimate_run_time(
        self, inputs: List[dict], max_tau: int, n_disorder: int
    ) -> float:
        """Estimate run time in seconds for a list of proposed inputs."""
        total_time = 0
        for entry in inputs:
            constant = self.run_time_constants[entry['spin_model']]
            params = dict(entry['spin_model_params'])

            # Add 1 because $\sum_{i=0}^{n} 2^i = 2^{n+1} - 1$.
            params['tau'] = max_tau + 1

            # The number of disorders is the count.
            params['count'] = n_disorder
            entry_time = constant*count_updates(entry['spin_model'], params)
            total_time += entry_time
        return total_time


def count_updates(spin_model: str, params) -> float:
    """Number of count updates."""
    if spin_model == 'RandomBondIsingModel2D':
        return params['L_x']*params['L_y']*2**params['tau']
    if spin_model == 'LoopModel2D':
        return 2*params['L_x']*params['L_y']*2**params['tau']
    else:
        return 1


def load_analysis(data_dir):
    """Load an analysis that had already been finished and saved."""
    analysis_dir = os.path.join(data_dir, 'analysis')

    analysis = SimpleAnalysis(data_dir)

    analysis_json = os.path.join(analysis_dir, 'analysis.json')

    with open(analysis_json) as f:
        analysis_attr = json.load(f)
    analysis.observable_names = analysis_attr['observable_names']
    analysis.independent_variables = analysis_attr['independent_variables']
    analysis.run_time_constants = analysis_attr['run_time_constants']

    estimates_pkl = os.path.join(analysis_dir, 'estimates.pkl')
    results_pkl = os.path.join(analysis_dir, 'results.pkl')
    inputs_pkl = os.path.join(analysis_dir, 'inputs.pkl')
    analysis_json = os.path.join(analysis_dir, 'analysis.json')

    analysis.estimates = pd.read_pickle(estimates_pkl)
    analysis.results_df = pd.read_pickle(results_pkl)
    analysis.inputs_df = pd.read_pickle(inputs_pkl)
    return analysis
