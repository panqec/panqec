import os
import json
import re
import pytest
import numpy as np
import pandas as pd
from panqec.analysis import (
    get_subthreshold_fit_function, get_single_qubit_error_rate, Analysis,
    deduce_bias, count_fails
)
from panqec.simulation import read_input_json
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def test_subthreshold_cubic_fit_function():
    C_0, C_1, C_2, C_3 = 0, 1, 2, 1
    log_p_L_th, log_p_th = np.log(0.5), np.log(0.5)

    subthreshold_fit_function = get_subthreshold_fit_function(order=3)

    p = 0.5
    L = 10
    log_p_L = subthreshold_fit_function(
        (np.log(p), L),
        log_p_L_th, log_p_th, C_0, C_1, C_2, C_3
    )
    assert log_p_L == np.log(0.5)


class TestGetSingleQubitErrorRates:

    def test_trivial(self):
        effective_error_list = []
        p_est, p_se = get_single_qubit_error_rate(effective_error_list)
        assert np.isnan(p_est) and np.isnan(p_se)

    def test_simple_example(self):
        effective_error_list = [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ]

        n_results = len(effective_error_list)
        assert n_results == 7

        # Rate of any error occuring should be 3/4.
        p_est, p_se = get_single_qubit_error_rate(effective_error_list, i=0)
        assert np.isclose(p_est, (n_results - 1)/n_results)

        # Probability of each error type occuring should be based on count.
        p_x, p_x_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='X'
        )
        assert np.isclose(p_x, 1/n_results)

        p_y, p_y_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='Y'
        )
        assert np.isclose(p_y, 2/n_results)

        p_z, p_z_se = get_single_qubit_error_rate(
            effective_error_list, i=0, error_type='Z'
        )
        assert np.isclose(p_z, 3/n_results)

    def test_simple_example_on_2nd_qubit(self):
        effective_error_list = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
        ]

        # Rate of any error occuring should be 3/4.
        p_est, p_se = get_single_qubit_error_rate(effective_error_list, i=1)
        assert np.isclose(p_est, 6/7)
        assert np.isclose(p_se, np.sqrt((6/7)*(1 - (6/7))/(7 + 1)))

        # Probability of each error type occuring should be based on count.
        p_x, p_x_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='X'
        )
        assert np.isclose(p_x, 1/7)
        assert np.isclose(p_x_se, np.sqrt(p_x*(1 - p_x)/(7 + 1)))

        p_y, p_y_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='Y'
        )
        assert np.isclose(p_y, 2/7)

        p_z, p_z_se = get_single_qubit_error_rate(
            effective_error_list, i=1, error_type='Z'
        )
        assert np.isclose(p_z, 3/7)


@pytest.mark.xfail
class TestAnalysis:

    def test_save_and_load_analysis(self, tmpdir):
        results_path = os.path.join(DATA_DIR, 'toric')
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 'inf'}, 'truncate': {'error_rate': {
                'min': 0.06, 'max': 0.14,
            }}},
        ]})
        # analysis.analyze()
        save_path = os.path.join(tmpdir, 'myanalysis.json.gz')
        analysis.save(save_path)

        new_analysis = Analysis()
        new_analysis.load(save_path)
        assert new_analysis.results.shape[0] > 0
        assert new_analysis.results.shape[0] == analysis.results.shape[0]
        assert new_analysis.thresholds.shape[0] > 0
        assert new_analysis.thresholds.shape[0] == analysis.thresholds.shape[0]
        assert 'total' in new_analysis.sectors
        assert 'X' in new_analysis.sectors
        assert 'Z' in new_analysis.sectors
        for sector in new_analysis.sectors:
            for table_name in ['trunc_results', 'thresholds']:
                assert (
                    new_analysis.sectors[sector][table_name].shape[0]
                    == analysis.sectors[sector][table_name].shape[0]
                )

        # Attempt to make plots using loaded Analysis.
        import matplotlib
        import warnings
        matplotlib.use('Agg')
        warnings.filterwarnings('ignore')
        new_analysis.make_plots(tmpdir)
        assert len([
            name for name in os.listdir(tmpdir) if '.pdf' in name
        ]) == 2

    def test_analyse_toric_2d_results(self, tmpdir):
        results_path = os.path.join(DATA_DIR, 'toric')
        assert os.path.exists(results_path)
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 'inf'}, 'truncate': {'error_rate': {
                'min': 0.06, 'max': 0.14,
            }}},
        ]})
        analysis.analyze()
        results_required = [
            'code', 'error_model', 'decoder',
            'error_rate',
            'p_est', 'n_trials', 'n_fail',
        ]
        assert set(analysis.results.columns) == set([
            'size', 'code', 'n', 'k', 'd', 'error_model', 'decoder',
            'error_rate', 'wall_time', 'n_trials', 'n_fail',
            'effective_error', 'success', 'codespace', 'bias', 'results_file',
            'p_est', 'p_se', 'p_word_est', 'p_word_se', 'single_qubit_p_est',
            'single_qubit_p_se', 'code_family', 'error_model_family',
            'p_est_X', 'p_se_X', 'n_fail_X', 'n_trials_X',
            'p_est_Z', 'p_se_Z', 'n_fail_Z', 'n_trials_Z',
        ])
        threshold_required = [
            'code_family', 'error_model', 'decoder',
            'p_th_fss', 'p_th_fss_left', 'p_th_fss_right',
            'fss_params', 'params_bs',
            'fit_found'
        ]
        assert (
            set(analysis.thresholds.columns).intersection(threshold_required)
            == set(threshold_required)
        ), 'thresholds should have required columns'
        assert (
            set(analysis.trunc_results.columns).intersection(results_required)
            == set(results_required)
        ), 'trunc_results should have required columns'

        for sector in ['X', 'Z']:
            assert (
                set(analysis.sectors[sector]['thresholds']).intersection(
                    threshold_required
                ) == set(threshold_required)
            ), f'{sector} thresholds should have required columns'
            assert (
                set([
                    'code', 'error_model', 'decoder', 'rescaled_p',
                ]).issubset(analysis.sectors[sector]['trunc_results'])
            ), f'{sector} trunc_results should have required columns'

        import matplotlib
        import warnings
        matplotlib.use('Agg')
        warnings.filterwarnings('ignore')
        analysis.make_plots(tmpdir)
        assert any([
            bool(re.match(r'.*collapse.pdf$', name))
            for name in os.listdir(tmpdir)
        ])
        assert any([
            bool(re.match(r'.*thresholds-vs-bias.pdf$', name))
            for name in os.listdir(tmpdir)
        ])

    def test_different_sector_different_truncations(self):
        results_path = os.path.join(DATA_DIR, 'toric3d.zip')
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 0.5}, 'skip': True},
            {'filters': {'bias': 'inf'}, 'skip': True},
            {'filters': {'bias': 10}, 'truncate': {
                'error_rate': {'min': 0.18, 'max': 0.28}
            }},
            {'filters': {'bias': 10}, 'sector': 'X', 'truncate': {
                'error_rate': {'min': 0.20, 'max': 0.40}
            }},
            {'filters': {'bias': 10}, 'sector': 'Z', 'truncate': {
                'error_rate': {'min': 0.18, 'max': 0.28}
            }},
        ]})
        analysis.analyze()
        assert analysis.thresholds.shape[0] == 1

        # Check the overall threshold.
        threshold = analysis.thresholds['p_th_fss'].iloc[0]
        assert 0.22 < threshold and threshold < 0.23

        # Check the sector thresholds.
        t_thresh = analysis.sectors['total']['thresholds']['p_th_fss'].iloc[0]
        assert 0.23 < t_thresh and t_thresh < 0.24

        x_thresh = analysis.sectors['X']['thresholds']['p_th_fss'].iloc[0]
        assert 0.32 < x_thresh and x_thresh < 0.33

        z_thresh = analysis.sectors['Z']['thresholds']['p_th_fss'].iloc[0]
        assert 0.22 < z_thresh and z_thresh < 0.23

        # Check the custom sector-by-sector truncation was done correctly.
        epsilon = 1e-5
        p_X = analysis.sectors['X']['trunc_results']['error_rate'].unique()
        assert all(p_X <= 0.40 + epsilon)
        assert all(0.20 - epsilon <= p_X)

        p_Z = analysis.sectors['Z']['trunc_results']['error_rate'].unique()
        assert all(p_Z <= 0.28 + epsilon)
        assert all(0.18 - epsilon <= p_Z)

    def test_skip_entry(self):
        results_path = os.path.join(DATA_DIR, 'toric')
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 0.5}, 'skip': True},
            {'filters': {'bias': 'inf'}, 'truncate': {'error_rate': {
                'min': 0.06, 'max': 0.14,
            }}},
        ]})
        analysis.analyze()
        assert 0.5 not in analysis.thresholds['bias'].values
        assert analysis.thresholds.shape[0] > 0

    def test_replace_threshold_with_given_value(self):
        results_path = os.path.join(DATA_DIR, 'toric')
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 'inf'}, 'truncate': {'error_rate': {
                'min': 0.06, 'max': 0.14,
            }}},
            {
              "filters": {
                "code_family": "Toric",
                "error_model_family": "Deformed XZZX Pauli",
                "bias": "inf",
                "decoder": "BP-OSD decoder"
              },
              "replace": {"p_th_fss": 0.5, "p_th_fss_se": 0.01}
            }
        ]})
        analysis.analyze()
        assert 0.5 not in analysis.thresholds['bias'].values
        assert analysis.thresholds.shape[0] > 0

    @pytest.mark.skip(reason='missing inputs')
    def test_generate_missing_inputs_can_be_read(self, tmpdir):

        class FakeAnalysis(Analysis):
            """Analysis class with fake get_missing_points() method"""
            def get_missing_points(self):
                with open(os.path.join(DATA_DIR, 'fake_missing.json')) as f:
                    missing = pd.DataFrame(json.load(f))
                missing = missing.groupby(self.POINT_KEYS).first()
                return missing

        # Generate the inputs file.
        path = os.path.join(tmpdir, 'missing.json')
        analysis = FakeAnalysis(os.path.join(DATA_DIR, 'toric'))
        analysis.generate_missing_inputs(path)
        assert os.path.isfile(path)

        # Try to read in the generated file for simulation.
        batch_sim = read_input_json(path)
        assert len(batch_sim) == 6

    def test_apply_overrides(self):
        analysis = Analysis()
        for sector in analysis.SECTOR_KEYS:
            assert not analysis.overrides[sector]
        analysis.results = pd.DataFrame([
          {
            'code_family': 'Toric',
            'error_model': 'Deformed XZZX Pauli X0.0005Y0.0005Z0.9990',
            'decoder': 'BP-OSD decoder',
            'bias': 1000,
            'code': 'Toric 9x9x9',
            'error_rate': 0.18
          },
          {
            'code_family': 'Toric',
            'error_model': 'Deformed XZZX Pauli X0.0161Y0.0161Z0.9677',
            'decoder': 'BP-OSD decoder',
            'bias': 30,
            'code': 'Toric 9x9x9',
            'error_rate': 0.1
          },
          {
            'code_family': 'Toric',
            'error_model': 'Pauli X0.0000Y0.0000Z1.0000',
            'decoder': 'BP-OSD decoder',
            'bias': 'inf',
            'code': 'Toric 9x9x9',
            'error_rate': 0.204
          }
        ])
        analysis.overrides_spec = {
            'overrides': [
                {
                    'filters': {
                        'code_family': 'Toric',
                        'bias': 30,
                        'decoder': 'BP-OSD decoder'
                    },
                    'truncate': {
                        'error_rate': [0.1, 0.2]
                    }
                }
            ]
        }
        analysis.apply_overrides()
        for sector in analysis.SECTOR_KEYS:
            assert analysis.overrides[sector] == {
                (
                    'Toric', 'Deformed XZZX Pauli X0.0161Y0.0161Z0.9677',
                    'BP-OSD decoder'
                ): {
                    'error_rate': [0.1, 0.2]
                }
            }


@pytest.mark.skip(reason='missing inputs')
def test_convert_missing_to_input():
    missing_entry = {
        'code': 'Rhombic 10x10x10',
        'error_model': 'Deformed Checkerboard XZZX'
                       'Rhombic Pauli X0.3333Y0.3333Z0.3333',
        'decoder': 'BP-OSD decoder',
        'error_rate': 0.01,
        'code_family': 'Rhombic',
        'error_model_family': 'Deformed Rhombic Pauli',
        'bias': 0.5,
        'd': 10,
        'n_trials': 5226,
        'n_missing': 4774,
        'time_per_trial': 1.2045530740528128,
        'time_remaining': pd.Timedelta('0 days 01:35:50.536375528')
    }
    expected_entry = {
        'label': 'unlabelled',
        'code': {
            'name': 'RhombicToricCode',
            'parameters': {'L_x': 10}
        },
        'error_model': {
            'name': 'PauliErrorModel',
            'parameters': {
                'r_x': 0.3333333333333333,
                'r_y': 0.33333333333333337,
                'r_z': 0.33333333333333337,
                'deformation_name': 'Checkerboard XZZX',
                'deformation_axis': 'z'
            }
        },
        'decoder': {
            'name': 'BeliefPropagationOSDDecoder',
            'parameters': {
                'max_bp_iter': 1000,
                'osd_order': 0
            }
        },
        'error_rate': 0.01,
        'trials': 4774,
    }
    analysis = Analysis()
    entry = analysis.convert_missing_to_input(missing_entry)
    assert entry == expected_entry


@pytest.mark.xfail
@pytest.mark.parametrize('error_model,bias', [
    ('Deformed XZZX Pauli X0.0005Y0.0005Z0.9990', 1000),
    ('Deformed XZZX Pauli X0.0050Y0.0050Z0.9901', 100),
    ('Pauli X0.0000Y0.0000Z1.0000', 'inf'),
    ('Pauli X0.0455Y0.0455Z0.9091', 10),
    ('Deformed XZZX Pauli X0.1250Y0.1250Z0.7500', 3),
    ('Pauli X0.3333Y0.3333Z0.3333', 0.5),
])
def test_deduce_bias(error_model, bias):
    assert bias == deduce_bias(error_model)


class TestCountFails:
    def test_easy_case(self):
        effective_error = np.array([
            [0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1],
        ], dtype=np.uint8)
        codespace = np.array([True, True, False, True], dtype=bool)
        assert count_fails(effective_error, codespace, 'X') == 6
        assert count_fails(effective_error, codespace, 'Z') == 4


class TestAnalysisClusterTutorial:

    def test_analyze_cluster_example(self):
        results_json = os.path.join(DATA_DIR, 'merged-results.json.gz')
        analysis = Analysis(results_json)
        analysis.get_results()
        analysis.thresholds
        analysis.sector_thresholds
        analysis.min_thresholds
