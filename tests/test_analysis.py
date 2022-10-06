import os
import json
import pytest
import numpy as np
import pandas as pd
from panqec.analysis import (
    get_subthreshold_fit_function, get_single_qubit_error_rate, Analysis,
    deduce_bias, fill_between_values, count_fails
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


class TestAnalysis:

    def test_analyse_toric_2d_results(self):
        results_path = os.path.join(DATA_DIR, 'toric')
        assert os.path.exists(results_path)
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 'inf'}, 'truncate': {'probability': {
                'min': 0.06, 'max': 0.14,
            }}},
        ]})
        analysis.analyze()
        results_required = [
            'code', 'error_model', 'decoder',
            'probability',
            'p_est', 'n_trials', 'n_fail',
        ]
        assert set(analysis.results.columns) == set([
            'size', 'code', 'n', 'k', 'd', 'error_model', 'decoder',
            'probability', 'wall_time', 'n_trials', 'n_fail',
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

    # TODO come up with spec for different sectors.
    def test_different_sector_different_truncations(self):
        results_path = os.path.join(DATA_DIR, 'toric')
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 0.5}, 'skip': True},
            {'filters': {'bias': 'inf'}, 'truncate': {'probability': {
                'min': 0.06, 'max': 0.14,
            }}},
        ]})
        analysis.analyze()

    def test_skip_entry(self):
        results_path = os.path.join(DATA_DIR, 'toric')
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 0.5}, 'skip': True},
            {'filters': {'bias': 'inf'}, 'truncate': {'probability': {
                'min': 0.06, 'max': 0.14,
            }}},
        ]})
        analysis.analyze()
        assert 0.5 not in analysis.thresholds['bias'].values
        assert analysis.thresholds.shape[0] > 0

    def test_replace_threshold_with_given_value(self):
        results_path = os.path.join(DATA_DIR, 'toric')
        analysis = Analysis(results_path, overrides={'overrides': [
            {'filters': {'bias': 'inf'}, 'truncate': {'probability': {
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
            'probability': 0.18
          },
          {
            'code_family': 'Toric',
            'error_model': 'Deformed XZZX Pauli X0.0161Y0.0161Z0.9677',
            'decoder': 'BP-OSD decoder',
            'bias': 30,
            'code': 'Toric 9x9x9',
            'probability': 0.1
          },
          {
            'code_family': 'Toric',
            'error_model': 'Pauli X0.0000Y0.0000Z1.0000',
            'decoder': 'BP-OSD decoder',
            'bias': 'inf',
            'code': 'Toric 9x9x9',
            'probability': 0.204
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
                        'probability': [0.1, 0.2]
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
                    'probability': [0.1, 0.2]
                }
            }


def test_convert_missing_to_input():
    missing_entry = {
        'code': 'Rhombic 10x10x10',
        'error_model': 'Deformed Rhombic Pauli X0.3333Y0.3333Z0.3333',
        'decoder': 'BP-OSD decoder',
        'probability': 0.01,
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
            'model': 'RhombicCode',
            'parameters': {'L_x': 10}
        },
        'noise': {
            'model': 'DeformedRhombicErrorModel',
            'parameters': {
                'r_x': 0.3333333333333333,
                'r_y': 0.33333333333333337,
                'r_z': 0.33333333333333337,
            }
        },
        'decoder': {
            'model': 'BeliefPropagationOSDDecoder',
            'parameters': {
                'max_bp_iter': 1000,
                'osd_order': 0
            }
        },
        'probability': 0.01,
        'trials': 4774,
    }
    analysis = Analysis()
    entry = analysis.convert_missing_to_input(missing_entry)
    assert entry == expected_entry


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


class TestFillBetweenValues:

    @pytest.mark.parametrize('old_values,n_target,new_values', [
        ([1, 2.1, 4], 4, [1, 2.1, 3, 4]),
        ([1, 2, 4, 5, 7, 8], 8, [1, 2, 3, 4, 5, 6, 7, 8]),
    ])
    def test_easy_cases(self, old_values, n_target, new_values):
        assert new_values == fill_between_values(old_values, n_target)
        assert set(old_values).issubset(new_values)

    def test_long_array_does_not_change(self):
        old_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        n_target = 8
        new_values = fill_between_values(old_values, n_target)
        assert new_values == old_values


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
