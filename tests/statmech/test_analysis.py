import os
import shutil
import pytest
from bn3d.statmech.analysis import SimpleAnalysis

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_results')


@pytest.fixture
def analysis(tmpdir):
    data_dir = os.path.join(tmpdir, 'sample_results')
    shutil.copytree(SAMPLE_DIR, data_dir)
    assert os.path.isdir(data_dir)
    an = SimpleAnalysis(data_dir)
    return an


def test_analyse(analysis):
    analysis.analyse()
    expected_columns = set([
        'L_x', 'L_y', 'p', 'temperature', 'tau', 'Magnetization_estimate',
        'Magnetization_uncertainty', 'Magnetization_n_disorders',
        'Susceptibility0_estimate', 'Susceptibility0_uncertainty',
        'Susceptibility0_n_disorders', 'Susceptibilitykmin_estimate',
        'Susceptibilitykmin_uncertainty', 'Susceptibilitykmin_n_disorders',
        'CorrelationLength_estimate',
        'CorrelationLength_uncertainty',
    ])
    assert analysis.estimates.shape[0] == 2
    assert expected_columns.issubset(analysis.estimates.columns)


def test_estimate_run_time(analysis):
    analysis.analyse()
    estimated_time = analysis.estimate_run_time(
        inputs=analysis.data_manager.load('inputs'),
        max_tau=analysis.results_df['tau'].max(),
        n_disorder=analysis.run_time_stats['count'].max()
    )
    actual_time = analysis.run_time_stats['total_time'].sum()

    # Estimate should be within 50% of the actual.
    relative_difference = abs(estimated_time - actual_time)/actual_time
    assert relative_difference < 0.5
