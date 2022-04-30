import os
import shutil
import pytest
import gzip
import json
from glob import glob
from panqec.statmech.analysis import SimpleAnalysis

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_results')


@pytest.fixture
def analysis(tmpdir):
    data_dir = os.path.join(tmpdir, 'sample_results')
    shutil.copytree(SAMPLE_DIR, data_dir)
    subdirs = ['inputs', 'models', 'results', 'runs']
    os.makedirs(os.path.join(data_dir, 'runs'), exist_ok=True)

    for subdir in subdirs:
        assert os.path.isdir(os.path.join(data_dir, subdir))
        file_list = glob(os.path.join(data_dir, subdir, '*.json'))
        for json_path in file_list:
            gzip_path = os.path.splitext(json_path)[0] + '.gz'
            with open(json_path) as f:
                entry = json.load(f)
            with gzip.open(gzip_path, 'w') as f:
                f.write(
                    json.dumps(entry, sort_keys=True, indent=2)
                    .encode('utf-8')
                )
            os.remove(json_path)

    assert os.path.isdir(data_dir)
    an = SimpleAnalysis(data_dir)
    return an


def test_analyse(analysis):
    analysis.analyse()
    expected_columns = set([
        'L_x', 'L_y', 'p', 'temperature', 'tau', 'n_disorder',
        'Energy_estimate',
        'Magnetization_estimate', 'Magnetization_uncertainty',
        'Susceptibility0_estimate', 'Susceptibility0_uncertainty',
        'Susceptibilitykmin_estimate', 'Susceptibilitykmin_uncertainty',
        'CorrelationLength_estimate', 'CorrelationLength_uncertainty',
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
