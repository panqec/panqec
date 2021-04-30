import os
import pytest
from bn3d.app import read_input_json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


@pytest.mark.parametrize(
    'file_name, expected_runs',
    [
        ('single_input.json', 1),
        ('range_input.json', 27),
    ]
)
def test_read_json_input(file_name, expected_runs):
    single_input_json = os.path.join(DATA_DIR, file_name)
    batch_simulation = read_input_json(single_input_json)
    assert batch_simulation is not None
    assert len(batch_simulation._simulations) == expected_runs
