import os
import pytest
from shutil import copyfile
from bn3d.statmech.core import generate_inputs, start_sampling
from bn3d.statmech.controllers import DataManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


@pytest.fixture
def data_dir(tmpdir):
    copyfile(
        os.path.join(DATA_DIR, 'targets.json'),
        os.path.join(tmpdir, 'targets.json')
    )
    return tmpdir


def test_generate_inputs(data_dir):
    assert os.path.exists(os.path.join(data_dir, 'targets.json'))
    generate_inputs(data_dir)
    data_manager = DataManager(data_dir)
    inputs = data_manager.load('inputs')
    assert len(inputs) == 1*2*3*4
    assert os.path.isfile(os.path.join(data_dir, 'info.json'))


def test_start_sampling(data_dir):
    generate_inputs(data_dir)
    start_sampling(data_dir)
    data_manager = DataManager(data_dir)
    results = data_manager.load('results')
    assert len(results) == 1*2*3*4*5
