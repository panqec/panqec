import os
import json
import pytest
from panqec.statmech.controllers import SimpleController, DataManager

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAMPLE_INPUTS_JSON = os.path.join(BASE_DIR, 'sample_inputs.json')


@pytest.fixture
def controller(tmpdir, inputs):
    con = SimpleController(tmpdir)
    return con


@pytest.fixture
def inputs():
    with open(SAMPLE_INPUTS_JSON) as f:
        inputs = json.load(f)
    return inputs


class TestSimpleController:

    def test_integration(self, controller, inputs):
        max_tau = 5
        controller.data_manager.save('inputs', inputs)
        controller.run(max_tau)
        summary = controller.get_results()
        assert len(summary) == len(inputs)*(max_tau + 1)


@pytest.fixture
def data_manager(tmpdir):
    dm = DataManager(tmpdir)
    dm.save('results', [
        {
            'hash': '000b',
            'model': '000b',
            'seed': 0,
            'tau': 3,
        },
        {
            'hash': '000a',
            'model': '000a',
            'seed': 0,
            'tau': 0,
        },
        {
            'hash': '000a',
            'model': '001a',
            'seed': 0,
            'tau': 1,
        }
    ])
    return dm


class TestDataManager:

    def test_directory_structure(self, data_manager):
        for subdir in ['inputs', 'results', 'models', 'runs']:
            assert os.path.exists(data_manager.subdirs[subdir])

    def test_get_name_results(self, data_manager):
        name = data_manager.get_name('results', {
            'hash': '0123456789abcdef',
            'seed': 0,
            'tau': 5,
        })
        assert name == 'results_tau5_0123456789abcdef_seed0.gz'

    def test_get_name_inputs(self, data_manager):
        name = data_manager.get_name('inputs', {
            'hash': '0123456789abcdef',
            'disorder': [-1, 1],
            'disorder_model': 'Rbim2DIidDisorder',
            'disorder_params': {'p': 0.1},
            'spin_model': 'RandomBondIsingModel2D',
            'spin_model_params': {'L_x': 12, 'L_y': 12},
            'temperature': 1.23,
        })
        assert name == 'input_0123456789abcdef.gz'

    def test_load_results(self, data_manager):
        results = data_manager.load('results', {
            'hash': '000a',
        })
        assert len(results) == 2
