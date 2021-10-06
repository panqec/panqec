import os
import json
import pytest
from bn3d.statmech.controllers import SimpleController

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
        controller.init_models(inputs)
        assert len(inputs) == 2
        assert len(controller.models) == len(inputs)

        max_tau = 5
        controller.run(max_tau)
        summary = controller.get_summary()
        assert len(summary) == len(inputs)*(max_tau + 1)
