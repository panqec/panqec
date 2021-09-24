import pytest
import numpy as np
from bn3d.statmech.rbim2d import RandomBondIsingModel2D


class TestRBIM2D:
    L_x = 4
    L_y = 5

    @pytest.fixture(autouse=True)
    def model(self):
        model = RandomBondIsingModel2D(self.L_x, self.L_y)
        model.spins = np.ones_like(model.spins)
        model.disorder = np.ones_like(model.disorder)
        return model

    def test_shapes(self, model):
        assert model.spins.shape == (self.L_x, self.L_y)
        assert model.disorder.shape == (2, self.L_x, self.L_y)

    def test_energy_diff_all_up(self, model):
        energy_diff = model.delta_energy((0, 0))
        assert energy_diff == -8.0

    def test_run(self, model):
        model.run()
