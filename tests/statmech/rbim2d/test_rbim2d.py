import pytest
import numpy as np
from bn3d.statmech.rbim2d import RandomBondIsingModel2D


class TestRBIM2DNoDisorder:
    L_x = 4
    L_y = 5

    @pytest.fixture(autouse=True)
    def model(self):
        """An instance of a model with default."""
        model = RandomBondIsingModel2D(self.L_x, self.L_y)
        model.rng = np.random.default_rng(seed=0)
        model.init_spins(np.ones_like(model.spins))
        model.init_disorder(np.ones_like(model.disorder))
        return model

    def test_default_attributes(self, model):
        assert model.temperature == 1
        assert model.moves_per_sweep == self.L_x*self.L_y
        assert np.all(model.disorder == 1)
        assert np.all(model.spins == 1)
        assert np.all(model.couplings == 1)

    def test_label(self, model):
        assert model.label == 'RandomBondIsingModel2D 4x5'

    def test_shapes(self, model):
        assert model.spins.shape == (self.L_x, self.L_y)
        assert model.disorder.shape == (2, self.L_x, self.L_y)
        assert model.disorder.shape == model.couplings.shape

    def test_energy_diff_all_up(self, model):
        energy_diff = model.delta_energy((0, 0))
        assert energy_diff == -8.0

    def test_sample_changes_spins(self, model):
        n_sweeps = 3
        spins_0 = model.spins.copy()
        model.sample(n_sweeps)
        spins_1 = model.spins.copy()
        assert np.any(spins_0 != spins_1)

    def test_random_move_and_update_flips_spin(self, model):
        spins_0 = model.spins.copy()
        move = model.random_move()
        model.update(move)
        spins_1 = model.spins.copy()
        assert np.any(spins_0 != spins_1)
        assert spins_0[move] == -spins_1[move]
