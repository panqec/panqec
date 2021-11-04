import json
import pytest
import numpy as np
from bn3d.statmech.loop2d import LoopModel2D


class TestLoopModel2D:
    L_x = 4
    L_y = 5

    @pytest.fixture(autouse=True)
    def model(self):
        """An instance of a model with default."""
        model = LoopModel2D(self.L_x, self.L_y)
        model.rng = np.random.default_rng(seed=0)
        model.init_spins()
        model.init_disorder()
        return model

    def test_default_attributes(self, model):
        assert model.temperature == 1
        assert model.n_spins == 2*self.L_x*self.L_y
        assert model.moves_per_sweep == model.n_spins

    def test_spins_are_on_edges(self, model):
        spin_indices = np.array(np.where(model.spins)).T
        for x, y in spin_indices:
            assert (x % 2, y % 2) in [(0, 1), (1, 0)]

    def test_disorders_are_on_vertices(self, model):
        bond_indices = np.array(np.where(model.disorder)).T
        for x, y in bond_indices:
            assert (x % 2, y % 2) == (0, 0)

    def test_couplings_are_on_vertices(self, model):
        coupling_indices = np.array(np.where(model.couplings)).T
        for x, y in coupling_indices:
            assert (x % 2, y % 2) == (0, 0)

    def test_label(self, model):
        assert model.label == 'LoopModel2D 4x5'

    def test_shapes(self, model):
        assert model.spins.shape == (2*self.L_x, 2*self.L_y)
        assert model.disorder.shape == (2*self.L_x, 2*self.L_y)
        assert model.disorder.shape == model.couplings.shape

    @pytest.mark.xfail
    def test_energy_diff_all_up(self, model):
        energy_diff = model.delta_energy((0, 0))
        assert energy_diff == 4

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

    def test_to_json_and_load_json(self, model):
        model.init_spins()
        data = model.to_json()
        data_str = json.dumps(data)
        data_reread = json.loads(data_str)

        old_stats = model.sample(1)

        new_model = LoopModel2D(self.L_x, self.L_y)
        new_model.load_json(data_reread)

        new_stats = new_model.sample(1)

        # Check that output is exactly reproduced.
        assert old_stats['total'] == new_stats['total']
        assert old_stats['acceptance'] == new_stats['acceptance']
        assert np.all(model.spins == new_model.spins)
