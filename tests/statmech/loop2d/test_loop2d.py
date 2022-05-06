import json
import pytest
import numpy as np
from panqec.statmech.loop2d import LoopModel2D, LoopModel2DIidDisorder
from tests.statmech.utils import assert_flip_energies_consistent


class TestLoopModel2DEnergy:

    @pytest.fixture(autouse=True)
    def all_up(self):
        L_x, L_y = 2, 3
        model = LoopModel2D(L_x, L_y)
        assert np.all((model.spins == 1) | (model.spins == 0))
        return model

    def test_all_up_total_energy(self, all_up):
        model = all_up
        assert model.total_energy() == -model.n_bonds

    def test_flip_one_by_one(self, all_up):
        model = all_up
        for move in model.spin_index:
            assert_flip_energies_consistent(model, move)


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

    def test_spins_are_pm_one(self, model):
        for x, y in model.spin_index:
            spin = model.spins[x, y]
            assert (spin == 1) or (spin == -1)

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

    def test_energy_diff_all_up(self, model):
        move = model.random_move()
        energy_diff = model.delta_energy(move)
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


class TestLoopModel2DIidDisorder:
    L_x = 5
    L_y = 6

    def test_disorder_reproducible_seed(self):
        seed = 0
        spin_model_params = {'L_x': 5, 'L_y': 6}
        disorder_params = {'p': 0.4}

        # Generate some disorder configuration using the seed.
        rng = np.random.default_rng(seed)
        disorder_model = LoopModel2DIidDisorder(rng)
        disorder_1 = disorder_model.generate(
            spin_model_params, disorder_params
        )

        # Do it again.
        rng = np.random.default_rng(seed)
        disorder_model = LoopModel2DIidDisorder(rng)
        disorder_2 = disorder_model.generate(
            spin_model_params, disorder_params
        )

        # Make sure they're the same.
        assert np.all(disorder_1 == disorder_2)

    def test_delta_energy_agrees_with_delta_energy_disordered(self):
        seed = 0
        spin_model_params = {'L_x': self.L_x, 'L_y': self.L_y}
        disorder_params = {'p': 0.4}

        rng = np.random.default_rng(seed)
        disorder_model = LoopModel2DIidDisorder(rng)

        disorder_model = LoopModel2DIidDisorder(rng)
        disorder = disorder_model.generate(
            spin_model_params, disorder_params
        )

        model = LoopModel2D(self.L_x, self.L_y)
        model.rng = rng
        model.init_spins()
        model.init_disorder(disorder)

        for i_move in range(100):
            initial_energy = model.total_energy()
            move = model.random_move()
            delta_energy = model.delta_energy(move)
            model.update(move)
            final_energy = model.total_energy()
            assert delta_energy == final_energy - initial_energy
