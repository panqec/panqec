from itertools import product
import json
import pytest
import numpy as np
from panqec.statmech.rbim2d import RandomBondIsingModel2D, Rbim2DIidDisorder
from tests.statmech.utils import assert_flip_energies_consistent


class TestRBIM2DEnergy:

    @pytest.fixture(autouse=True)
    def antiferro_2x2(self):
        model = RandomBondIsingModel2D(2, 2)
        model.rng = np.random.default_rng(seed=0)
        model.init_spins(np.array([
            [1, -1],
            [-1, 1]
        ]))
        return model

    def test_total_energy_2x2_random(self):
        L_x = 2
        L_y = 2
        model = RandomBondIsingModel2D(L_x, L_y)
        model.rng = np.random.default_rng(seed=0)
        model.init_spins()

        for i_move in range(100):
            move = model.random_move()
            assert_flip_energies_consistent(model, move)

    def test_total_energy_all_up(self):
        model = RandomBondIsingModel2D(L_x=2, L_y=3)
        model.init_spins(np.ones(model.spin_shape, dtype=int))
        assert model.total_energy() == -model.n_bonds

    def test_total_energy_all_down(self):
        model = RandomBondIsingModel2D(L_x=2, L_y=3)
        model.init_spins(-np.ones(model.spin_shape, dtype=int))
        assert model.total_energy() == -model.n_bonds

    def test_total_energy_anti_ferromagnetic(self, antiferro_2x2):
        model = antiferro_2x2
        assert model.total_energy() == model.n_bonds

    def test_flip_spins_move_by_move(self):
        L_x, L_y = 2, 2
        model = RandomBondIsingModel2D(L_x, L_y)
        model.init_spins(np.ones(model.spin_shape, dtype=int))
        initial_energy = model.total_energy()
        delta_energy_list = []
        total_energy_change_list = []
        total_energy_list = []
        moves = list(product(range(L_x), range(L_y)))
        for i_move, move in enumerate(moves):
            energy_before_move = model.total_energy()
            delta_energy = model.delta_energy(move)
            delta_energy_list.append(delta_energy)
            model.update(move)
            energy_after_move = model.total_energy()
            total_energy_list.append(energy_after_move)
            energy_change = energy_after_move - energy_before_move
            total_energy_change_list.append(energy_change)
        final_energy = model.total_energy()
        assert final_energy == initial_energy
        assert delta_energy_list == total_energy_change_list
        assert initial_energy + sum(delta_energy_list) == final_energy

    def test_flip_anti_ferromagnetic(self, antiferro_2x2):
        L_x, L_y = 2, 2
        for move in product(range(L_x), range(L_y)):
            model = antiferro_2x2
            assert_flip_energies_consistent(
                model, move, message=f'Disagreement at (x, y) = {move}'
            )

    def test_flip_spins_everywhere_all_up(self):
        L_x = 2
        L_y = 3
        model = RandomBondIsingModel2D(L_x, L_y)
        for move in product(range(L_x), range(L_y)):
            model.init_spins(np.ones(model.spin_shape, dtype=int))
            assert_flip_energies_consistent(
                model, move, message=f'Disagreement at (x, y) = {move}'
            )


class TestRBIM2DNoDisorder:
    L_x = 4
    L_y = 5

    @pytest.fixture(autouse=True)
    def model(self):
        """An instance of a model with default."""
        model = RandomBondIsingModel2D(self.L_x, self.L_y)
        model.rng = np.random.default_rng(seed=0)
        model.init_spins(np.ones_like(model.spins, dtype=int))
        model.init_disorder(np.ones_like(model.disorder, dtype=int))
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
        assert energy_diff == 8.0

    def test_sample_changes_spins(self, model):
        n_sweeps = 10
        model.init_spins()
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

        new_model = RandomBondIsingModel2D(self.L_x, self.L_y)
        new_model.load_json(data_reread)

        new_stats = new_model.sample(1)

        # Check that output is exactly reproduced.
        assert old_stats['total'] == new_stats['total']
        assert old_stats['acceptance'] == new_stats['acceptance']
        assert np.all(model.spins == new_model.spins)

    def test_delta_energy_agrees_with_delta_energy(self, model):
        model.init_spins()
        for i_move in range(100):
            move = model.random_move()
            assert_flip_energies_consistent(
                model, move, message=f'Failed on move {i_move}'
            )


class TestRbim2DIidDisorder:
    L_x = 5
    L_y = 6

    def test_disorder_reproducible_seed(self):
        seed = 0
        spin_model_params = {'L_x': 5, 'L_y': 6}
        disorder_params = {'p': 0.4}

        # Generate some disorder configuration using the seed.
        rng = np.random.default_rng(seed)
        disorder_model = Rbim2DIidDisorder(rng)
        disorder_1 = disorder_model.generate(
            spin_model_params, disorder_params
        )

        # Do it again.
        rng = np.random.default_rng(seed)
        disorder_model = Rbim2DIidDisorder(rng)
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
        disorder_model = Rbim2DIidDisorder(rng)

        disorder_model = Rbim2DIidDisorder(rng)
        disorder = disorder_model.generate(
            spin_model_params, disorder_params
        )

        model = RandomBondIsingModel2D(self.L_x, self.L_y)
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
