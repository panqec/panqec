def assert_flip_energies_consistent(model, move, message=''):
    initial_energy = model.total_energy()
    delta_energy = model.delta_energy(move)
    model.update(move)
    final_energy = model.total_energy()
    assert delta_energy == final_energy - initial_energy, message
