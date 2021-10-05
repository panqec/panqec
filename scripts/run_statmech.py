import sys
import numpy as np
from tqdm import tqdm
from bn3d.statmech.rbim2d import Rbim2DIidDisorder
from bn3d.statmech.controllers import SimpleController
import psutil
from itertools import product


def generate_inputs(n_disorder):
    entries = []
    temperatures = np.arange(0.6, 2, 0.2).round(6).tolist()
    spin_model_params_list = [
        {
            'L_x': int(L),
            'L_y': int(L),
        }
        for L in [8, 10, 12, 16]
    ]
    disorder_params_list = [
        {
            'p': float(p),
        }
        for p in [0.02, 0.05]
    ]
    for spin_model_params, disorder_params in product(
        spin_model_params_list, disorder_params_list
    ):
        for i_disorder in range(n_disorder):
            rng = np.random.default_rng(seed=i_disorder)
            disorder_model = Rbim2DIidDisorder(rng=rng)
            disorder = disorder_model.generate(
                spin_model_params, disorder_params
            )
            for temperature in temperatures:
                entry = {
                    'spin_model': 'RandomBondIsingModel2D',
                    'temperature': temperature,
                    'spin_model_params': spin_model_params,
                    'disorder_model': 'Rbim2DIidDisorder',
                    'disorder': disorder.tolist(),
                    'disorder_params': disorder_params,
                }
                entries.append(entry)
    return entries


def main():
    print(psutil.cpu_percent(percpu=True))

    data_dir = sys.argv[1]
    n_disorder = 100
    max_tau = 10

    controller = SimpleController(data_dir)
    controller.init_models(generate_inputs(n_disorder))
    controller.run(max_tau, progress=tqdm)


if __name__ == '__main__':
    main()
