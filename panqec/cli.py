import os
from typing import Optional, List, Dict, Tuple
import click
import panqec
from tqdm import tqdm
import numpy as np
import json
from json.decoder import JSONDecodeError
from .app import run_file, merge_results_dicts
from .config import CODES, ERROR_MODELS, DECODERS, PANQEC_DIR, BASE_DIR
from .slurm import (
    generate_sbatch, get_status, generate_sbatch_nist, count_input_runs,
    clear_out_folder, clear_sbatch_folder
)
from .statmech.cli import statmech
from .utils import get_direction_from_bias_ratio
from panqec.gui import GUI
from glob import glob


@click.group(invoke_without_command=True)
@click.version_option(version=panqec.__version__, prog_name='panqec')
@click.pass_context
def cli(ctx):
    """
    panqec - biased noise in 3D simulations.

    See panqec COMMAND --help for command-specific help.
    """
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


@click.command()
@click.option('-p', '--port', 'port')
def start_gui(port: Optional[int]):
    gui = GUI()
    gui.run(port=port)


@click.command()
@click.pass_context
@click.option('-f', '--file', 'file_')
@click.option('-t', '--trials', default=100, type=click.INT, show_default=True)
@click.option('-s', '--start', default=None, type=click.INT, show_default=True)
@click.option(
    '-o', '--output_dir', default=PANQEC_DIR, type=click.STRING,
    show_default=True
)
@click.option(
    '-n', '--n_runs', default=None, type=click.INT, show_default=True
)
def run(
    ctx,
    file_: Optional[str],
    trials: int,
    start: Optional[int],
    n_runs: Optional[int],
    output_dir: Optional[str]
):
    """Run a single job or run many jobs from input file."""
    if file_ is not None:
        run_file(
            os.path.abspath(file_), trials,
            start=start, n_runs=n_runs, progress=tqdm,
            output_dir=output_dir
        )
    else:
        print(ctx.get_help())


@click.command()
@click.argument('model_type', required=False, type=click.Choice(
    ['codes', 'noise', 'decoders'],
    case_sensitive=False
))
def ls(model_type=None):
    """List available codes, noise models and decoders."""
    if model_type is None or model_type == 'codes':
        print('Codes:')
        print('\n'.join([
            '    ' + name for name in sorted(CODES.keys())
        ]))
    if model_type is None or model_type == 'noise':
        print('Error Models (Noise):')
        print('\n'.join([
            '    ' + name for name in sorted(ERROR_MODELS.keys())
        ]))
    if model_type is None or model_type == 'decoders':
        print('Decoders:')
        print('\n'.join([
            '    ' + name for name in sorted(DECODERS.keys())
        ]))


def read_bias_ratios(eta_string: str) -> list:
    """Read bias ratios from comma separated string."""
    bias_ratios = []
    for s in eta_string.split(','):
        s = s.strip()
        if s == 'inf':
            bias_ratios.append(np.inf)
        elif float(s) % 1 == 0:
            bias_ratios.append(int(s))
        else:
            bias_ratios.append(float(s))
    return bias_ratios


def read_range_input(specification: str) -> List[float]:
    """Read range input string and return list."""
    values: List[float] = []
    if ':' in specification:
        parts = specification.split(':')
        min_value = float(parts[0])
        max_value = float(parts[1])
        step = 0.005
        if len(parts) == 3:
            step = float(parts[2])
        values = np.arange(min_value, max_value + step, step).tolist()
    elif ',' in specification:
        values = [float(s) for s in specification.split(',')]
    else:
        values = [float(specification)]
    return values


@click.command()
@click.option(
    '-i', '--input_dir', required=True, type=str,
    help='Directory to save input .json files'
)
@click.option(
    '-l', '--lattice', default='kitaev',
    show_default=True,
    type=click.Choice(['rotated', 'kitaev']),
    help='Lattice rotation'
)
@click.option(
    '-b', '--boundary', default='toric',
    show_default=True,
    type=click.Choice(['toric', 'planar']),
    help='Boundary conditions'
)
@click.option(
    '-d', '--deformation', default='none',
    show_default=True,
    type=click.Choice(['none', 'xzzx', 'xy']),
    help='Deformation'
)
@click.option(
    '-r', '--ratio', default='equal', type=click.Choice(['equal', 'coprime']),
    show_default=True, help='Lattice aspect ratio spec'
)
@click.option(
    '--decoder', default='BeliefPropagationOSDDecoder',
    show_default=True,
    type=click.Choice(DECODERS.keys()),
    help='Decoder name'
)
@click.option(
    '-s', '--sizes', default='5,9,7,13', type=str,
    show_default=True,
    help='List of sizes'
)
@click.option(
    '--bias', default='Z', type=click.Choice(['X', 'Y', 'Z']),
    show_default=True,
    help='Pauli bias'
)
@click.option(
    '--eta', default='0.5,1,3,10,30,100,inf', type=str,
    show_default=True,
    help='Bias ratio'
)
@click.option(
    '--prob', default='0:0.6:0.005', type=str,
    show_default=True,
    help='min:max:step or single value or list of values'
)
@click.option(
    '--code_class', default=None, type=str,
    show_default=True,
    help='Explicitly specify the code class, e.g. Toric3DCode'
)
@click.option(
    '--noise_class', default=None, type=str,
    show_default=True,
    help='Explicitly specify the noise class, e.g. DeformedXZZXErrorModel'
)
def generate_input(
    input_dir, lattice, boundary, deformation, ratio, sizes, decoder, bias,
    eta, prob, code_class, noise_class
):
    """Generate the json files of every experiment.

    \b
    Example:
    panqec generate-input -i /path/to/inputdir \\
            -l rotated -b planar -d xzzx -r equal \\
            -s 2,4,6,8 --decoder BeliefPropagationOSDDecoder \\
            --bias Z --eta '10,100,1000,inf' \\
            --prob 0:0.5:0.005
    """
    if lattice == 'kitaev' and boundary == 'planar':
        raise NotImplementedError("Kitaev planar lattice not implemented")

    os.makedirs(input_dir, exist_ok=True)

    delta = 0.005
    probabilities = np.arange(0, 0.5+delta, delta).tolist()
    probabilities = read_range_input(prob)
    bias_ratios = read_bias_ratios(eta)

    for eta in bias_ratios:
        direction = get_direction_from_bias_ratio(bias, eta)
        for p in probabilities:
            label = "regular" if deformation == "none" else deformation
            label += f"-{lattice}"
            label += f"-{boundary}"
            if eta == np.inf:
                label += "-bias-inf"
            else:
                label += f"-bias-{eta:.2f}"
            label += f"-p-{p:.3f}"

            code_model = ''
            if lattice == 'rotated':
                code_model += 'Rotated'
            if boundary == 'toric':
                code_model += 'Toric'
            else:
                code_model += 'Planar'
            code_model += '3DCode'

            # Explicit override.
            if code_class is not None:
                code_model = code_class

            L_list = [int(s) for s in sizes.split(',')]
            if ratio == 'coprime':
                code_parameters = [
                    {"L_x": L, "L_y": L + 1, "L_z": L}
                    for L in L_list
                ]
            else:
                if code_model == 'RotatedPlanar3DCode':
                    code_parameters = [
                        {"L_x": L, "L_y": L, "L_z": L}
                        for L in L_list
                    ]
                else:
                    code_parameters = [
                        {"L_x": L, "L_y": L, "L_z": L}
                        for L in L_list
                    ]
            code_dict = {
                "model": code_model,
                "parameters": code_parameters
            }

            if deformation == "none":
                noise_model = "PauliErrorModel"
            elif deformation == "xzzx":
                noise_model = 'DeformedXZZXErrorModel'
            elif deformation == "xy":
                noise_model = 'DeformedXYErrorModel'

            # Explicit override option for noise model.
            if noise_class is not None:
                noise_model = noise_class

            noise_parameters = direction
            noise_dict = {
                "model": noise_model,
                "parameters": noise_parameters
            }

            if decoder == "BeliefPropagationOSDDecoder":
                decoder_model = "BeliefPropagationOSDDecoder"
                decoder_parameters = {'max_bp_iter': 1000,
                                      'osd_order': 10}
            else:
                decoder_model = decoder
                decoder_parameters = {}

            decoder_dict = {"model": decoder_model,
                            "parameters": decoder_parameters}

            ranges_dict = {"label": label,
                           "code": code_dict,
                           "noise": noise_dict,
                           "decoder": decoder_dict,
                           "probability": [p]}

            json_dict = {"comments": "",
                         "ranges": ranges_dict}

            filename = os.path.join(input_dir, f'{label}.json')
            with open(filename, 'w') as json_file:
                json.dump(json_dict, json_file, indent=4)


@click.group(invoke_without_command=True)
@click.pass_context
def slurm(ctx):
    """Routines for generating and running slurm scripts."""
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


@click.command()
@click.option('-o', '--outdir', required=True, type=str, nargs=1)
@click.argument('dirs', type=click.Path(exists=True), nargs=-1)
def merge_dirs(outdir, dirs):
    """Merge result directories that had been split into outdir."""
    os.makedirs(outdir, exist_ok=True)

    if len(dirs) == 0:
        results_dirs = glob(os.path.join(os.path.dirname(outdir), 'results_*'))
        results_dirs = [path for path in results_dirs if os.path.isdir(path)]
    else:
        results_dirs = list(dirs)

    print(f'Merging {len(results_dirs)} dirs into {outdir}')
    file_lists: Dict[Tuple[str, str], List[str]] = dict()
    for sep_dir in results_dirs:
        for sub_dir in os.listdir(sep_dir):
            for file_path in glob(os.path.join(sep_dir, sub_dir, '*.json')):
                base_name = os.path.basename(file_path)
                key = (sub_dir, base_name)
                if key not in file_lists:
                    file_lists[key] = []
                file_lists[key].append(file_path)
    print(len(file_lists))

    iterator = tqdm(file_lists.items(), total=len(file_lists))
    for (sub_dir, base_name), file_list in iterator:
        os.makedirs(os.path.join(outdir, sub_dir), exist_ok=True)
        combined_file = os.path.join(outdir, sub_dir, base_name)

        results_dicts = []
        for file_path in file_list:
            try:
                with open(file_path) as f:
                    results_dicts.append(json.load(f))
            except JSONDecodeError:
                print(f'Error reading {file_path}, skipping')

        combined_results = merge_results_dicts(results_dicts)

        with open(combined_file, 'w') as f:
            json.dump(combined_results, f)


@click.command()
@click.argument('sbatch_file', required=True)
@click.option('-d', '--data_dir', type=click.Path(exists=True), required=True)
@click.option('-n', '--n_array', default=6, type=click.INT, show_default=True)
@click.option('-q', '--queue', default='defq', type=str, show_default=True)
@click.option(
    '-w', '--wall_time', default='0-20:00', type=str, show_default=True
)
@click.option(
    '-t', '--trials', default='0-20:00', type=str, show_default=True
)
@click.option(
    '-s', '--split', default=1, type=click.INT, show_default=True
)
def pi_sbatch(sbatch_file, data_dir, n_array, queue, wall_time, trials, split):
    """Generate PI-style sbatch file with parallel and array job."""
    template_file = os.path.join(
        os.path.dirname(BASE_DIR), 'scripts', 'pi_template.sh'
    )
    with open(template_file) as f:
        text = f.read()

    inputs_dir = os.path.join(data_dir, 'inputs')
    assert os.path.isdir(inputs_dir), (
        f'{inputs_dir} missing, please create it and generate inputs'
    )
    name = os.path.basename(data_dir)
    replace_map = {
        '${TRIALS}': trials,
        '${DATADIR}': data_dir,
        '${TIME}': wall_time,
        '${NAME}': name,
        '${NARRAY}': str(n_array),
        '${QUEUE}': queue,
        '${SPLIT}': str(split),
    }
    for template_string, value in replace_map.items():
        text = text.replace(template_string, value)

    with open(sbatch_file, 'w') as f:
        f.write(text)
    print(f'Wrote to {sbatch_file}')


@click.command()
@click.argument('sbatch_file', required=True)
@click.option('-d', '--data_dir', type=click.Path(exists=True), required=True)
@click.option('-n', '--n_array', default=6, type=click.INT, show_default=True)
@click.option(
    '-a', '--account', default='def-raymond', type=str, show_default=True
)
@click.option(
    '-e', '--email', default='mvasmer@pitp.ca', type=str, show_default=True
)
@click.option(
    '-w', '--wall_time', default='04:00:00', type=str, show_default=True
)
@click.option(
    '-m', '--memory', default='16GB', type=str, show_default=True
)
@click.option(
    '-t', '--trials', default=1000, type=click.INT, show_default=True
)
@click.option(
    '-s', '--split', default=1, type=click.INT, show_default=True
)
def cc_sbatch(
    sbatch_file, data_dir, n_array, account, email, wall_time, memory, trials,
    split
):
    """Generate Compute Canada-style sbatch file with parallel array jobs."""
    template_file = os.path.join(
        os.path.dirname(BASE_DIR), 'scripts', 'cc_template.sh'
    )
    with open(template_file) as f:
        text = f.read()

    inputs_dir = os.path.join(data_dir, 'inputs')
    assert os.path.isdir(inputs_dir), (
        f'{inputs_dir} missing, please create it and generate inputs'
    )
    name = os.path.basename(data_dir)
    replace_map = {
        '${ACCOUNT}': account,
        '${EMAIL}': email,
        '${TIME}': wall_time,
        '${MEMORY}': memory,
        '${NAME}': name,
        '${NARRAY}': str(n_array),
        '${DATADIR}': os.path.abspath(data_dir),
        '${TRIALS}': str(trials),
        '${SPLIT}': str(split),
    }
    for template_string, value in replace_map.items():
        text = text.replace(template_string, value)

    with open(sbatch_file, 'w') as f:
        f.write(text)
    print(f'Wrote to {sbatch_file}')


@click.command()
@click.argument('sbatch_file', required=True)
@click.option('-d', '--data_dir', type=click.Path(exists=True), required=True)
@click.option('-n', '--n_array', default=6, type=click.INT, show_default=True)
@click.option(
    '-w', '--wall_time', default='0-23:00', type=str, show_default=True
)
@click.option(
    '-m', '--memory', default='32GB', type=str, show_default=True
)
@click.option(
    '-t', '--trials', default=1000, type=click.INT, show_default=True
)
@click.option(
    '-s', '--split', default=1, type=click.INT, show_default=True
)
@click.option('-p', '--partition', default='pml', type=str, show_default=True)
def nist_sbatch(
    sbatch_file, data_dir, n_array, wall_time, memory, trials, split, partition
):
    """Generate NIST-style sbatch file with parallel array jobs."""
    template_file = os.path.join(
        os.path.dirname(BASE_DIR), 'scripts', 'nist_template.sh'
    )
    with open(template_file) as f:
        text = f.read()

    inputs_dir = os.path.join(data_dir, 'inputs')
    assert os.path.isdir(inputs_dir), (
        f'{inputs_dir} missing, please create it and generate inputs'
    )
    name = os.path.basename(data_dir)
    replace_map = {
        '${TIME}': wall_time,
        '${MEMORY}': memory,
        '${NAME}': name,
        '${NARRAY}': str(n_array),
        '${DATADIR}': os.path.abspath(data_dir),
        '${TRIALS}': str(trials),
        '${SPLIT}': str(split),
        '${QUEUE}': partition,
    }
    for template_string, value in replace_map.items():
        text = text.replace(template_string, value)

    with open(sbatch_file, 'w') as f:
        f.write(text)
    print(f'Wrote to {sbatch_file}')


@click.command()
@click.argument('qsub_file', required=True)
@click.option('-d', '--data_dir', type=click.Path(exists=True), required=True)
@click.option('-n', '--n_array', default=6, type=click.INT, show_default=True)
@click.option(
    '-w', '--wall_time', default='0-23:00', type=str, show_default=True
)
@click.option(
    '-m', '--memory', default='32GB', type=str, show_default=True
)
@click.option(
    '-t', '--trials', default=1000, type=click.INT, show_default=True
)
@click.option(
    '-s', '--split', default=1, type=click.INT, show_default=True
)
@click.option('-p', '--partition', default='pml', type=str, show_default=True)
def generate_qsub(
    qsub_file, data_dir, n_array, wall_time, memory, trials, split, partition
):
    """Generate qsub (PBS) file with parallel array jobs."""
    template_file = os.path.join(
        os.path.dirname(BASE_DIR), 'scripts', 'qsub_template.sh'
    )
    with open(template_file) as f:
        text = f.read()

    inputs_dir = os.path.join(data_dir, 'inputs')
    assert os.path.isdir(inputs_dir), (
        f'{inputs_dir} missing, please create it and generate inputs'
    )
    name = os.path.basename(data_dir)
    replace_map = {
        '${TIME}': wall_time,
        '${MEMORY}': memory,
        '${NAME}': name,
        '${NARRAY}': str(n_array),
        '${DATADIR}': os.path.abspath(data_dir),
        '${TRIALS}': str(trials),
        '${SPLIT}': str(split),
        '${QUEUE}': partition,
    }
    for template_string, value in replace_map.items():
        text = text.replace(template_string, value)

    with open(qsub_file, 'w') as f:
        f.write(text)
    print(f'Wrote to {qsub_file}')


@click.command()
@click.option('--n_trials', default=1000, type=click.INT, show_default=True)
@click.option('--partition', default='defq', show_default=True)
@click.option('--time', default='10:00:00', show_default=True)
@click.option('--cores', default=1, type=click.INT, show_default=True)
def gen(n_trials, partition, time, cores):
    """Generate sbatch files."""
    generate_sbatch(n_trials, partition, time, cores)


@click.command()
@click.argument('name', required=True)
@click.option('--n_trials', default=1000, type=click.INT, show_default=True)
@click.option('--nodes', default=1, type=click.INT, show_default=True)
@click.option('--ntasks', default=1, type=click.INT, show_default=True)
@click.option('--cpus_per_task', default=40, type=click.INT, show_default=True)
@click.option('--mem', default=10000, type=click.INT, show_default=True)
@click.option('--time', default='10:00:00', show_default=True)
@click.option('--split', default=1, type=click.INT, show_default=True)
@click.option('--partition', default='pml', show_default=True)
@click.option(
    '--cluster', default='nist', show_default=True,
    type=click.Choice(['nist', 'symmetry'])
)
def gennist(
    name, n_trials, nodes, ntasks, cpus_per_task, mem, time, split, partition,
    cluster
):
    """Generate sbatch files for NIST cluster."""
    generate_sbatch_nist(
        name, n_trials, nodes, ntasks, cpus_per_task, mem, time, split,
        partition, cluster
    )


@click.command()
@click.argument('folder', required=True, type=click.Choice(
    ['all', 'out', 'sbatch'],
    case_sensitive=False
))
def clear(folder):
    """Clear generated files."""
    if folder == 'out' or folder == 'all':
        clear_out_folder()
    if folder == 'sbatch' or folder == 'all':
        clear_sbatch_folder()


@click.command()
@click.argument('name', required=True)
def count(name):
    """Count number of input parameters contained."""
    n_runs = count_input_runs(name)
    print(n_runs)


@click.command()
def status():
    """Show the status of running jobs."""
    get_status()


slurm.add_command(gen)
slurm.add_command(gennist)
slurm.add_command(status)
slurm.add_command(count)
slurm.add_command(clear)
cli.add_command(start_gui)
cli.add_command(run)
cli.add_command(ls)
cli.add_command(slurm)
cli.add_command(generate_input)
cli.add_command(statmech)
cli.add_command(pi_sbatch)
cli.add_command(cc_sbatch)
cli.add_command(merge_dirs)
cli.add_command(nist_sbatch)
cli.add_command(generate_qsub)
