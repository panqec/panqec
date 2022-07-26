
![panqec-screenshot](https://user-images.githubusercontent.com/1157968/180657086-48ea0da0-6da4-4f9c-88e5-4a3f1233db40.png)

# PanQEC

PanQEC is a Python package that simplifies the simulation and visualization of quantum error correction codes.
In particular, it provides the following features:

* **Simple implementation of topological codes**, through lists of coordinates for qubits, stabilizers and logicals. Parity-check matrices and other properties of the code are automatically computed.
* End-to-end **sparse implementation of all the vectors and matrices**, optimized for fast and memory-efficient simulations.
* Library of high-level functions to evaluate the performance of a code with a given decoder and noise model. In particular, simulating the code and establishing its **threshold** and **subthreshold scaling** performance is made particularly easy.
* **2D and 3D visualization of topological codes** on the web browser, with a simple process to add your own codes on the visualization. This visualizer can also be found [online](https://gui.quantumcodes.io)
* Large collection of codes, error models and decoders.

In its current version, PanQEC implements the following codes

* **2D and 3D surface codes**, with periodic/open boundary, on both the rotated and traditional lattices.
* **Rhombic code**
* **Fractons codes**: X-Cube model and Haah's code

We also include an option to Hadamard-deform all those codes, i.e. applying a Hadamard gate on all qubits of a given axis. It includes the **XZZX version of the 2D and 3D surface codes**.

PanQEC also currently offers the following decoders:

* **BP-OSD** (Belief Propagation with Ordered Statistic Decoding), using the library [ldpc](https://github.com/quantumgizmos/ldpc) developed by Joschka Roffe. Works for all codes.
* **MBP** (Belief Propagation with Memory effect), as described in [this paper](https://arxiv.org/abs/2104.13659). Works for all codes.
* **MWPM** (Minimum-Weight Perfect Matching decoder) for 2D surface codes, using the library [PyMatching](https://pymatching.readthedocs.io) developed by Oscar Higgott.
* **SweepMatch** for 3D surface codes (using our implementation of the [sweep decoder](https://arxiv.org/abs/2004.07247) for loop-syndrome decoding and PyMatching for point-syndrome).

# Installation and documentation

PanQEC documentation is available at [panqec.readthedocs.io](https://panqec.readthedocs.io).

You can install it using `pip install panqec`.

# Start the GUI

To start the GUI, run
```
panqec start-gui
```
Then open your browser and go to the link printed out in the command line.

# Setup for development

Clone the repo and enter the repo
```
git clone https://github.com/ehua7365/panqec.git
cd panqec
```

Within a Python 3.8 environment, install the library and its dependencies for development mode.
```
pip install -r requirements/dev.txt
pip install -e .
```

Copy the `env_template.txt` file and rename it to `.env`.
```
cp env_template.txt .env
```
Your `.env` file contains environmental variables that can be read in for to
customize.

Edit the .env file to change the `PANQEC_DIR` path to a directory on
your storage device where you want the output files to be written.
If you don't do this, as a fallback, the data files will be written to the
`temp` directory in this repository.

Optionally, if you use dark theme in Jupyter Lab, setting
`PANQEC_DARK_THEME=True` in the `.env` file will make the plots show up nicer.

# Build the documentation

Within the root directory of the repository, run
```
sphinx-build -b html docs/ docs/_build/html
```
You can then read it by opening the file `docs/_build/html/index.html` on
your web browser.

Before you start running things on the cluster,
make sure you create this folder if it doesn't exist already.
```
mkdir -p temp/paper/share
```

# Run the tests

After you've activated your virtual environment and installed the dependences,
you can run the tests to make sure everything has been correctly installed.

```
pytest
```
If all the tests pass everything should be working as expected.

There are also some more comprehensive tests which are very slow to run.
To also run those, use
```
pytest --runslow
```

# Run the Demos
If everything is working, you can run the demo notebooks in the `demo`
directory which contains Jupyter notebooks demonstrating key milestones.
You can start Jupyter notebook using
```
jupyter lab
```

In your browser, Jupyter will open and you can navigate to the `demo` folder
and open some notebooks.
Run the notebook you want to run and see the results computed interactively.

# Code Style
Just follow PEP 8, but to actually test that the code is compliant, run the
linter using
```
make lint
```
It will run `flake8` and print out a list of non-compliances.
It will also run `mypy` to do type checking.

# Running simulations via the command line
After installing the package, to run a single simulation create an input
`.json`. Suppose you name it `myinputs.json` and have the following input
parameters in side it, for example,
```
{
  "comments": "Some comments about this run.",
  "runs": [
    {
      "label": "myrun",
      "code": {
        "model": "Toric3DCode",
        "parameters": {"L_x": 3, "L_y": 3, "L_z": 3}
      },
      "noise": {
        "model": "PauliErrorModel",
        "parameters": {"r_x": 1, "r_y": 0, "r_z": 0}
      },
      "decoder": {
        "model": "SweepMatchDecoder"
      },
      "probability": 0.1,
    }
  ]
}
```
You may have more than one run, but each runs needs a label, which is
the name of the subfolder the raw results will be saved to,
the code and its parameters, the noise model and its parameters and a decoder
(with parameters if needed) and a probability of error.
To run the file, just use

```
panqec run --file myinputs.json --trials 100
```

You may also have the option of running many different parameters.
```
{
  "comments": "Some comments about this run.",
  "ranges": {
    "label": "myrun",
    "code": {
      "model": "Toric3DCode",
      "parameters": [
        {"L_x": 3, "L_y": 3, "L_z": 3},
        {"L_x": 4, "L_y": 4, "L_z": 4},
        {"L_x": 5, "L_y": 5, "L_z": 5}
      ]
    },
    "noise": {
      "model": "PauliErrorModel",
      "parameters": [
        {"r_x": 1, "r_y": 0, "r_z": 0},
        {"r_x": 0.5, "r_y": 0, "r_z": 0.5},
        {"r_x": 0, "r_y": 0, "r_z": 1},
      ]
    },
    "decoder": {
      "model": "SweepMatchDecoder"
    },
    "probability": [0.1, 0.2, 0.3]
  }
}
```
You would run the file using simliar commands.

# Creating sbatch files for slurm
If you want to run jobs with slurm, you may use this repo to automatically
generate `.sbatch` files that slurm can run.

Suppose you have done some sample runs locally of `myinputs.json`, which
you can put in the `slurm/inputs/` directory.

To generate `.sbatch` files, run
```
panqec slurm gen --n_trials 1000 --partition defq --cores 3 --time 10:00:00
```
After running this command, `.sbatch` files will be created in the directory
`slurm/sbatch/`.
If you view them, you will see that they have the above information encoded
in them, and you can just run then using the `sbatch` command.

You can do the above generation on the cluster you are running it on so it
works nicely with the file system there.

# Generating sbatch files for NIST cluster
To generate sbatch files for NIST cluster for job `deformed_rays` split over 19
sbatch files.
```
panqec slurm gennist deformed_rays --n_trials 1000 --split 100
```
This will split the job into 100 sbatch files, saved to the `slurm/sbatch`
directory, each of which run 1000 repeated trials.
To submit the jobs on slurm, simply run the generated `submit_deformed_rays.sh`
in the same directory as where the generated `sbatch` files are saved.

For more options see the help
```
panqec slurm gennist --help
```
You can adjust the walltime, memory, etc.

# How to generate parameters to run
The space of Pauli operators is parameterized by a simplex (triangle).
See `notebooks/12-eh-generate_params.ipynb` for how to generate values
of `r_x`, `r_y` and `r_z` that parameterize things in terms of well-established
values of eta, as defined by Dave Tuckett.

# How to generate figures for the PSI essay
Take a look at `notebooks/11-eh-essay_figures.ipynb` to generate the figures,
but note that you will need to run the simulations first to have the data to
make the plots.

# How to make array jobs
Once you have a json input file, there is a unique list of parameters.
All you need to do is have something like this line in your sbatch file.

```
panqec run --file /path/to/inputs.json --trials 1000 --start $START --n_runs $N_RUNS
```

The `--trials` flag is the number of Monte Carlo runs.
`$START` is the index to start.
`$N_RUNS` is hte number of indices to run.

# Reproducing results of paper

## XZZX rotated surface code with BP-OSD, equal aspect, odd sizes
To generate the input files in an inputs directory, run the following command
```
panqec generate-input -i /path/to/inputdir \
        -l rotated -b planar -d xzzx -r equal \
        --decoder BeliefPropagationOSDDecoder --bias Z
```
The input `.json` files will be generated and saved to the given directory,
which you can check by opening some of them and inspecting them.
To understand what the above flags mean, you can always ask for help.
```
panqec generate-input --help
```

Suppose you want to run one of these generated input files for 50 trials of
sampling and say it's named `xzzx-rotated-planar-bias-1.00-p-0.250.json`.
Then you can run.

```
panqec run -f /path/to/inputdir/xzzx-rotated-planar-bias-1.00-p-0.250.json \
        -t 50 -o /path/to/outputdir
```
The above will run simulations for 50 trials each and save the results to the
given output path.
You can similarly ask for help for this command.

To get complete results, you will appreciate a cluster.
For large-scale simulations on clusters that use job schedulers such as slurm
or pbs, you may run many of these jobs together in parallel over many cores
and array jobs.
To see an example, see `scripts/pi_bposd.sbatch` for an example of how to run
on slurm and `scripts/qsub.job` for example of how to run on pbs.

If you are on a slurm cluster, the following might work.
First, generate the input files and sbatch files using the script.

```
bash scripts/paper_generate_inputs.sh
```

You will see that it will create lots of files and directories in `temp/paper`
which are all the input files.
For example, you might find lots of files in
```
ls temp/paper/rot_bposd_xzzx_zbias/inputs/
```
The generated sbatch files are all in `temp/paper/sbatch/`.
To submit all the jobs at once, just run
```
ls temp/paper/sbatch/*.sbatch | xargs -n1 sbatch
```
To make sure you're maximizing usage of your available resources on each node,
You can check the CPU and RAM usage live of your running jobs using
```
ls temp/paper/*/logs/usage_*.txt | xargs -n1 tail | sort
```
You can also print out the progress bars of all runnning simulations.
```
ls temp/paper/*/logs/1/*/stderr | xargs -d'\n' -n1 tail -n1
```
You can also see the logs from each node.
```
cat slurm/out/*
```
Once the runs are complete, or even when they are are almost complete,
you may want to merge the results directories of each directory into one
results folder.
You would need to do this to analyse the results if you look at a particular
directory and notice multiple results directories like the following example:
```
$ ls temp/paper/rot_bposd_xzzx_zbias
inputs  results_1  results_10  results_2  results_3  results_4  results_5
results_6  results_7  results_8  results_9
```
The reason there are so many was because the job was split up between multiple
cores in a certain way.
To create a merged results directory, just run,
just run
```
panqec merge-dirs -o temp/paper/rot_bposd_xzzx_zbias/results
```
Doing so would create a directory `temp/paper/rot_bposd_xzzx_zbias/results`
that contains all the results.

You can then use the notebook `notebooks/14-eh-paper_figures.ipynb`
to analyse the results.

# The team

PanQEC is currently developed and maintained by [Eric Huang](https://github.com/ehua7365) and [Arthur Pesah](http://arthurpesah.me/). The implementation of 3D toric and fracton codes was done under the supervision of Arpit Dua, Michael Vasmer and Christopher Chubb.

Note: PanQEC was greatly inspired by [qecsim](https://qecsim.github.io/) at its inception, and we would like to thank its author David Tuckett for providing us with the first seed of this project.
