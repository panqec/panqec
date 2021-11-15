# bn3d

Biased noise for 3D quantum error-correcting codes

# Setup for development

Clone the repo and enter the repo
```
git clone https://github.com/ehua7365/bn3d.git
cd bn3d
```

Install Python3.8 if you haven't already.
```
sudo apt-get install python3.8
```

Install virtualenv if you haven't already
```
python3.8 -m pip install virtualenv
```

Create a virtual environment.
```
python3.8 -m virtualenv --python=/usr/bin/python3.8 venv
```

Activate the virtual environment
```
source venv/bin/activate
```

Install the dependencies for development mode.
```
pip install -r requirements/dev.txt
```

Copy the `env_template.txt` file and rename it to `.env`.
```
cp env_template.txt .env
```
Your `.env` file contains environmental varaibles that can be read in for to
customize.

Edit the .env file to change the `BN3D_DIR` path to a directory on
your storage device where you want the output files to be written.
If you don't do this, as a fallback, the data files will be written to the
`temp` directory in this repository.

Optionally, if you use dark theme in Jupyter Lab, setting
`BN3D_DARK_THEME=True` in the `.env` file will make the plots show up nicer.

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
        "model": "ToricCode3D",
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
bn3d run --file myinputs.json --trials 100
```

You may also have the option of running many different parameters.
```
{
  "comments": "Some comments about this run.",
  "ranges": {
    "label": "myrun",
    "code": {
      "model": "ToricCode3D",
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
bn3d slurm gen --n_trials 1000 --partition defq --cores 3 --time 10:00:00
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
bn3d slurm gennist deformed_rays --n_trials 1000 --split 100
```
This will split the job into 100 sbatch files, saved to the `slurm/sbatch`
directory, each of which run 1000 repeated trials.
To submit the jobs on slurm, simply run the generated `submit_deformed_rays.sh`
in the same directory as where the generated `sbatch` files are saved.

For more options see the help
```
bn3d slurm gennist --help
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
bn3d run --file /path/to/inputs.json --trials 1000 --start $START --n_runs $N_RUNS
```

The `--trials` flag is the number of Monte Carlo runs.
`$START` is the index to start.
`$N_RUNS` is hte number of indices to run.

# Statistical Mechanics Markov Chain Monte Carlo
This package now also includes classes for dealing with classical statistical
mechanics models which have a correspondence with quantum error correcting
codes on lattices.
They are implemented in an object-oriented manner.

The main annoyance with running things on clusters is that things get killed
and we have to start over again.
It takes a long time for MC chains to equilibrate.
Hence, there should be a way to save chains when then equilibrate and run them
later to get finer data.
Analysis should be aggregated at the end.

# Run the 3D GUI
To run the 3D GUI, after you have activated the environment, run
```
python3 gui/gui.py
```
Then open your browser and go to the link printed out in the command line.
Click on "3D Toric Code".

If you find the browser non-responsive check back on what error messages are
coming out of the command line.

# Run the 2D GUI
To run the 2D GUI, after you have activated the environment, run
```
python2 gui2d.py
```
This is not as well-developed as the 3D GUI.


# Reproducing results of paper

## XZZX rotated surface code with BP-OSD, equal aspect, odd sizes
To generate the input files in an inputs directory, run the following command
```
bn3d generate-input -i /path/to/inputdir \
        -l rotated -b planar -d xzzx -r equal \
        --decoder BeliefPropagationOSDDecoder --bias Z
```
The input `.json` files will be generated and saved to the given directory,
which you can check by opening some of them and inspecting them.
To understand what the above flags mean, you can always ask for help.
```
bn3d generate-input --help
```

Suppose you want to run one of these generated input files for 50 trials of
sampling and say it's named `xzzx-rotated-planar-bias-1.00-p-0.250.json`.
Then you can run.

```
bn3d run -f /path/to/inputdir/xzzx-rotated-planar-bias-1.00-p-0.250.json \
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
bn3d merge-dirs -o temp/paper/rot_bposd_xzzx_zbias/results
```
Doing so would create a directory `temp/paper/rot_bposd_xzzx_zbias/results`
that contains all the results.

You can then use the notebook `notebooks/14-eh-paper_figures.ipynb`
to analyse the results.

# Example of how to run statmech
In the `temp/statmech` directory,
create an empty directory, say `test_7` for example.
Then create a file called `targets.json` with the following contents
```
{
  "comments": "Just a single data point.",
  "ranges": [
    {
      "spin_model": "RandomBondIsingModel2D",
      "spin_model_params": [
        {"L_x": 6, "L_y": 6},
        {"L_x": 8, "L_y": 8},
        {"L_x": 10, "L_y": 10}
      ],
      "disorder_model": "Rbim2DIidDisorder",
      "disorder_params": [
        {"p": 0.02}
      ],
      "temperature": [
        1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
      ],
      "max_tau": 11,
      "n_disorder": 10
    }
  ]
}
```

This file specifies the inputs for this run.
Here, `max_tau=10` means each MCMC run will sample for `2^10` sweeps,
and `n_disorder=10` means 10 disorders will be generated and saved
into `temp/statmech/test_7`.

To generate some inputs, run
```
bn3d statmech generate temp/statmech/test_7
```

You will notice that many `.json` files have geen generated in
`temp/statmech/test_7/inputs`.
Each of these contains a set of disorders,
and is named with a hash of its contents.
Note that `max_tau` and `n_disorder` are not in these input files.
Instead, they are recorded in the `info.json` file.

Suppose you want to sample one of these inputs and it's called
`input_0123456890abcdef.json`.
To do so, run
```
bn3d statmech sample temp/statmech/test_7/input_0123456890abcdef.json
```

You will notice that results will be saved to the `results` directory,
one results `.json` file for each `tau`.
You may also notice the `models` directory saving the exact states at each tau.

To run all of them at once,
just use some bash script to run the above bash command for every input json
file.
On PBS or slurm,
you would write an sbatch script.
Checkout `scripts/statmech.sbatch` for an example
of how to parallelize to running across array jobs with many cores each.
If running locally,
you can use the command to run them all in parallel
```
bn3d statmech sample-parallel temp/statmech/test_7
```
You can adjust the options depending on how many cores you have.

To perform the analysis, you can use the command
```
bn3d statmech analyse temp/statmech/test_7
```
To plot the results, go to Jupyter lab and open `notebooks/13-eh-statmech_test_7.ipynb`.
Run the notebook and you should see the plots of the observables,
from which you should hopefully see phase transitions!
(Although something seems wrong because it's not crossing!)
 
Of course, that was rather noisy data with only 10 disorders.
You may want to increase the `n_disorder` to 100 or 1000 to get meaningful
plots.

# Running 2D Repetition Code Stat Mech
Use the following `temp/statmech/test_8/targets.json` file.
```
{
  "comments": "2D Repetition Code Stat Mech.",
  "ranges": [
    {
      "spin_model": "LoopModel2D",
      "spin_model_params": [
        {"L_x": 6, "L_y": 6},
        {"L_x": 8, "L_y": 8},
        {"L_x": 10, "L_y": 10}
      ],
      "disorder_model": "LoopModel2DIidDisorder",
      "disorder_params": [
        {"p": 0.10}
      ],
      "temperature": [
        1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8
      ],
      "max_tau": 11,
      "n_disorder": 10
    }
  ]
}
```

Then run the following commands to generate,
sample and analyse.

```
bn3d statmech generate temp/statmech/test_8
bn3d statmech sample-parallel temp/statmech/test_8
bn3d statmech analyse temp/statmech/test_8
```
The sampling takes about 10 minutes to run on a laptop with 8 cores.
