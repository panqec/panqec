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
