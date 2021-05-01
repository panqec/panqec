# bn3d

Biased noise for 3D quantum error-correcting codes

# TODO
- Add biased-weight decoder to CLI
- Add biased sweep decoder to CLI
- Fix CLI to deal with noise-dependent decoders
- Get demo notebook working with new API
- Get tests of CLI run command working
- Get simulations running on slurm

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

# Running a simulation via the command line
After installing the package, to run a single simulation on the command
line, use
```
bn3d run --code 'ToricCode3D(3, 3, 3)' \
        --noise 'PauliErrorModel(1, 0, 0)' \
        --decoder 'PyMatchingSweepDecoder3D()'
        --probability 0.1
        --trials 10
```

Alternatively, if you find that too verbose, you can specify the job you
want to run in a `.json` file and run the following.
```
bn3d run -f myinputs.json
```

An example of what to put in `myinputs.json` would be the following.
```
{
  "comments": "Some comments about this run.",
  "runs": [
    {
      "label": "myrun",
      "code": {
        "model": "ToricCode3D",
        "parameters": [3, 3, 3]
      },
      "noise": {
        "model": "PauliErrorModel",
        "parameters": [1, 0, 0]
      },
      "decoder": {
        "model": "SweepMatchDecoder"
      },
      "probability": 0.1,
      "trials": 10
    }
  ]
}
```
You can have more runs with different parameters as you see fit.

You may also have the option of running many different parameters.
```
{
  "comments": "Some comments about this run.",
  "ranges": {
    "label": "myrun",
    "code": {
      "model": "ToricCode3D",
      "parameters": [
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5]
      ]
    },
    "noise": {
      "model": "PauliErrorModel",
      "parameters": [
        [1, 0, 0],
        [0.5, 0, 0.5],
        [0, 0, 1]
      ]
    },
    "decoder": {
      "model": "SweepMatchDecoder"
    },
    "probability": [0.1, 0.2, 0.3],
    "trials": 10
  }
}
```

# Code Style
Just follow PEP 8, but to actually test that the code is compliant, run the
linter using
```
make lint
```
It will run `flake8` and print out a list of non-compliances.
It will also run `mypy` to do type checking.
