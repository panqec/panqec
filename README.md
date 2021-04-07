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

Then edit the .env file to change the `BN3D_DIR` path to a directory on
your storage device where you want the output files to be written.
If you don't do this, as a fallback, the data files will be written to the
`temp` directory in this repository.

# Run the tests
After you've activated your virtual environment and installed the dependences,
you can run the tests to make sure everything has been correctly installed.

```
pytest
```
If all the tests pass everything should be working as expected.

# Code Style
Just follow PEP 8, but to actually test that the code is compliant, run the
linter using
```
make lint
```
It will run `flake8` and print out a list of non-compliances.
