
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

## Install PanQEC from the repository

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

## Run the tests

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

## Check the code Style

Just follow PEP 8, but to actually test that the code is compliant, run the
linter using
```
make lint
```
It will run `flake8` and print out a list of non-compliances.
It will also run `mypy` to do type checking.

## Build the documentation

Within the root directory of the repository, run
```
sphinx-build -b html docs/ docs/_build/html
```
You can then read it by opening the file `docs/_build/html/index.html` on
your web browser.

# The team

PanQEC is currently developed and maintained by [Eric Huang](https://github.com/ehua7365) and [Arthur Pesah](http://arthurpesah.me/). The implementation of 3D toric and fracton codes was done under the supervision of Arpit Dua, Michael Vasmer and Christopher Chubb.

Note: PanQEC was greatly inspired by [qecsim](https://qecsim.github.io/) at its inception, and we would like to thank its author David Tuckett for providing us with the first seed of this project.
