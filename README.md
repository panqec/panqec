
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
* **SweepMatch** for 3D surface codes (using our implementation of the [sweep decoder](https://arxiv.org/abs/2004.07247) for loop-syndrome decoding and PyMatching for point-syndrome decoding).

# Installation and documentation

PanQEC documentation is available at [panqec.readthedocs.io](https://panqec.readthedocs.io).

You can install it using `pip install panqec`.

# Start the GUI

To start the GUI, run
```
panqec start-gui
```
Then open your browser and go to the link printed out in the command line.

# Contributing
PRs from the community are very welcome!
Check out the [development](https://panqec.readthedocs.io) section of the
documentation for instructions of how to set up the development environment.

# Paper
An example of a paper which uses PanQEC is [here](https://arxiv.org/abs/2211.02116).


# The team

PanQEC is currently developed and maintained by [Eric Huang](https://github.com/ehua7365) and [Arthur Pesah](http://arthurpesah.me/). The implementation of 3D toric and fracton codes was done under the supervision of Arpit Dua, Michael Vasmer and Christopher Chubb.

Note: PanQEC was greatly inspired by [qecsim](https://qecsim.github.io/) at its inception, and we would like to thank its author David Tuckett for providing us with the first seed of this project.
