# bn3d todo list

### Integration with qecsim
- [x] Add biased-weight decoder to CLI
- [x] Add biased sweep decoder to CLI
- [x] Update demo notebook to use new API

### CLI
- [x] Fix CLI to deal with noise-dependent decoders
- [x] Get demo notebook working with new API
- [ ] Get tests of CLI run command working

### Slurm
- [x] Get simulations running on slurm

### Sweep decoder
- [ ] investigate why thresholds

### Finite-size scaling
- [x] implement it
- [ ] Fit $D L^(1/\mu)$ term for finite-size scaling
- [x] Minimal example of how to use

### Stat mech
- [x] `SpinModel` class
    - [x] `energy_diff`
- [x] `Observable` class
- [x] `Controller` class
- [x] Convergence test - 3 error bars per tau window
- [x] Be able to save and load existing MC chains
- [x] Fix slowness by avoiding reading all files
- [x] Compute time estimation
- [ ] Parallelize running for Controller
- [ ] Wilson loop observable
- [x] Error bars for correlation length
- [ ] CLI command to generate and run
- [ ] zip compression of json files managed by DataManager
- [x] Separate results viewer notebook/web app
- [ ] Z2GaugeModel2D
- [ ] Z2GaugeModel3D
- [ ] Julia wrapper
- [ ] Optional C++ extension
- [ ] Script for array jobs on slurm
- [ ] Figures in inkscape
- [ ] p = 0 case
- [ ] p > 0 case
- [ ] finite bias case

### Fault-tolerant thresholds 3D
-[ ] Deformed Y noise fault-tolerant threshold 3D
