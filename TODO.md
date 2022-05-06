# panqec todo list

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
- [x] investigate why thresholds

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
- [x] Parallelize running for Controller (distribute tasks among jobs and cores)
- [ ] Wilson loop observable
- [x] Error bars for correlation length
- [x] CLI command to generate and run
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

### Paper
- [x] AP Numerics What are rough estimates for thresholds with BP+OSD?
- [x] EH Numerics Generate inputs and commands for AD to run script cluster
- [x] AP Table 1 check size formulae correct
- [x] EH Fig change colours and labels back to CSS
- [x] EH Fig Add rotated coordinate axes for 3D and 2D
- [ ] EH Fig Shaded transparent circle on qubit to show deformation
- [ ] AP Plots Thresholds estimates and thresholds from data?
- [ ] AP Plots Subtreshold scaling plots 100,000 trials, fix p vary L log scale
- [x] EH Numerics 100 values of p in 0.005 increments too much, only need a 16
  data points near estimated threshold.
- [ ] AP fill in Table 2 thresholds
- [ ] EH Fig Show 2 layers only (low priority)

### Post-QIP
- [ ] Deformed Y noise fault-tolerant threshold 3D
- [x] Go beyond 0.5 error rate
- [x] EH subthreshold simulations bias 10, 100, 1000, error rate 5% fixed
