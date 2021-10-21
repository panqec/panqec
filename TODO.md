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
- [x] investigate why thresholds

### Finite-size scaling
- [x] implement it

### Stat mech
- [ ] `SpinModel` class
    - [ ] `energy_diff`
- [ ] `Observable` class
- [ ] `MCSampler` class
- [ ] Convergence test - 3 error bars per tau window


### Paper
- [ ] AP Numerics What are rough estimates for thresholds with BP+OSD?
- [ ] EH Numerics Generate inputs and commands for AD to run script cluster
- [ ] AP Table 1 check size formulae correct
- [ ] EH Fig change colours and labels back to CSS
- [ ] EH Fig Add rotated coordinate axes for 3D and 2D
- [ ] EH Fig Shaded transparent circle on qubit to show deformation
- [ ] AP Plots Thresholds estimates and thresholds from data?
- [ ] AP Plots Subtreshold scaling plots 100,000 trials, fix p vary L log scale
- [ ] EH Numerics 100 values of p in 0.005 increments too much, only need a 16
  data points near estimated threshold.
- [ ] AP fill in Table 2 thresholds
- [ ] EH Fig Show 2 layers only (low priority)
- [x] Go beyond 0.5 error rate
- [x] EH subthreshold simulations bias 10, 100, 1000, error rate 5% fixed
