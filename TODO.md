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

### Stat mech
- [x] `SpinModel` class
    - [x] `energy_diff`
- [x] `Observable` class
- [x] `Controller` class
- [x] Convergence test - 3 error bars per tau window
- [ ] Be able to save and load existing MC chains
