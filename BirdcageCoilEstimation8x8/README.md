# Birdcage coil-estimation control

This isolated 8 x 8 control compares ESPIRiT maps with analytic eight-channel
`BirdcageCoilSens` data. It uses voxelwise unit-norm ESPIRiT maps, R=2 Cartesian
SENSE encoding, and the same nonnegative data-consistency objective for AD and
finite differences. It does not fit coil gains, image phase, or signal scale.

Run both methods from the repository project in Julia:

```julia
include("BirdcageCoilEstimation8x8/run_all.jl")
```

Each method writes `truth.png`, `reconstructed_density.png`, and `metrics.txt`
to its corresponding result directory.
