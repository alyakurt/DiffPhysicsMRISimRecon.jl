# T1 reconstruction on the image-reconstruction nodes

The fixed proton-density array is the exact 128 x 128 complex node vector produced by the measured accelerated image reconstruction:

`AD/PlainEspiritComplexAD/AcceleratedR2_128x128/reconstructed_image.cf32`

The experiment generates eight accelerated `ComplexF32` inversion-recovery signals from those nodes and recovers one real T1 value per node.

```sh
julia --project=. AD/T1/run.jl
```
