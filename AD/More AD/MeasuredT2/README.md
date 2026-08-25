# Approximate measured T2

This experiment reconstructs four measured fat-saturated spin-echo EPI acquisitions as
`ComplexF32` images and fits one mono-exponential T2 value at every 128 x 128 image node.

The measured echo times are 11.028, 38.468, 55.406, and 104.686 ms. The acquisitions use
different EPI shot/acceleration patterns, so the result is an approximate measured T2 map,
not a reference quantitative map. EPI distortion, T2-star blurring, motion, and the absence
of an independently acquired ground-truth map remain possible error sources.

Run from the repository root:

```sh
julia --project=. AD/MeasuredT2/run.jl
```

Outputs are written to `AD/MeasuredT2/128x128/`.
