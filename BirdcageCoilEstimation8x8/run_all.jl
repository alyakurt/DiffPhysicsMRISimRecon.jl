include("common.jl")
using .BirdcageCoilEstimation8x8

problem = BirdcageCoilEstimation8x8.build_problem()
ad_result = BirdcageCoilEstimation8x8.run_reconstruction(:AD, problem)
finite_difference_result = BirdcageCoilEstimation8x8.run_reconstruction(:FiniteDiff, problem)
