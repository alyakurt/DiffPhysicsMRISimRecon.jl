using Optim

include(joinpath(@__DIR__, "..", "reconstruct.jl"))
include(joinpath(@__DIR__, "..", "experiment_plots.jl"))

function run_lbfgs_reconstruction(voxel_grid; iterations=ITERATIONS)
    resolution = first(voxel_grid)
    output_directory = joinpath(@__DIR__, "$(resolution)x$(resolution)")
    model = build_problem(voxel_grid[1:2], output_directory; accelerated=true)
    controls = (; voxel_grid=voxel_grid[1:2], output_directory)

    device_model = map(Reactant.to_rarray, model)
    initial_density = zeros(Float32, prod(controls.voxel_grid))
    device_density = Reactant.to_rarray(initial_density)
    compiled_gradient = Reactant.@compile sync=true objective_gradient(device_density, device_model)
    initial_loss, initial_gradient = compiled_gradient(device_density, device_model)
    analytic_gradient = -2f0 .* sense_adjoint(model.data, model)
    gradient_error = norm(Array(initial_gradient) - analytic_gradient) / norm(analytic_gradient)
    gradient_error < 2f-3 || error("AD gradient check failed: $gradient_error")

    cached_density = fill(Float32(NaN), length(initial_density))
    cached_gradient = similar(initial_density)
    cached_loss = Ref(Float32(NaN))
    function evaluate!(density)
        density == cached_density && return cached_loss[]
        loss, gradient = compiled_gradient(Reactant.to_rarray(density), device_model)
        copyto!(cached_density, density)
        copyto!(cached_gradient, Array(gradient))
        cached_loss[] = Reactant.to_number(loss)
    end
    objective_host(density) = evaluate!(density)
    function gradient_host!(gradient, density)
        evaluate!(density)
        copyto!(gradient, cached_gradient)
        nothing
    end

    losses = Float64[]
    save_density(initial_density, "iteration_00.png", "L-BFGS-B iteration 0", controls)
    function record_iteration(state)
        loss = Float64(objective_host(state.x))
        push!(losses, loss)
        save_density(
            state.x,
            "iteration_$(lpad(state.iteration, 2, '0')).png",
            "L-BFGS-B iteration $(state.iteration)",
            controls,
        )
        println("L-BFGS-B $(resolution)x$(resolution) iteration $(state.iteration): loss = $loss")
        false
    end

    options = Optim.Options(
        iterations=iterations,
        callback=record_iteration,
        f_abstol=0,
        f_reltol=0,
        x_abstol=0,
        x_reltol=0,
    )
    lower = zeros(Float32, length(initial_density))
    upper = fill(Float32(Inf), length(initial_density))
    result = Optim.optimize(
        objective_host,
        gradient_host!,
        lower,
        upper,
        initial_density,
        Optim.LBFGSB(m=10),
        options,
    )

    density = Optim.minimizer(result)
    final_loss = Float64(Optim.minimum(result))
    isempty(losses) || last(losses) == final_loss || push!(losses, final_loss)
    data_norm = sum(abs2, model.data)
    save_density(
        density,
        "reconstructed_density.png",
        "L-BFGS-B $(resolution)x$(resolution) reconstructed brain density",
        controls,
    )
    save_loss_curve(
        losses,
        joinpath(output_directory, "loss_convergence.png"),
        "L-BFGS-B convergence, $(resolution)x$(resolution)",
    )
    loss_table = join(
        ["iteration,loss"; ["$iteration,$loss" for (iteration, loss) in zip(0:(length(losses) - 1), losses)]],
        '\n',
    )
    write(joinpath(output_directory, "loss.csv"), loss_table * "\n")

    metrics = """
    method = L-BFGS-B with reverse-mode AD gradient
    resolution = $(resolution)x$(resolution)
    iterations = $(Optim.iterations(result))
    converged = $(Optim.converged(result))
    initial_loss = $(Reactant.to_number(initial_loss))
    final_loss = $final_loss
    relative_data_reduction = $(1 - final_loss / data_norm)
    gradient_relative_error = $gradient_error
    density_extrema = $(extrema(density))
    display_range = $DISPLAY_RANGE
    """
    write(joinpath(output_directory, "metrics.txt"), metrics)
    println(metrics)
    density
end
