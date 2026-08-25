module MeasuredT2AD128

using Enzyme
using KomaMRI
using KomaMRIPlots
using LinearAlgebra: dot, norm
using MRICoilSensitivities: espirit
using Reactant
using Statistics: quantile
@eval using $(Sys.isapple() ? :Metal : :CUDA)

Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", Sys.isapple() ? "cpu" : "cuda"))
Reactant.allowscalar(false)

const RECON_SIZE = (128, 128)
const NAVIGATOR_COUNT = 3
const IMAGE_ITERATIONS = 40
const FIT_ITERATIONS = 500
const SUPPORT_FRACTION = 0.05f0
const T2_LIMITS = (0.005f0, 0.5f0)
const ARCHIVE_DIRECTORY = joinpath(homedir(), "Desktop", "Archive (1)")
const OUTPUT_DIRECTORY = joinpath(@__DIR__, "MeasuredT2", "128x128")
const ECHO_ACQUISITIONS = (
    (;
        echo_time=0.011028f0,
        filename="meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd",
        lines=collect(0:127),
    ),
    (;
        echo_time=0.038468f0,
        filename="meas_MID01092_FID34192_hard_epi_accel3_one_set_fatsat.mrd",
        lines=collect(1:3:127),
    ),
    (;
        echo_time=0.055406f0,
        filename="meas_MID01091_FID34191_hard_epi_accel2_one_set_fatsat.mrd",
        lines=collect(0:2:126),
    ),
    (;
        echo_time=0.104686f0,
        filename="meas_MID01093_FID34193_hard_epi_1shot_fatsat.mrd",
        lines=collect(0:127),
    ),
)

centered_axis(width, count) = range(
    -width / 2 + width / (2count);
    step=width / count,
    length=count,
)

function interpolation_matrix(source, target)
    matrix = zeros(Float32, length(target), length(source))
    for (row, position) in enumerate(target)
        if position <= first(source)
            matrix[row, 1] = 1
        elseif position >= last(source)
            matrix[row, end] = 1
        else
            left = searchsortedlast(source, position)
            weight = Float32((position - source[left]) / (source[left + 1] - source[left]))
            matrix[row, left] = 1 - weight
            matrix[row, left + 1] = weight
        end
    end
    matrix
end

function averaged_acquisition(specification)
    filename = joinpath(ARCHIVE_DIRECTORY, "mrd_hdf5", specification.filename)
    raw = RawAcquisitionData(ISMRMRDFile(filename))
    imaging_profiles = raw.profiles[(NAVIGATOR_COUNT + 1):end]
    expected_lines = length(specification.lines)
    averages = sort(unique(profile.head.idx.average for profile in imaging_profiles))
    groups = [
        filter(profile -> profile.head.idx.average == average, imaging_profiles) for
        average in averages
    ]
    complete_groups = filter(groups) do group
        raw_lines = unique(profile.head.idx.kspace_encode_step_1 for profile in group)
        length(group) == expected_lines && length(raw_lines) == expected_lines
    end
    isempty(complete_groups) && error("No complete averages in $filename")

    template = first(complete_groups)
    profiles = map(template) do source
        raw_line = source.head.idx.kspace_encode_step_1
        matches = map(complete_groups) do group
            only(filter(profile -> profile.head.idx.kspace_encode_step_1 == raw_line, group))
        end
        profile = deepcopy(source)
        fill!(profile.data, 0)
        foreach(match -> profile.data .+= match.data, matches)
        profile.data ./= Float32(length(matches))
        profile.head.idx.kspace_encode_step_1 = UInt16(specification.lines[Int(raw_line) + 1])
        profile
    end
    raw.profiles = profiles
    acquisition = AcquisitionData(raw)
    acquisition.traj[1].circular = false
    acquisition, filename, length(complete_groups)
end

function sense_signal(image, model)
    interpolated = model.interpolation_x *
        reshape(image, RECON_SIZE) *
        transpose(model.interpolation_y)
    coil_images = reshape(interpolated, :, 1) .* model.maps
    shifted = reshape(
        coil_images[model.spatial_indices, :],
        RECON_SIZE...,
        size(model.maps, 2),
    )
    spectrum = KomaMRI.fft(shifted, (1, 2))
    reshape(spectrum, :, size(model.maps, 2))[model.sample_indices, :]
end

function image_loss(image, model)
    residual = sense_signal(image, model) .- model.data
    sum(abs2, residual)
end

function image_loss_gradient(image, model)
    result = Enzyme.gradient(
        Enzyme.ReverseWithPrimal,
        image_loss,
        image,
        Enzyme.Const(model),
    )
    result.val, result.derivs[1]
end

function sense_adjoint(data, model)
    spectrum = zeros(ComplexF32, prod(RECON_SIZE), size(model.maps, 2))
    for sample in axes(data, 1)
        spectrum[model.sample_indices[sample], :] .+= @view data[sample, :]
    end
    shifted = prod(RECON_SIZE) .* KomaMRI.ifft(
        reshape(spectrum, RECON_SIZE..., size(model.maps, 2)),
        (1, 2),
    )
    coil_images = zeros(ComplexF32, prod(RECON_SIZE), size(model.maps, 2))
    coil_images[model.spatial_indices, :] .= reshape(shifted, :, size(model.maps, 2))
    combined = reshape(
        vec(sum(conj.(model.maps) .* coil_images; dims=2)),
        RECON_SIZE,
    )
    vec(transpose(model.interpolation_x) * combined * model.interpolation_y)
end

function gram_norm(model)
    image = fill(ComplexF32(inv(sqrt(Float32(prod(RECON_SIZE))))), prod(RECON_SIZE))
    for _ in 1:30
        product = sense_adjoint(sense_signal(image, model), model)
        image .= product ./ norm(product)
    end
    real(dot(image, sense_adjoint(sense_signal(image, model), model)))
end

function reconstruct_echo(model)
    device_model = map(Reactant.to_rarray, model)
    image = Reactant.to_rarray(zeros(ComplexF32, prod(RECON_SIZE)))
    compiled_gradient = Reactant.@compile sync=true image_loss_gradient(image, device_model)
    loss, gradient = compiled_gradient(image, device_model)
    analytic_gradient = -2f0 .* sense_adjoint(model.data, model)
    gradient_error = norm(Array(gradient) - analytic_gradient) / norm(analytic_gradient)
    gradient_error < 2f-3 || error("Complex image AD gradient check failed: $gradient_error")
    step_size = 0.45f0 / gram_norm(model)
    losses = Float64[Reactant.to_number(loss)]
    for _ in 1:IMAGE_ITERATIONS
        image = image .- step_size .* gradient
        loss, gradient = compiled_gradient(image, device_model)
        push!(losses, Reactant.to_number(loss))
    end
    Array(image), losses, gradient_error
end

function t2_signal(control, model)
    signal_scale = reshape(model.signal_scale .* control[:, 1], :, 1)
    t2 = reshape(model.t2_lower .+ model.t2_span .* control[:, 2], :, 1)
    signal_scale .* exp.(-reshape(model.echo_times, 1, :) ./ t2)
end

function t2_loss(control, model)
    residual = (t2_signal(control, model) .- model.magnitudes) .* model.support
    sum(abs2, residual) / model.data_norm
end

function t2_loss_gradient(control, model)
    result = Enzyme.gradient(
        Enzyme.ReverseWithPrimal,
        t2_loss,
        control,
        Enzyme.Const(model),
    )
    result.val, result.derivs[1]
end

function gradient_relative_error(control, model, gradient)
    direction = gradient ./ norm(gradient)
    step = 1f-3
    finite_difference = (
        t2_loss(control .+ step .* direction, model) -
        t2_loss(control .- step .* direction, model)
    ) / (2step)
    directional_ad = dot(gradient, direction)
    abs(directional_ad - finite_difference) /
        max(abs(directional_ad), abs(finite_difference), eps(Float32))
end

function fit_t2(magnitudes, echo_times)
    threshold = SUPPORT_FRACTION * maximum(@view magnitudes[:, 1])
    support = Float32.(magnitudes[:, 1] .>= threshold)
    initial_t2 = 0.08f0
    lower, upper = T2_LIMITS
    signal_scale = maximum(magnitudes)
    initial_signal = magnitudes[:, 1] .* exp(echo_times[1] / initial_t2) ./ signal_scale
    initial = hcat(
        clamp.(initial_signal, 0f0, 2f0),
        fill((initial_t2 - lower) / (upper - lower), prod(RECON_SIZE)),
    )
    model = (;
        magnitudes=Float32.(magnitudes),
        echo_times,
        support=reshape(support, :, 1),
        signal_scale=Float32(signal_scale),
        t2_lower=lower,
        t2_span=upper - lower,
        data_norm=Float32(sum(abs2, magnitudes .* reshape(support, :, 1))),
    )
    device_model = map(Reactant.to_rarray, model)
    control = Reactant.to_rarray(initial)
    compiled_gradient = Reactant.@compile sync=true t2_loss_gradient(control, device_model)
    loss, gradient = compiled_gradient(control, device_model)
    gradient_error = gradient_relative_error(initial, model, Array(gradient))
    gradient_error < 1f-2 || error("Measured T2 AD gradient check failed: $gradient_error")
    losses = Float64[Reactant.to_number(loss)]
    step_size = 0.05f0 / max(maximum(abs, Array(gradient)), eps(Float32))

    for iteration in 1:FIT_ITERATIONS
        accepted = false
        trial_step = step_size
        for _ in 1:12
            trial = control .- trial_step .* gradient
            signal_control = clamp.(trial[:, 1], 0f0, 2f0)
            t2_control = clamp.(trial[:, 2], 0f0, 1f0)
            trial = hcat(signal_control, t2_control)
            trial_loss, trial_gradient = compiled_gradient(trial, device_model)
            trial_loss_value = Reactant.to_number(trial_loss)
            if trial_loss_value <= last(losses)
                control = trial
                gradient = trial_gradient
                push!(losses, trial_loss_value)
                step_size = 1.05f0 * trial_step
                accepted = true
                break
            end
            trial_step *= 0.5f0
        end
        accepted || error("T2 line search failed at iteration $iteration")
        iteration == 1 || iteration % 25 == 0 || continue
        println("Measured T2 AD iteration $iteration: loss = $(last(losses))")
    end

    fitted = Array(control)
    t2 = lower .+ (upper - lower) .* fitted[:, 2]
    t2 .*= support
    predicted = t2_signal(fitted, model)
    residual = sqrt.(sum(abs2, predicted .- magnitudes; dims=2)) ./
        max.(sqrt.(sum(abs2, magnitudes; dims=2)), eps(Float32))
    t2, vec(residual) .* support, losses, count(!iszero, support), gradient_error
end

function save_map(values, filename, title; maximum_value)
    figure = plot_image(
        reshape(values, RECON_SIZE);
        title,
        zmin=0,
        zmax=maximum_value,
    )
    savefig(figure, joinpath(OUTPUT_DIRECTORY, filename))
    nothing
end


function save_t2_map(t2)
    colors = replace.(
        string.(KomaMRIPlots.relaxationColorMap("T2") .* 255),
        "RGB{Float64}" => "rgb",
    )
    colorscale = [
        (position, color) for
        (position, color) in zip(range(0, 1; length=length(colors)), colors)
    ]
    figure = plot_image(
        reshape(1f3 .* t2, RECON_SIZE);
        title="Approximate measured T2 map (ms)",
        zmin=0,
        zmax=300,
        colorscale,
    )
    savefig(figure, joinpath(OUTPUT_DIRECTORY, "measured_t2.png"))
    nothing
end

function save_loss(losses)
    plotly = KomaMRIPlots.PlotlyBase
    trace = plotly.scatter(;
        x=0:(length(losses) - 1),
        y=max.(losses, eps(Float64)),
        mode="lines",
    )
    layout = plotly.Layout(;
        title="Measured T2 AD convergence",
        xaxis_title="Iteration",
        yaxis_title="Normalized magnitude-fit loss",
        yaxis_type="log",
        template="plotly_white",
    )
    savefig(plotly.Plot(trace, layout), joinpath(OUTPUT_DIRECTORY, "loss_convergence.png"))
    nothing
end

function run()
    mkpath(OUTPUT_DIRECTORY)
    acquisitions = map(averaged_acquisition, ECHO_ACQUISITIONS)
    reference = first(acquisitions)[1]
    sensitivity_maps = espirit(
        reference,
        (6, 6),
        30,
        RECON_SIZE;
        eigThresh_1=0.02,
        eigThresh_2=0.0,
    )
    sensitivity_maps ./= max.(sqrt.(sum(abs2, sensitivity_maps; dims=4)), eps(Float32))
    maps = ComplexF32.(reshape(@view(sensitivity_maps[:, :, 1, :]), prod(RECON_SIZE), :))

    fov = Float32.(reference.fov[1:2])
    map_x = collect(LinRange(-fov[1] / 2, fov[1] / 2, RECON_SIZE[1]))
    map_y = collect(LinRange(-fov[2] / 2, fov[2] / 2, RECON_SIZE[2]))
    voxel_x, voxel_y = centered_axis.(fov, RECON_SIZE)
    interpolation_x = interpolation_matrix(voxel_x, map_x)
    interpolation_y = interpolation_matrix(voxel_y, map_y)
    spatial_indices = vec(KomaMRI.ifftshift(reshape(1:prod(RECON_SIZE), RECON_SIZE), (1, 2)))
    frequency_indices = vec(KomaMRI.fftshift(reshape(1:prod(RECON_SIZE), RECON_SIZE), (1, 2)))

    images = Matrix{ComplexF32}(undef, prod(RECON_SIZE), length(acquisitions))
    image_metrics = String[]
    for (echo, (acquisition, filename, averages)) in enumerate(acquisitions)
        data = ComplexF32.(acquisition.kdata[1])
        model = (;
            interpolation_x,
            interpolation_y,
            maps,
            spatial_indices,
            sample_indices=frequency_indices[acquisition.subsampleIndices[1]],
            data,
        )
        image, losses, gradient_error = reconstruct_echo(model)
        images[:, echo] .= image
        echo_ms = 1f3 * ECHO_ACQUISITIONS[echo].echo_time
        stem = "echo_$(round(Int, echo_ms))ms"
        open(joinpath(OUTPUT_DIRECTORY, "$stem.cf32"), "w") do io
            write(io, image)
        end
        display_maximum = Float32(quantile(abs.(image), 0.995))
        save_map(abs.(image), "$stem.png", "Measured spin-echo EPI, TE = $echo_ms ms"; maximum_value=display_maximum)
        push!(image_metrics, "echo_time_ms = $echo_ms, samples_per_coil = $(size(data, 1)), complete_averages = $averages, image_final_loss = $(last(losses)), gradient_relative_error = $gradient_error, measured_data = $filename")
        println(last(image_metrics))
    end

    magnitudes = abs.(images)
    echo_times = Float32[specification.echo_time for specification in ECHO_ACQUISITIONS]
    t2, residual, losses, support_nodes, gradient_error = fit_t2(magnitudes, echo_times)
    open(joinpath(OUTPUT_DIRECTORY, "measured_t2_nodes.f32"), "w") do io
        write(io, Float32.(t2))
    end
    save_t2_map(t2)
    save_map(residual, "relative_fit_residual.png", "Relative mono-exponential fit residual"; maximum_value=1f0)
    save_loss(losses)
    open(joinpath(OUTPUT_DIRECTORY, "loss.csv"), "w") do io
        println(io, "iteration,loss")
        foreach(enumerate(losses)) do (iteration, loss)
            println(io, "$(iteration - 1),$loss")
        end
    end

    supported_t2 = t2[t2 .> 0]
    supported_residual = residual[t2 .> 0]
    metrics = """
    method = measured multi-echo spin-echo EPI reconstruction followed by node-wise AD mono-exponential fitting
    parameter = approximate T2
    resolution = 128x128
    image_node_type = ComplexF32
    relaxation_node_type = Float32 seconds
    echo_times_ms = $(1f3 .* echo_times)
    image_iterations_per_echo = $IMAGE_ITERATIONS
    fit_iterations = $FIT_ITERATIONS
    initial_fit_loss = $(first(losses))
    final_fit_loss = $(last(losses))
    relative_fit_loss_reduction = $(1 - last(losses) / first(losses))
    fit_gradient_relative_error = $gradient_error
    support_nodes = $support_nodes
    support_threshold_fraction = $SUPPORT_FRACTION
    supported_t2_extrema_ms = $(extrema(1f3 .* supported_t2))
    supported_t2_median_ms = $(1f3 * quantile(supported_t2, 0.5))
    median_relative_fit_residual = $(quantile(supported_residual, 0.5))
    limitation = acquisitions use different EPI shot and acceleration patterns; EPI distortion, T2-star readout blurring, and inter-scan motion can bias the fitted T2
    $(join(image_metrics, '\n'))
    """
    write(joinpath(OUTPUT_DIRECTORY, "metrics.txt"), metrics)
    println(metrics)
    t2
end

end
