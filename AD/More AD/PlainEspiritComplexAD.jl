module PlainEspiritComplexAD128

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
const ITERATIONS = 40
const NAVIGATOR_COUNT = 3
const DEFAULT_DISPLAY_RANGE = (0f0, 2f-5)
const ARCHIVE_DIRECTORY = joinpath(homedir(), "Desktop/Archive (1)")
const OUTPUT_ROOT = joinpath(@__DIR__, "PlainEspiritComplexAD")

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

function sense_signal(x, model)
    image = model.interpolation_x * reshape(x, RECON_SIZE) * transpose(model.interpolation_y)
    coil_images = reshape(image, :, 1) .* model.maps
    shifted_images = reshape(
        coil_images[model.spatial_indices, :],
        RECON_SIZE...,
        size(model.maps, 2),
    )
    spectrum = KomaMRI.fft(shifted_images, (1, 2))
    reshape(spectrum, :, size(model.maps, 2))[model.sample_indices, :]
end

function objective(x, model)
    residual = sense_signal(x, model) - model.data
    sum(abs2, residual)
end

function objective_gradient(x, model)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, objective, x, Enzyme.Const(model))
    result.val, result.derivs[1]
end

function sense_adjoint(data, model)
    spectrum = zeros(ComplexF32, prod(RECON_SIZE), size(model.maps, 2))
    for sample in axes(data, 1)
        spectrum[model.sample_indices[sample], :] .+= @view data[sample, :]
    end
    shifted_images = prod(RECON_SIZE) .* KomaMRI.ifft(
        reshape(spectrum, RECON_SIZE..., size(model.maps, 2)),
        (1, 2),
    )
    coil_images = zeros(ComplexF32, prod(RECON_SIZE), size(model.maps, 2))
    coil_images[model.spatial_indices, :] .= reshape(
        shifted_images,
        :,
        size(model.maps, 2),
    )
    image = reshape(
        vec(sum(conj.(model.maps) .* coil_images; dims=2)),
        RECON_SIZE,
    )
    vec(transpose(model.interpolation_x) * image * model.interpolation_y)
end

function gram_norm(model)
    vector = fill(
        ComplexF32(inv(sqrt(Float32(prod(RECON_SIZE))))),
        prod(RECON_SIZE),
    )
    for _ in 1:30
        product = sense_adjoint(sense_signal(vector, model), model)
        vector .= product ./ norm(product)
    end
    real(dot(vector, sense_adjoint(sense_signal(vector, model), model)))
end

function build_problem(; accelerated)
    reference_file = joinpath(
        ARCHIVE_DIRECTORY,
        "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd",
    )
    measured_file = accelerated ? joinpath(
        ARCHIVE_DIRECTORY,
        "mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd",
    ) : reference_file
    sequence_file = joinpath(
        ARCHIVE_DIRECTORY,
        "seq",
        accelerated ?
            "hard_epi_2x_20interleaves_5avg_fatsat.seq" :
            "hard_epi_20interleaves_5avg_fatsat.seq",
    )

    seq = read_seq(sequence_file)
    acquired_lines = if accelerated
        shots = Int(seq.DEF["EpiShots"])
        acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
        [line for shot in acquired_shots for line in shot:shots:(RECON_SIZE[2] - 1)]
    else
        collect(0:(RECON_SIZE[2] - 1))
    end

    raw_reference = RawAcquisitionData(ISMRMRDFile(reference_file))
    raw_reference.profiles = raw_reference.profiles[
        (NAVIGATOR_COUNT + 1):(NAVIGATOR_COUNT + RECON_SIZE[2])
    ]
    reference = AcquisitionData(raw_reference)
    reference.traj[1].circular = false
    sensitivity_maps = espirit(
        reference,
        (6, 6),
        30,
        RECON_SIZE;
        eigThresh_1=0.02,
        eigThresh_2=0.0,
    )
    maps = reshape(Array(@view sensitivity_maps[:, :, 1, :]), prod(RECON_SIZE), :)

    raw_measured = RawAcquisitionData(ISMRMRDFile(measured_file))
    raw_measured.profiles = raw_measured.profiles[
        (NAVIGATOR_COUNT + 1):(NAVIGATOR_COUNT + length(acquired_lines))
    ]
    if accelerated
        foreach(raw_measured.profiles, acquired_lines) do profile, line
            profile.head.idx.kspace_encode_step_1 = UInt16(line)
        end
    end
    measured = AcquisitionData(raw_measured)
    measured.traj[1].circular = false

    fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3
    map_x = collect(LinRange(-fov[1] / 2, fov[1] / 2, RECON_SIZE[1]))
    map_y = collect(LinRange(-fov[2] / 2, fov[2] / 2, RECON_SIZE[2]))
    voxel_x, voxel_y = centered_axis.(fov[1:2], RECON_SIZE)
    interpolation_x = interpolation_matrix(voxel_x, map_x)
    interpolation_y = interpolation_matrix(voxel_y, map_y)
    spatial_indices = vec(KomaMRI.ifftshift(reshape(1:prod(RECON_SIZE), RECON_SIZE), (1, 2)))
    frequency_indices = vec(KomaMRI.fftshift(reshape(1:prod(RECON_SIZE), RECON_SIZE), (1, 2)))
    sample_indices = frequency_indices[measured.subsampleIndices[1]]
    data = ComplexF32.(measured.kdata[1])
    model = (; interpolation_x, interpolation_y, maps, spatial_indices, sample_indices, data)
    model, (; reference_file, measured_file, sequence_file)
end

function save_magnitude(
    x,
    filename,
    title,
    output_directory;
    display_range=DEFAULT_DISPLAY_RANGE,
)
    magnitude = clamp.(reshape(abs.(x), RECON_SIZE), display_range...)
    figure = plot_image(
        magnitude;
        title,
        zmin=first(display_range),
        zmax=last(display_range),
    )
    savefig(figure, joinpath(output_directory, filename))
    nothing
end

function save_loss_convergence(losses, name, output_directory)
    plotly = KomaMRIPlots.PlotlyBase
    losses = losses[1:min(length(losses), 6)]
    iterations = 0:(length(losses) - 1)
    case_title = name == "FullySampled128x128" ?
        "Fully sampled - 128 × 128" : "Accelerated R = 2 - 128 × 128"
    scaled_losses = 1e4 .* losses
    loss_labels = string.(round.(scaled_losses; digits=4))
    line_trace = plotly.scatter(;
        x=iterations,
        y=scaled_losses,
        mode="lines",
        name="Optimization trajectory",
        legendrank=2,
        line=plotly.attr(; color="#D84A5B", width=4),
        hoverinfo="skip",
    )
    point_trace = plotly.scatter(;
        x=iterations,
        y=scaled_losses,
        customdata=losses,
        mode="markers",
        name="Loss at each iteration",
        legendrank=1,
        marker=plotly.attr(;
            color="#14877C",
            size=10,
            line=plotly.attr(; color="white", width=1.5),
        ),
        hovertemplate="Iteration %{x}<br>Loss = %{customdata:.6e}<extra></extra>",
    )
    final_trace = plotly.scatter(;
        x=[last(iterations)],
        y=[last(scaled_losses)],
        mode="markers",
        marker=plotly.attr(;
            symbol="circle-open",
            color="#374151",
            size=20,
            line=plotly.attr(; color="#374151", width=3),
        ),
        hoverinfo="skip",
        showlegend=false,
    )
    label_shifts = map(
        loss -> loss < 0.08 * maximum(scaled_losses) ? 18 : -18,
        scaled_losses[1:(end - 1)],
    )
    annotations = [plotly.attr(;
        xref="x",
        yref="y",
        x=iteration,
        y=loss,
        text=label,
        showarrow=false,
        xanchor=iteration == first(iterations) ? "left" : "center",
        yanchor=shift < 0 ? "top" : "bottom",
        yshift=shift,
        font=plotly.attr(; family="Arial", size=13, color="#3F4650"),
    ) for (iteration, loss, label, shift) in
         zip(iterations[1:(end - 1)], scaled_losses[1:(end - 1)], loss_labels[1:(end - 1)], label_shifts)]
    push!(annotations, plotly.attr(;
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.01,
        text="<b>Complex AD Loss Convergence</b><br><span style='font-size:18px'>$(case_title)</span>",
        showarrow=false,
        xanchor="center",
        yanchor="bottom",
        align="center",
        font=plotly.attr(; family="Arial", size=28, color="#202124"),
    ))
    push!(annotations, plotly.attr(;
        x=last(iterations),
        y=last(scaled_losses),
        text="Iteration $(last(iterations))<br><b>loss = $(last(loss_labels)) × 10<sup>-4</sup></b>",
        showarrow=true,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#374151",
        ax=-145,
        ay=-75,
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        borderpad=4,
        font=plotly.attr(; family="Arial", size=15, color="#1F2937"),
    ))
    layout = plotly.Layout(;
        font=plotly.attr(; family="Arial", size=18, color="#202124"),
        margin=plotly.attr(; l=115, r=45, t=155, b=90),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=plotly.attr(;
            x=0.985,
            y=0.985,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="#C7CBD1",
            borderwidth=1,
            font=plotly.attr(; size=15),
        ),
        xaxis=plotly.attr(;
            title=plotly.attr(; text="Optimization iteration, <i>k</i>", standoff=15),
            range=[-0.75, last(iterations) + 0.75],
            nticks=6,
            showline=true,
            mirror=true,
            ticks="outside",
            ticklen=6,
            linecolor="#7A7F87",
            gridcolor="#C9CDD2",
            gridwidth=1,
            zeroline=false,
        ),
        yaxis=plotly.attr(;
            title=plotly.attr(;
                text="Loss, <i>L</i><sub>k</sub> (×10<sup>-4</sup>)",
                standoff=12,
            ),
            range=[0, 1.18 * maximum(scaled_losses)],
            showline=true,
            mirror=true,
            ticks="outside",
            ticklen=6,
            linecolor="#7A7F87",
            gridcolor="#C9CDD2",
            gridwidth=1,
            zeroline=false,
        ),
        annotations,
    )
    figure = plotly.Plot([line_trace, point_trace, final_trace], layout)
    savefig(figure, joinpath(output_directory, "loss_convergence.png"); width=1100, height=820, scale=2)
    savefig(figure, joinpath(output_directory, "loss_convergence.pdf"); width=1100, height=820)
    nothing
end

function read_image(filename)
    image = Vector{ComplexF32}(undef, prod(RECON_SIZE))
    open(filename) do io
        read!(io, image)
        eof(io) || error("Unexpected extra image data in $filename")
    end
    image
end

function run_case(name; accelerated, iterations=ITERATIONS)
    output_directory = joinpath(OUTPUT_ROOT, name)
    mkpath(output_directory)
    model, inputs = build_problem(; accelerated)
    step = 0.45f0 / gram_norm(model)

    zero_image = Reactant.to_rarray(zeros(ComplexF32, prod(RECON_SIZE)))
    device_model = map(Reactant.to_rarray, model)
    compiled_gradient = Reactant.@compile sync=true objective_gradient(zero_image, device_model)
    loss, gradient = compiled_gradient(zero_image, device_model)
    analytic_gradient = -2f0 .* sense_adjoint(model.data, model)
    gradient_error = norm(Array(gradient) - analytic_gradient) / norm(analytic_gradient)
    gradient_error < 2f-3 || error("Complex AD gradient check failed: $gradient_error")

    data_norm = sum(abs2, model.data)
    losses = Float64[Reactant.to_number(loss)]
    x = zero_image
    for iteration in 1:iterations
        x = x .- step .* gradient
        loss, gradient = compiled_gradient(x, device_model)
        push!(losses, Reactant.to_number(loss))
        image = Array(x)
        stem = "iteration_$(lpad(iteration, 2, '0'))"
        save_magnitude(
            image,
            "$stem.png",
            "Complex plain ESPIRiT AD $name iteration $iteration",
            output_directory,
        )
        open(joinpath(output_directory, "$stem.cf32"), "w") do io
            write(io, image)
        end
        println("$name iteration $iteration: loss = $(last(losses))")
    end

    image = Array(x)
    final_loss = last(losses)
    save_loss_convergence(losses, name, output_directory)
    save_magnitude(
        image,
        "reconstructed_magnitude.png",
        "Complex plain ESPIRiT AD $name",
        output_directory,
    )
    open(joinpath(output_directory, "reconstructed_image.cf32"), "w") do io
        write(io, image)
    end
    metrics = """
    method = complex AD with plain ESPIRiT maps
    case = $name
    resolution = 128x128
    acceleration = $(accelerated ? 2 : 1)
    coils = $(size(model.data, 2))
    samples_per_coil = $(size(model.data, 1))
    iterations = $iterations
    final_loss = $final_loss
    relative_data_reduction = $(1 - final_loss / data_norm)
    gradient_relative_error = $gradient_error
    magnitude_extrema = $(extrema(abs.(image)))
    display_range = $DEFAULT_DISPLAY_RANGE
    sensitivity_maps = plain ESPIRiT from $(inputs.reference_file)
    measured_data = $(inputs.measured_file)
    sequence = $(inputs.sequence_file)
    """
    write(joinpath(output_directory, "metrics.txt"), metrics)
    println(metrics)
    image
end

function render_case(name, iterations; percentile=0.995)
    output_directory = joinpath(OUTPUT_ROOT, name)
    final_file = joinpath(
        output_directory,
        "iteration_$(lpad(iterations, 2, '0')).cf32",
    )
    final_image = read_image(final_file)
    display_maximum = Float32(quantile(abs.(final_image), percentile))
    display_range = (0f0, display_maximum)

    for iteration in 1:iterations
        stem = "iteration_$(lpad(iteration, 2, '0'))"
        image = read_image(joinpath(output_directory, "$stem.cf32"))
        save_magnitude(
            image,
            "$stem.png",
            "Complex plain ESPIRiT AD $name iteration $iteration",
            output_directory;
            display_range,
        )
    end
    save_magnitude(
        final_image,
        "reconstructed_magnitude.png",
        "Complex plain ESPIRiT AD $name",
        output_directory;
        display_range,
    )

    scale = """
    display_percentile = $(100percentile)
    display_range = $display_range
    display_scale_is_case_specific = true
    """
    write(joinpath(output_directory, "display_scale.txt"), scale)
    metrics_file = joinpath(output_directory, "metrics.txt")
    metrics = replace(
        read(metrics_file, String),
        r"display_range = .*\n" => "display_range = $display_range\n",
    )
    write(metrics_file, metrics * "display_percentile = $(100percentile)\n")
    println("$name display range = $display_range")
    display_range
end

function run_all()
    run_case("FullySampled128x128"; accelerated=false)
    run_case("AcceleratedR2_128x128"; accelerated=true)
    render_case("FullySampled128x128", ITERATIONS)
    render_case("AcceleratedR2_128x128", ITERATIONS)
    nothing
end

end
