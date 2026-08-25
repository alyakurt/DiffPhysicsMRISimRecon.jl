using DelimitedFiles
using Distributed
using Enzyme
using KomaMRI
using KomaMRIPlots
using KomaMRIPlots: QMRIColors
using LinearAlgebra: Diagonal
using Statistics: quantile

Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.strictAliasing!(false)

const IMAGE_SIZE = (192, 144)
const CONTRASTS = 8
const ITERATIONS = 80
const BATCH_SIZE = 256
const SHARDS = 8
const CHECKPOINTS = (0, 20, 40, 80)
const INPUT_DIRECTORY = isempty(ARGS) ?
    joinpath(@__DIR__, "FINAL_T1_AD_OPENMOLLI", "inputs") : abspath(ARGS[1])
const OUTPUT_DIRECTORY = length(ARGS) < 2 ?
    joinpath(@__DIR__, "FINAL_T1_AD_ENZYME_FORWARD") : abspath(ARGS[2])
const MODE = Enzyme.set_runtime_activity(Enzyme.ForwardWithPrimal)

function load_sequence(input_directory)
    sequence = read_seq(joinpath(input_directory, "OpenMOLLI_LCD_S.seq"))
    adc_blocks = findall(i -> is_ADC_on(sequence[i]), eachindex(sequence.DUR))
    sequence = sequence[1:last(adc_blocks)]
    rr = 1e-3 .* [
        1212.5, 1135.0, 1117.5, 1052.5, 925.0, 1082.5,
        1195.0, 1032.5, 1165.0, 1177.5, 1010.0, 1030.0,
    ]
    sequence = resolve_triggers(
        sequence, CardiacSignal(; rr_intervals=rr, first_peak=100e-6),
    )
    sim_params = KomaMRICore.default_sim_params(Dict{String,Any}(
        "sim_method" => Bloch(), "gpu" => false, "Nthreads" => 1,
        "return_type" => "mat", "precision" => "f32",
    ))
    sampling_rule = KomaMRICore.simulation_sampling_rule(Bloch(), sim_params)
    seqd = discretize(sequence; sampling_rule)
    acquired = findall(seqd.ADC)
    centers = [
        ((contrast - 1) * 95 + 52) * 192 + 96
        for contrast in 1:CONTRASTS
    ]
    adc = falses(length(seqd.ADC))
    adc[acquired[centers]] .= true
    seqd = DiscreteSequence(
        seqd.Gx, seqd.Gy, seqd.Gz, seqd.B1, seqd.Δf, seqd.ψ,
        adc, seqd.excitation_bool, seqd.t, seqd.Δt,
    ) |> f32
    parts, excitation = KomaMRICore.get_sim_ranges(
        seqd;
        max_block_length=sim_params["max_block_length"],
        max_rf_block_length=sim_params["max_rf_block_length"],
        eval_intervals_per_step=KomaMRICore.eval_intervals_per_step(Bloch()),
    )
    max_adc = maximum(
        (count(@view seqd.ADC[(first(part) + 1):last(part)]) for part in parts);
        init=0,
    )
    (; seqd, parts, excitation, max_adc)
end

function load_target(input_directory)
    contrasts = Array{Float32}(undef, IMAGE_SIZE..., CONTRASTS)
    read!(joinpath(input_directory, "signed_contrasts.f32"), contrasts)
    support = Array{UInt8}(undef, IMAGE_SIZE)
    read!(joinpath(input_directory, "support.u8"), support)
    indices = findall(!iszero, vec(support))
    signals = reshape(permutedims(contrasts, (3, 1, 2)), CONTRASTS, :)
    permutedims(signals[:, indices]), Bool.(support), indices
end

function A(nodes, p)
    count = size(p.target, 1)
    object = Phantom(;
        x=zeros(Float32, count), y=zeros(Float32, count), z=zeros(Float32, count),
        ρ=exp.(nodes[(count + 1):(2count)]), T1=nodes[1:count],
        T2=fill(0.05f0, count), T2s=fill(Float32(Inf), count),
    )
    state, object = KomaMRICore.initialize_spins_state(object, Bloch())
    signal = zeros(ComplexF32, CONTRASTS, count, 1)
    KomaMRICore.run_sim_time_iter!(
        object, p.sequence.seqd, signal, state, p.scanner, Bloch(),
        KomaMRICore.KA.CPU();
        Nblocks=length(p.sequence.parts), Nthreads=1, parts=p.sequence.parts,
        excitation_bool=p.sequence.excitation,
        max_adc_samples=p.sequence.max_adc,
    )
    transpose(imag.(signal[:, :, 1]))
end

function f(nodes, p)
    residual = A(nodes, p) .- p.target
    reference_energy = max.(sum(abs2, p.target; dims=2), floatmin(Float32))
    vec(sum(abs2, residual; dims=2) ./ reference_energy)
end

function f_and_gradient(nodes, seed, p)
    result = Enzyme.autodiff(
        MODE, f, Enzyme.Duplicated(nodes, seed), Enzyme.Const(p),
    )
    result[2], result[1]
end

function run_shard(shard, input_directory)
    sequence = load_sequence(input_directory)
    target, _, linear_indices = load_target(input_directory)
    width = cld(length(linear_indices), SHARDS)
    pixels = ((shard - 1) * width + 1):min(shard * width, length(linear_indices))
    target = target[pixels, :]
    linear_indices = linear_indices[pixels]
    count = length(pixels)
    T1 = ones(Float32, count)
    log_density = log.(max.(maximum(abs, target; dims=2)[:], eps(Float32)))
    moment = zeros(Float32, 2count)
    variance = zeros(Float32, 2count)
    gradient = similar(moment)
    losses = zeros(Float32, count)
    snapshots = Dict(iteration => zeros(Float32, count) for iteration in CHECKPOINTS)
    scanner = Scanner(; receiver=KomaMRICore.CoilSensitivities(
        Diagonal(ones(ComplexF32, BATCH_SIZE)), nothing,
    ))
    t1_seed = vcat(ones(Float32, BATCH_SIZE), zeros(Float32, BATCH_SIZE))
    density_seed = reverse(t1_seed)
    batches = collect(Iterators.partition(1:count, BATCH_SIZE))

    function evaluate!(batch)
        actual = collect(batch)
        padded = [actual; fill(last(actual), BATCH_SIZE - length(actual))]
        p = (; sequence, scanner, target=target[padded, :])
        nodes = vcat(T1[padded], log_density[padded])
        values, t1_gradient = f_and_gradient(nodes, t1_seed, p)
        _, density_gradient = f_and_gradient(nodes, density_seed, p)
        n = length(actual)
        losses[actual] .= values[1:n]
        gradient[actual] .= t1_gradient[1:n]
        gradient[count .+ actual] .= density_gradient[1:n]
    end

    history = zeros(Float64, ITERATIONS + 1)
    for iteration in 0:ITERATIONS
        foreach(evaluate!, batches)
        history[iteration + 1] = sum(losses) / count
        iteration in CHECKPOINTS && (snapshots[iteration] .= T1)
        println("shard=$shard iteration=$iteration loss=$(history[iteration + 1])")
        iteration == ITERATIONS && break
        moment .= 0.9f0 .* moment .+ 0.1f0 .* gradient
        variance .= 0.999f0 .* variance .+ 0.001f0 .* abs2.(gradient)
        correction = sqrt(1f0 - 0.999f0^(iteration + 1)) /
            (1f0 - 0.9f0^(iteration + 1))
        step = 0.02f0 .* correction .* moment ./ (sqrt.(variance) .+ 1f-8)
        T1 .= clamp.(T1 .- step[1:count], 0.1f0, 3f0)
        log_density .-= step[(count + 1):(2count)]
    end
    (; linear_indices, snapshots, density=exp.(log_density), history)
end

function save_results(results, input_directory, output_directory)
    _, support, _ = load_target(input_directory)
    maps = Dict(iteration => zeros(Float32, prod(IMAGE_SIZE)) for iteration in CHECKPOINTS)
    density = zeros(Float32, prod(IMAGE_SIZE))
    loss = zeros(Float64, ITERATIONS + 1)
    total = 0
    for result in results
        count = length(result.linear_indices)
        for iteration in CHECKPOINTS
            maps[iteration][result.linear_indices] .= result.snapshots[iteration]
        end
        density[result.linear_indices] .= result.density
        loss .+= count .* result.history
        total += count
    end
    mkpath(output_directory)
    for iteration in CHECKPOINTS
        write(joinpath(output_directory, "iteration_$(lpad(iteration, 3, '0')).f32"),
            reshape(maps[iteration], IMAGE_SIZE))
    end
    write(joinpath(output_directory, "quantitative_t1.f32"),
        reshape(maps[ITERATIONS], IMAGE_SIZE))
    write(joinpath(output_directory, "quantitative_spin_density.f32"),
        reshape(density, IMAGE_SIZE))
    write(joinpath(output_directory, "support.u8"), UInt8.(support))
    loss ./= total
    rows = ["iteration,loss"; ["$i,$value" for (i, value) in zip(0:ITERATIONS, loss)]]
    write(joinpath(output_directory, "loss.csv"), join(rows, '\n') * "\n")
    (; T1=reshape(maps[ITERATIONS], IMAGE_SIZE), density=reshape(density, IMAGE_SIZE),
       support, loss)
end

function render_results(result, output_directory)
    plotly = KomaMRIPlots.PlotlyBase
    t1 = reverse(result.T1; dims=(1, 2)) .* 1f3
    density = reverse(result.density; dims=(1, 2))
    support = reverse(result.support; dims=(1, 2))
    t1_values = Matrix{Union{Missing,Float32}}(missing, IMAGE_SIZE)
    density_values = similar(t1_values)
    fill!(density_values, missing)
    t1_values[support] .= t1[support]
    density_values[support] .= density[support]
    colors = replace.(string.(QMRIColors.relaxationColorMap("T1") .* 255),
        "RGB{Float64}" => "rgb")
    scale = collect(zip(range(0, 1; length=length(colors)), colors))
    layout(title) = plotly.Layout(;
        title=plotly.attr(; text=title, x=0.5), width=3000, height=3600,
        margin=plotly.attr(; t=270, l=160, r=460, b=140),
        xaxis=plotly.attr(; visible=false),
        yaxis=plotly.attr(; visible=false, autorange="reversed", scaleanchor="x"),
        paper_bgcolor="white", plot_bgcolor="black",
        font=plotly.attr(; family="Arial", size=56, color="#263c5c"),
    )
    t1_plot = plotly.Plot(plotly.heatmap(;
        z=t1_values, zmin=0, zmax=2000, zsmooth="best", colorscale=scale,
        colorbar=plotly.attr(; title="T1 (ms)", tickvals=0:500:2000),
    ), layout("openMOLLIst Koma Enzyme forward AD T1 - 80 iterations"))
    density_plot = plotly.Plot(plotly.heatmap(;
        z=density_values, zmin=0, zmax=quantile(density[result.support], 0.995),
        colorscale=[(0.0, "black"), (1.0, "white")], showscale=false,
    ), layout("openMOLLIst Koma Enzyme forward AD spin density"))
    loss_plot = plotly.Plot(plotly.scatter(;
        x=0:ITERATIONS, y=result.loss, mode="lines",
    ), plotly.Layout(; title="Koma Enzyme forward AD T1 loss convergence",
        width=2400, height=1400, xaxis=plotly.attr(; title="Iteration"),
        yaxis=plotly.attr(; title="Normalized measured-data loss")))
    KomaMRIPlots.savefig(t1_plot, joinpath(output_directory, "t1_map.png");
        width=3000, height=3600)
    KomaMRIPlots.savefig(density_plot, joinpath(output_directory, "spin_density.png");
        width=3000, height=3600)
    KomaMRIPlots.savefig(loss_plot, joinpath(output_directory, "loss_convergence.png");
        width=2400, height=1400)
end

function run_final_t1_ad()
    project = dirname(Base.active_project())
    workers_added = addprocs(SHARDS; exeflags=`--project=$project --threads=1`)
    try
        for worker in workers_added
            remotecall_wait(Base.include, worker, Main, @__FILE__)
        end
        results = pmap(shard -> run_shard(shard, INPUT_DIRECTORY), 1:SHARDS)
        result = save_results(results, INPUT_DIRECTORY, OUTPUT_DIRECTORY)
        render_results(result, OUTPUT_DIRECTORY)
        println("initial_loss=$(result.loss[1]) final_loss=$(result.loss[end])")
        result
    finally
        rmprocs(workers_added)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && myid() == 1 && run_final_t1_ad()
