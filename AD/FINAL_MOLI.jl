using CUDA, Enzyme, KomaMRI, KomaMRIPlots
using KomaMRIPlots: QMRIColors
using LinearAlgebra: Diagonal

const SIZE = (192, 144)
const CONTRASTS, PROFILES, SAMPLES = 8, 95, 192
const CENTER_PROFILE, CENTER_SAMPLE = 52, 96
const ITERATIONS, GRID = 80, 1024
const T1_MIN, T1_MAX = 0.1f0, 3f0
const INPUT = isempty(ARGS) ? joinpath(@__DIR__, "More AD", "FINAL_T1_AD_OPENMOLLI", "inputs") : abspath(ARGS[1])
const OUTPUT = length(ARGS) < 2 ? joinpath(@__DIR__, "FINAL_MOLI") : abspath(ARGS[2])
const MODE = Enzyme.set_runtime_activity(Enzyme.ForwardWithPrimal)
Enzyme.API.looseTypeAnalysis!(true); Enzyme.API.strictAliasing!(false); CUDA.allowscalar(false)
t1_colorscale() = let colors=replace.(string.(QMRIColors.relaxationColorMap("T1").*255), "RGB{Float64}"=>"rgb"); collect(zip(range(0,1; length=length(colors)), colors)) end

function load_problem(input)
    metadata = read(joinpath(input, "metadata.txt"), String)
    occursin("coil_sensitivity_source = AdjCoilSens", metadata) || error("AdjCoilSens maps required")
    occursin("fully_sampled_data = none", metadata) || error("Fully sampled data is forbidden")
    sequence = read_seq(joinpath(input, "OpenMOLLI_LCD_S.seq"))
    adc_blocks = findall(i -> is_ADC_on(sequence[i]), eachindex(sequence.DUR))
    rr = 1f-3 .* Float32[1212.5, 1135, 1117.5, 1052.5, 925, 1082.5, 1195, 1032.5, 1165, 1177.5, 1010, 1030]
    sequence = resolve_triggers(sequence[1:last(adc_blocks)], CardiacSignal(; rr_intervals=rr, first_peak=100f-6))
    params = KomaMRICore.default_sim_params(Dict{String,Any}("sim_method"=>Bloch(), "gpu"=>false, "Nthreads"=>1, "return_type"=>"mat", "precision"=>"f32"))
    seqd = f32(discretize(sequence; sampling_rule=KomaMRICore.simulation_sampling_rule(Bloch(), params)))
    centers = [((c - 1) * PROFILES + CENTER_PROFILE - 1) * SAMPLES + CENTER_SAMPLE for c in 1:CONTRASTS]
    acquired = findall(seqd.ADC); adc = falses(length(seqd.ADC)); adc[acquired[centers]] .= true
    seqd = f32(DiscreteSequence(seqd.Gx, seqd.Gy, seqd.Gz, seqd.B1, seqd.Δf, seqd.ψ, adc, seqd.excitation_bool, seqd.t, seqd.Δt))
    parts, excitation = KomaMRICore.get_sim_ranges(seqd; max_block_length=params["max_block_length"], max_rf_block_length=params["max_rf_block_length"], eval_intervals_per_step=KomaMRICore.eval_intervals_per_step(Bloch()))
    max_adc = maximum(count(@view seqd.ADC[(first(part)+1):last(part)]) for part in parts; init=0)
    contrasts = Array{Float32}(undef, SIZE..., CONTRASTS); read!(joinpath(input, "signed_contrasts.f32"), contrasts)
    support = Array{UInt8}(undef, SIZE); read!(joinpath(input, "support.u8"), support)
    indices = findall(!iszero, vec(support)); signals = reshape(permutedims(contrasts, (3, 1, 2)), CONTRASTS, :)
    scanner = Scanner(; receiver=KomaMRICore.CoilSensitivities(Diagonal(ones(ComplexF32, GRID)), nothing))
    (; seqd, parts, excitation, max_adc, scanner, target=signals[:, indices], support=Bool.(support), indices)
end

function response(T1, p)
    object = Phantom(; x=zeros(Float32, GRID), y=zeros(Float32, GRID), z=zeros(Float32, GRID), ρ=ones(Float32, GRID), T1, T2=fill(0.05f0, GRID), T2s=fill(Float32(Inf), GRID))
    state, object = KomaMRICore.initialize_spins_state(object, Bloch())
    signal = zeros(ComplexF32, CONTRASTS, GRID, 1)
    KomaMRICore.run_sim_time_iter!(object, p.seqd, signal, state, p.scanner, Bloch(), KomaMRICore.KA.CPU(); Nblocks=length(p.parts), Nthreads=1, parts=p.parts, excitation_bool=p.excitation, max_adc_samples=p.max_adc)
    imag.(@view signal[:, :, 1])
end

function dictionary(p)
    T1 = collect(range(T1_MIN, T1_MAX; length=GRID))
    result = Enzyme.autodiff(MODE, response, Enzyme.Duplicated(T1, ones(Float32, GRID)), Enzyme.Const(p))
    orientation = sign(sum(@view result[2][5, :]))
    CuArray(orientation .* result[2]), CuArray(orientation .* result[1])
end

function fit_kernel!(T1, logρ, m1, v1, m2, v2, losses, target, table, derivative, iteration, n)
    pixel = (blockIdx().x - 1) * blockDim().x + threadIdx().x; pixel > n && return
    Δ = (T1_MAX - T1_MIN) / (GRID - 1); u = clamp((T1[pixel] - T1_MIN) / Δ + 1, 1f0, Float32(GRID))
    lo = min(floor(Int32, u), GRID - 1); α = u - lo; ρ = exp(logρ[pixel]); g1 = 0f0; g2 = 0f0; norm = floatmin(Float32)
    for c in 1:CONTRASTS
        y = target[c, pixel]; s = (1-α)*table[c,lo] + α*table[c,lo+1]; ds = (1-α)*derivative[c,lo] + α*derivative[c,lo+1]
        residual = ρ*s - y; norm += y*y; g1 += 2residual*ρ*ds; g2 += 2residual*ρ*s
    end
    if iteration > 0
        g1 /= norm; g2 /= norm; m1[pixel] = 0.9f0*m1[pixel] + 0.1f0*g1; m2[pixel] = 0.9f0*m2[pixel] + 0.1f0*g2
        v1[pixel] = 0.999f0*v1[pixel] + 0.001f0*g1*g1; v2[pixel] = 0.999f0*v2[pixel] + 0.001f0*g2*g2
        η = 0.02f0*sqrt(1f0-0.999f0^iteration)/(1f0-0.9f0^iteration)
        T1[pixel] = clamp(T1[pixel] - η*m1[pixel]/(sqrt(v1[pixel])+1f-8), T1_MIN, T1_MAX)
        logρ[pixel] = clamp(logρ[pixel] - η*m2[pixel]/(sqrt(v2[pixel])+1f-8), -30f0, 30f0)
    end
    losses[pixel] = 0f0; u = clamp((T1[pixel]-T1_MIN)/Δ+1, 1f0, Float32(GRID)); lo = min(floor(Int32,u), GRID-1); α = u-lo; ρ = exp(logρ[pixel])
    for c in 1:CONTRASTS
        residual = ρ*((1-α)*table[c,lo]+α*table[c,lo+1]) - target[c,pixel]; losses[pixel] += residual*residual/norm
    end
    return
end

function save_map(T1, p, iteration)
    map = zeros(Float32, prod(SIZE)); map[p.indices] .= Array(T1); map = reshape(map, SIZE)
    write(joinpath(OUTPUT, "iteration_$(lpad(iteration, 2, '0')).f32"), map)
    image = Matrix{Union{Missing,Float32}}(missing, SIZE); image[p.support] .= 1f3 .* map[p.support]
    figure = plot_image(rotr90(reverse(image; dims=(1,2))); title="OpenMOLLI T1 iteration $iteration (ms)", zmin=0, zmax=2000, colorscale=t1_colorscale(), width=750, height=900); figure.layout.paper_bgcolor="white"; figure.layout.plot_bgcolor="black"; figure.layout.xaxis[:visible]=false; figure.layout.yaxis[:visible]=false; savefig(figure, joinpath(OUTPUT, "iteration_$(lpad(iteration, 2, '0')).png"))
    map
end

function run_final_moli()
    mkpath(OUTPUT); p = load_problem(INPUT); table, derivative = dictionary(p); n = length(p.indices)
    target = CuArray(p.target); T1 = CUDA.ones(Float32, n); logρ = CuArray(log.(max.(maximum(abs, p.target; dims=1)[:], eps(Float32))))
    m1 = CUDA.zeros(Float32, n); v1 = CUDA.zeros(Float32, n); m2 = CUDA.zeros(Float32, n); v2 = CUDA.zeros(Float32, n); losses = similar(m1); rows = ["iteration,loss"]
    println("gpu=$(CUDA.name(CUDA.device())) pixels=$n dictionary=$GRID")
    for iteration in 0:ITERATIONS
        @cuda threads=256 blocks=cld(n,256) fit_kernel!(T1,logρ,m1,v1,m2,v2,losses,target,table,derivative,iteration,n); synchronize()
        value = sum(Array(losses))/n; push!(rows, "$iteration,$value"); println("iteration=$iteration loss=$value")
        iteration % 20 == 0 && save_map(T1, p, iteration)
    end
    t1map = save_map(T1, p, ITERATIONS); write(joinpath(OUTPUT, "quantitative_t1.f32"), t1map)
    cp(joinpath(OUTPUT, "iteration_80.png"), joinpath(OUTPUT, "t1_map.png"); force=true)
    write(joinpath(OUTPUT, "loss.csv"), join(rows, '\n') * "\n"); t1map
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_final_moli()
