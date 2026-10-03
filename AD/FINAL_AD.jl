using Enzyme, KomaMRI, KomaMRIPlots, Reactant
using MRICoilSensitivities: espirit
Reactant.set_default_backend("cpu")
Reactant.allowscalar(false)

centered_axis(width, count) =
    Float32.(range(-width / 2 + width / (2count); step=width / count, length=count))

function estimate_coils(path, measured, x, y; floor=1f-2, regularization=1f-4)
    raw = RawAcquisitionData(ISMRMRDFile(path))
    names = getproperty.(raw.params["coilLabel"], :name)
    order = [only(findall(==(name), names)) for name in getproperty.(measured.params["coilLabel"], :name)]
    readout = raw.params["encodedSize"][1]
    surface = filter(p -> p.head.active_channels == length(names) && p.head.number_of_samples == readout, raw.profiles)
    body = filter(p -> p.head.active_channels == 2 && p.head.number_of_samples == readout, raw.profiles)
    raw.profiles = surface
    ky = [p.head.idx.kspace_encode_step_1 for p in raw.profiles]
    kz = [p.head.idx.kspace_encode_step_2 for p in raw.profiles]
    raw.params["encodedSize"] = [readout, maximum(ky) + 1, maximum(kz) + 1]
    calibration = AcquisitionData(raw)
    body_calibration = AcquisitionData(RawAcquisitionData(raw.params, body))
    calibration.traj[1].circular = body_calibration.traj[1].circular = false
    map_size = (readout, raw.params["reconSize"][2], raw.params["reconSize"][3])
    maps = espirit(calibration, (4, 4, 4), 16, map_size; eigThresh_1=0.02, eigThresh_2=0.0)[:, :, :, order]
    direct = Dict{Symbol,Any}(:reco => "direct", :reconSize => map_size)
    image_rss(acq) = dropdims(sqrt.(sum(abs2, reconstruction(acq, direct); dims=5)); dims=(4, 5, 6))
    surface_rss, body_rss = image_rss(calibration), image_rss(body_calibration)
    scale = surface_rss .* body_rss ./ (body_rss .^ 2 .+ regularization * maximum(body_rss)^2)
    scale ./= maximum(scale)
    adjustment = first(raw.profiles).head
    imaging = first(measured.profiles).head
    basis = hcat(Float32[adjustment.read_dir...], Float32[adjustment.phase_dir...], Float32[adjustment.slice_dir...])
    position =
        Float32[imaging.position...] .+ 1000f0 .* Float32[imaging.read_dir...] .* x' .+
        1000f0 .* Float32[imaging.phase_dir...] .* y'
    coordinates = basis' * (position .- Float32[adjustment.position...])
    axes = Tuple(centered_axis.(Float32.(raw.params["encodedFOV"]), map_size))
    receiver = ArbitraryCoilSens(axes..., maps)
    values = ComplexF32.(get_sens(receiver, coordinates[1, :], coordinates[2, :], coordinates[3, :]))
    intensity = real.(get_sens(
        ArbitraryCoilSens(axes..., reshape(ComplexF32.(scale), map_size..., 1)),
        coordinates[1, :],
        coordinates[2, :],
        coordinates[3, :]
    ))
    rss = sqrt.(sum(abs2, values; dims=2))
    intensity .* values ./ max.(rss, floor * maximum(rss))
end

function load_problem(data_directory, image_size)
    measured_path = joinpath(data_directory, "brain_gre_3t_acc/meas_MID00492_FID42184_bssfp_optimized_2x.mrd")
    adjustment_path = joinpath(data_directory, "brain_gre_3t_acc/meas_MID00478_FID42170_AdjCoilSens.mrd")
    sequence_path = joinpath(data_directory, "bssfp_slice_all_adc_optimized_R2.seq")
    sequence = resolve_triggers(read_seq(sequence_path), CardiacSignal(; heart_rate=1))
    adc = findall(block -> is_ADC_on(sequence[block]), eachindex(sequence.DUR))
    measured = RawAcquisitionData(ISMRMRDFile(measured_path))
    length(measured.profiles) == length(adc) || error("Measured profiles do not match sequence ADCs")
    target = reduce(vcat, ComplexF32.(profile.data) for profile in measured.profiles)
    fov = Float32.(measured.params["reconFOV"]) .* 1f-3
    x_axis, y_axis = centered_axis.(fov[1:2], image_size)
    x = repeat(x_axis, image_size[2])
    y = repeat(y_axis; inner=image_size[1])
    z = zeros(Float32, prod(image_size))
    coils = estimate_coils(adjustment_path, measured, x, y)
    object = Phantom(;
        x=vcat(x, x),
        y=vcat(y, y),
        z=vcat(z, z),
        ρ=ones(Float32, 2length(x)),
        T1=ones(Float32, 2length(x)),
        T2=ones(Float32, 2length(x)),
        T2s=ones(Float32, 2length(x))
    )
    receiver = vcat(coils, ComplexF32(0, 1) .* coils)
    scanner = Scanner(; receiver=KomaMRICore.CoilSensitivities(Reactant.to_rarray(receiver), nothing))
    sim = Dict{String,Any}(
        "sim_method" => Bloch(),
        "gpu" => true,
        "Nthreads" => 1,
        "return_type" => "mat",
        "precision" => "f32"
    )
    (; object=Reactant.to_rarray(object), sequence=sequence[1:last(adc)], scanner, sim, b=Reactant.to_rarray(target))
end

function A(x, T1, T2, p)
    object = copy(p.object)
    object.T1 .= T1
    object.T2 .= T2
    object.ρ .= vcat(real.(x), imag.(x))
    simulate(object, p.sequence, p.scanner; sim_params=p.sim, verbose=false)[:, :, 1]
end

loss(x, T1, T2, p) = sum(abs2, A(x, T1, T2, p) - p.b)
function loss_gradient(x, T1, T2, p)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, loss, x, T1, T2, Enzyme.Const(p))
    result.val, result.derivs[1], result.derivs[2], result.derivs[3]
end

function run_final_ad(;
    image_size=(128, 128),
    iterations=80,
    step=2f-5,
    relaxation_step=1f3,
    data_directory=isempty(ARGS) ?
        joinpath(homedir(), "Desktop/Data/cleaner brain data acq 21 august") : first(ARGS),
    output_directory=joinpath(@__DIR__, "FINAL_AD")
)
    mkpath(output_directory)
    p = load_problem(data_directory, image_size)
    x = Reactant.to_rarray(zeros(ComplexF32, prod(image_size)))
    T1, T2 = Reactant.to_rarray(Float32[1]), Reactant.to_rarray(Float32[1])
    gradient = Reactant.@allowscalar Reactant.compile(loss_gradient, (x, T1, T2, p); sync=true)
    rows = ["iteration,loss,T1,T2"]
    for iteration in 0:iterations
        value, ∇x, ∇T1, ∇T2 = gradient(x, T1, T2, p)
        value, t1, t2 = Reactant.to_number(value), only(Array(T1)), only(Array(T2))
        push!(rows, "$iteration,$value,$t1,$t2")
        image = reverse(reshape(abs.(Array(x)), image_size); dims=1)
        savefig(
            plot_image(image; title="AD iteration $iteration"),
            joinpath(output_directory, "iteration_$(lpad(iteration, 2, '0')).png")
        )
        println("iteration=$iteration loss=$value T1=$t1 T2=$t2")
        iteration == iterations && continue
        x, T1, T2 = x .- step .* ∇x,
            max.(T1 .- relaxation_step .* ∇T1, eps(Float32)),
            max.(T2 .- relaxation_step .* ∇T2, eps(Float32))
    end
    write(joinpath(output_directory, "loss.csv"), join(rows, '\n') * "\n")
    Array(x)
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_final_ad()
