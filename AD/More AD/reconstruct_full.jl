using Enzyme # Differentiates the real-density objective in reverse mode.
using KomaMRI # Supplies sequence physics, FFTs, files, scanners, and plotting.
using KomaMRIBase: ArbitraryCoilSens, Phantom, bilinear_stencil, get_sens, # Imports object, coil, and interpolation primitives.
    interpolate_voxels # Applies the same voxel-to-spin interpolation in AD.
using LinearAlgebra: dot, norm # Validates adjoints and measures optimization changes.
using MRICoilSensitivities # Estimates ESPIRiT sensitivity maps.
using Reactant # Compiles Enzyme gradients for the selected accelerator backend.
@eval using $(Sys.isapple() ? :Metal : :CUDA) # Selects the available KomaMRI GPU package.

Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", Sys.isapple() ? "cpu" : "cuda")) # Uses CPU Reactant on macOS and CUDA elsewhere by default.
Reactant.allowscalar(false) # Rejects accidental scalar operations on compiled arrays.

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count) # Places voxel centers symmetrically in the FOV.

function spin_object(density, params) # Converts per-spin density into a KomaMRI Phantom.
    return Phantom(; # Constructs the physical object passed to KomaMRI.
        name=params.object_name, # Identifies the reconstruction in simulator output.
        x=params.spin_x, # Supplies spin x coordinates in metres.
        y=params.spin_y, # Supplies spin y coordinates in metres.
        z=params.spin_z, # Supplies spin z coordinates in metres.
        ρ=density, # Assigns the current proton-density guess to spins.
        T1=fill(params.brain_T1, params.total_spin_count), # Gives every spin the selected T1.
        T2=fill(params.brain_T2, params.total_spin_count), # Gives every spin the selected T2.
        T2s=fill(params.brain_T2s, params.total_spin_count), # Gives every spin the selected T2*.
    ) # Finishes Phantom construction.
end # Completes the spin-object helper.

function koma_signal(x, params) # Evaluates KomaMRI for forward-model validation only.
    signal = simulate( # Runs the direct physical multicoil simulation.
        spin_object( # Builds spins from the validation density.
            interpolate_voxels(x, params.stencil) ./ params.simulation_spins_per_voxel, # Maps voxel density to density-conserving subspins.
            params, # Supplies coordinates and relaxation values.
        ), # Finishes the validation Phantom.
        params.seq, # Applies the measured fully sampled sequence.
        params.scanner; # Applies the normalized measured coil maps.
        sim_params=params.sim_params, # Uses the configured GPU Bloch simulation.
        physio=params.physio, # Resolves deterministic sequence triggers.
        verbose=false, # Avoids repeated simulator logs during checks.
    ) # Finishes the physical validation simulation.
    return @view signal[params.image_sample_indices, :] # Compares only image samples, not navigators.
end # Completes the KomaMRI validation forward.

function simulate_reference_response(params) # Captures sequence and relaxation weighting for the FFT model.
    reference_params = merge( # Reuses acquisition settings with one reference spin.
        params, # Starts from the reconstruction parameters.
        (; # Replaces only spatial spin fields.
            spin_x=zeros(Float64, 1), # Places the reference spin at x=0.
            spin_y=zeros(Float64, 1), # Places the reference spin at y=0.
            spin_z=zeros(Float64, 1), # Places the reference spin at z=0.
            total_spin_count=1, # Matches property arrays to one spin.
        ), # Finishes the reference-spin replacement.
    ) # Finishes the temporary parameter tuple.
    signal = simulate( # Simulates temporal weighting without receive coils.
        spin_object(ones(Float32, 1), reference_params), # Uses unit density so the result is a response factor.
        params.seq, # Preserves the full sequence history.
        Scanner(); # Removes spatial receive-coil weighting.
        sim_params=params.sim_params, # Uses the same Bloch method and precision.
        physio=params.physio, # Uses the same resolved trigger timing.
        verbose=false, # Suppresses one-time validation logs.
    ) # Finishes the reference simulation.
    return vec(Array(@view signal[params.image_sample_indices, :])) # Returns one complex weight per image sample.
end # Completes reference-response calculation.

function fft_encoding(params, receiver_values) # Builds the differentiable Cartesian encoding operator.
    settings = KomaMRICore.default_sim_params(copy(params.sim_params)) # Resolves KomaMRI simulation defaults consistently.
    sampling_rule = KomaMRICore.simulation_sampling_rule(settings["sim_method"], settings) # Matches trajectory sampling to the Bloch method.
    seq = KomaMRIBase.resolve_triggers(params.seq, params.physio) # Applies physiological timing before extracting k-space.
    _, trajectory = get_kspace(seq; sampling_rule) # Computes the sequence-defined k-space coordinates.
    trajectory = trajectory[params.image_sample_indices, :] # Keeps only measured image sample locations.
    modes_float = trajectory .* reshape(params.fov, 1, :) # Converts cycles per metre to discrete Fourier modes.
    modes = round.(Int, modes_float) # Maps nearly Cartesian coordinates to exact integer bins.
    mode_error = maximum(abs, modes_float - modes) # Quantifies the Cartesian rounding assumption.
    println("Cartesian trajectory mode error: ", mode_error) # Reports the trajectory validation value.
    mode_error < 5e-3 || error("Sequence trajectory is not Cartesian") # Rejects an invalid FFT factorization.

    Nx, Ny, Nz = params.simulation_grid # Names the three FFT grid sizes.
    sample_indices = @. 1 + mod(modes[:, 1], Nx) + # Maps x modes to linear FFT indices.
                        mod(modes[:, 2], Ny) * Nx + # Adds the y-mode stride.
                        mod(modes[:, 3], Nz) * Nx * Ny # Adds the z-mode stride.
    offsets = collect(1 .- inv.(Float64.((Nx, Ny, Nz)))) # Computes the voxel-center phase offsets of KomaMRI's grid.
    center_phase = exp.(ComplexF32(0, π) .* vec(sum(modes .* reshape(offsets, 1, :); dims=2))) # Aligns FFT phase with centered physical coordinates.
    sample_weight = ComplexF32.(simulate_reference_response(params) .* center_phase) # Combines relaxation, sequence, and centering phase.
    return (; # Packages the fixed linear operator.
        sample_indices, # Stores which FFT bin each ADC sample reads.
        sample_weight, # Stores complex temporal and centering weights.
        receiver_values, # Stores normalized complex coil sensitivities at spins.
        stencil=params.stencil, # Stores voxel-to-spin interpolation weights.
        simulation_grid=params.simulation_grid, # Stores FFT reshape dimensions.
        simulation_spins_per_voxel=params.simulation_spins_per_voxel, # Stores density-conservation scaling.
        voxel_count=params.voxel_count, # Sizes the adjoint output.
    ) # Finishes the encoding model.
end # Completes differentiable encoding construction.

function fft_signal(x, model) # Applies the differentiable forward operator A(x).
    spin_density = interpolate_voxels(x, model.stencil) ./ # Interpolates coarse density onto simulation spins.
                   model.simulation_spins_per_voxel # Preserves total density when using subspins.
    spin_coils = reshape(spin_density, :, 1) .* model.receiver_values # Multiplies density by each complex forward coil map.
    grid_coils = reshape(spin_coils, model.simulation_grid..., size(spin_coils, 2)) # Forms one Cartesian volume per receive coil.
    spectrum = KomaMRI.fft(grid_coils, (1, 2, 3)) # Fourier-encodes every coil volume.
    sampled = reshape(spectrum, :, size(spin_coils, 2))[model.sample_indices, :] # Reads bins in sequence ADC order.
    return reshape(model.sample_weight, :, 1) .* sampled # Applies sequence relaxation and phase response.
end # Completes the differentiable forward operator.

function interpolation_adjoint(spin_values, stencil, voxel_count) # Applies the transpose of bilinear voxel interpolation.
    voxel_values = zeros(eltype(spin_values), voxel_count) # Initializes accumulated voxel contributions.
    for spin in eachindex(spin_values) # Visits every simulated spin once.
        wx = stencil.wx[spin] # Reads the spin's fractional x position.
        wy = stencil.wy[spin] # Reads the spin's fractional y position.
        voxel_values[stencil.i00[spin]] += (1 - wx) * (1 - wy) * spin_values[spin] # Returns the lower-left contribution.
        voxel_values[stencil.i10[spin]] += wx * (1 - wy) * spin_values[spin] # Returns the lower-right contribution.
        voxel_values[stencil.i01[spin]] += (1 - wx) * wy * spin_values[spin] # Returns the upper-left contribution.
        voxel_values[stencil.i11[spin]] += wx * wy * spin_values[spin] # Returns the upper-right contribution.
    end # Completes all spin accumulations.
    return voxel_values # Returns values on the unknown-density grid.
end # Completes the interpolation adjoint.

function fft_adjoint(signal, model) # Applies the Hermitian encoding operator A^H.
    spectrum = zeros(eltype(signal), prod(model.simulation_grid), size(signal, 2)) # Initializes one FFT grid per coil.
    weighted_signal = conj.(model.sample_weight) .* signal # Conjugates forward temporal weights for the adjoint.
    for sample in axes(signal, 1) # Visits every acquired sample.
        spectrum[model.sample_indices[sample], :] .+= @view weighted_signal[sample, :] # Scatter-adds repeated samples into FFT bins.
    end # Completes k-space scattering.
    grid_spectrum = reshape(spectrum, model.simulation_grid..., size(signal, 2)) # Restores Cartesian coil grids.
    spin_coils = prod(model.simulation_grid) .* KomaMRI.ifft(grid_spectrum, (1, 2, 3)) # Applies the adjoint of Julia's unnormalized FFT.
    receiver_grid = reshape( # Restores the spatial coil-map grid.
        model.receiver_values, # Uses the same maps as the forward model.
        model.simulation_grid..., # Uses the simulation spatial dimensions.
        size(model.receiver_values, 2), # Retains the receive-coil dimension.
    ) # Finishes coil-map reshaping.
    spin_adjoint = vec(sum(conj.(receiver_grid) .* spin_coils; dims=4)) # Conjugates coil maps and sums all coil evidence.
    return interpolation_adjoint(spin_adjoint, model.stencil, model.voxel_count) ./ # Returns spin evidence to density voxels.
           model.simulation_spins_per_voxel # Matches the forward density-conservation factor.
end # Completes the Hermitian encoding operator.

A(x, model) = fft_signal(x, model) # Names the direct real-density forward prediction.

f(x, model) = sum(abs2, A(x, model) - model.b) # Defines f(x)=norm(A(x)-b)^2 without scaling or calibration.

function f_gradient(x, model) # Computes loss and reverse-mode density gradient together.
    result = Enzyme.gradient( # Differentiates the direct objective.
        Enzyme.ReverseWithPrimal, # Requests both primal loss and reverse gradient.
        f, # Differentiates the squared residual function.
        x, # Marks real density as the active variable.
        Enzyme.Const(model), # Treats acquisition and coil data as fixed.
    ) # Finishes the Enzyme request.
    return result.val, result.derivs[1] # Returns loss and the density gradient.
end # Completes the AD gradient helper.

save_density_image(x, name, title, params) = savefig(plot_image(reshape(x, params.voxel_grid[1:2]); title, zmin=first(params.proton_density_range), zmax=last(params.proton_density_range)), joinpath(params.iteration_directory, name)) # Saves every iterate on the same zero-based scale.

function reconstruct(x, controls, model, gradient_function, loss_function, initial_state) # Minimizes the compiled AD objective.
    save_density_image(Array(x), "iteration_00.png", "Iteration 0", controls) # Records the zero-density initialization.
    loss, gradient = initial_state # Reuses the compiled initial loss and gradient.
    println("Initial loss: ", Reactant.to_number(loss)) # Reports zero-density data inconsistency.
    λₖ = controls.λ₀ # Starts from the configured step length.
    θₖ = Float32(Inf) # Allows the first adaptive upper bound to be unconstrained.

    for iteration in 1:controls.maximum_iterations # Performs the requested optimization iterations.
        gradient_norm = norm(Array(gradient)) # Measures convergence and reports gradient size.
        iszero(gradient_norm) && break # Stops if the objective is stationary.
        previous_x = x # Preserves the current density for step adaptation.
        previous_gradient = gradient # Preserves the current gradient for step adaptation.
        x = x .- λₖ .* gradient # Takes one compiled gradient-descent step.

        if iteration < controls.maximum_iterations # Avoids an unused final gradient evaluation.
            loss, gradient = gradient_function(x, model) # Evaluates the next loss and reverse gradient.
        else # Handles the final iteration separately.
            loss = loss_function(x, model) # Evaluates only the final loss.
        end # Completes post-update evaluation.
        save_density_image( # Saves the current reconstructed density.
            Array(x), # Transfers the compiled density to a normal Julia array.
            "iteration_$(lpad(iteration, 2, '0')).png", # Gives the frame a sortable name.
            "Iteration $iteration", # Labels the frame with its iteration.
            controls, # Supplies grid, directory, and fixed display range.
        ) # Finishes saving the iteration image.
        println( # Reports numerical progress.
            "Iteration $iteration: loss = $(Reactant.to_number(loss)), " * # Prints the current data loss.
            "gradient = $gradient_norm, λ = $λₖ", # Prints gradient norm and step size.
        ) # Finishes the progress message.

        if iteration < controls.maximum_iterations # Updates the step only when another iteration remains.
            Δxₖ = norm(Array(x - previous_x)) # Measures the latest density displacement.
            Δ∇fₖ = norm(Array(gradient - previous_gradient)) # Measures the latest gradient displacement.
            λₖ₋₁, θₖ₋₁ = λₖ, θₖ # Preserves the previous adaptive state.
            λₖ = min( # Chooses the stable adaptive step.
                sqrt(1 + θₖ₋₁) * λₖ₋₁, # Limits growth relative to the previous step.
                iszero(Δ∇fₖ) ? λₖ₋₁ : Δxₖ / (2Δ∇fₖ), # Uses local inverse gradient variation when defined.
            ) # Finishes adaptive step selection.
            θₖ = λₖ / λₖ₋₁ # Records step growth for the next bound.
        end # Completes step adaptation.
    end # Completes all AD iterations.

    density = Array(x) # Transfers the optimized density to the host.
    save_density_image(density, "reconstructed_density.png", "Reconstructed brain density", controls) # Saves the final density on the shared scale.
    println("Final loss: ", Reactant.to_number(loss)) # Reports final data inconsistency.
    println("Density extrema: ", extrema(density)) # Reveals negative values and display saturation.
    println("Saved images to: ", controls.iteration_directory) # Reports the exact output folder.
    return x # Returns the optimized compiled density array.
end # Completes AD optimization.

function run_ad_full_reconstruction(voxel_grid; output_directory="ADFull") # Builds and runs one requested AD resolution.
    archive_directory = # Selects the acquisition archive.
        isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS) # Uses the default archive unless one is supplied.
    fully_sampled_mrd_file = joinpath( # Builds the calibration-data path.
        archive_directory, # Uses the selected archive root.
        "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd", # Selects the fully sampled reference MRD.
    ) # Finishes the reference path.
    sequence_file = # Builds the sequence path.
        joinpath(archive_directory, "seq/hard_epi_20interleaves_5avg_fatsat.seq") # Selects the matching fully sampled sequence.
    resolution = first(voxel_grid) # Names outputs by the in-plane resolution.
    iteration_directory = joinpath(@__DIR__, output_directory) # Keeps results in the launcher's selected folder beside AD.
    mkpath(iteration_directory) # Creates the result folder if needed.

    recon_size = (128, 128) # Matches the acquired image matrix.
    subspin_grid = (5, 2, 1) # Resolves each unknown voxel with ten spins.
    simulation_spins_per_voxel = prod(subspin_grid) # Converts voxel density to per-spin density.
    navigator_count = 3 # Skips the three phase-correction navigator profiles.
    brain_T1 = 1.0 # Sets the simulated longitudinal relaxation time in seconds.
    brain_T2 = 0.1 # Sets the simulated transverse relaxation time in seconds.
    brain_T2s = 0.05 # Sets the simulated effective transverse relaxation time in seconds.
    maximum_iterations = 20 # Bounds reconstruction work and saved frames.
    λ₀ = 1f-6 # Sets the first gradient-descent step.
    proton_density_range = (0f0, 2f-5) # Matches the display scale used by the other AD reconstructions.

    seq = read_seq(sequence_file) # Loads the exact physical pulse sequence.
    acquired_lines = 0:(recon_size[2] - 1) # Uses every phase-encode line in the fully sampled acquisition.

    raw_measured = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file)) # Loads raw fully sampled multicoil profiles.
    image_profile_indices = # Separates image profiles from navigators once.
        (navigator_count + 1):(navigator_count + length(acquired_lines)) # Selects only acquired image profiles.
    image_profiles = raw_measured.profiles[image_profile_indices] # Keeps profiles used by the density loss.
    b = ComplexF32.(reduce(vcat, profile.data for profile in image_profiles)) # Stacks measured samples by time and coil without scaling.
    image_sample_offset = navigator_count * size(first(image_profiles).data, 1) # Counts simulated navigator samples preceding image data.
    image_sample_indices = # Defines matching simulated image samples.
        (image_sample_offset + 1):(image_sample_offset + size(b, 1)) # Skips the simulated navigator samples.
    adc_blocks = findall(block -> is_ADC_on(seq[block]), 1:length(seq)) # Finds sequence blocks that acquire data.
    seq = seq[1:adc_blocks[navigator_count + length(acquired_lines)]] # Stops after the final measured image readout.

    raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file)) # Loads fully sampled calibration profiles.
    raw_reference.profiles = # Restricts calibration data to image lines.
        raw_reference.profiles[(navigator_count + 1):(navigator_count + recon_size[2])] # Removes three reference navigators and keeps 128 lines.
    acq_reference = AcquisitionData(raw_reference) # Converts raw profiles to ESPIRiT input data.
    acq_reference.traj[1].circular = false # Prevents a circular mask from discarding Cartesian corners.
    sensitivity_maps = espirit( # Estimates complex coil eigenvector maps.
        acq_reference, # Uses fully sampled calibration data.
        (6, 6), # Uses a 6 by 6 calibration kernel.
        30, # Crops the central 30 by 30 k-space calibration region.
        recon_size; # Returns maps on the 128 by 128 grid.
        eigThresh_1=0.02, # Rejects weak calibration-matrix modes.
        eigThresh_2=0.0, # Disables the usual 0.95 spatial eigenvalue support mask.
    ) # Finishes ESPIRiT calibration.
    sensitivity_maps ./= max.(sqrt.(sum(abs2, sensitivity_maps; dims=4)), eps(Float32)) # Enforces sum_c abs2(S_c)=1 at each supported pixel.

    map_fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3 # Converts the reference FOV from millimetres to metres.
    map_x = collect(LinRange(-map_fov[1] / 2, map_fov[1] / 2, recon_size[1])) # Assigns x coordinates to sensitivity pixels.
    map_y = collect(LinRange(-map_fov[2] / 2, map_fov[2] / 2, recon_size[2])) # Assigns y coordinates to sensitivity pixels.
    map_z = Float32[-map_fov[3] / 2, 0, map_fov[3] / 2] # Supplies a three-plane z grid for 2D maps.
    receiver = ArbitraryCoilSens( # Builds an interpolated receive-coil model.
        map_x, # Supplies coil-map x coordinates.
        map_y, # Supplies coil-map y coordinates.
        map_z, # Supplies coil-map z coordinates.
        repeat(sensitivity_maps, 1, 1, length(map_z), 1), # Repeats identical 2D maps across the thin z dimension.
    ) # Finishes the measured receiver model.
    scanner = Scanner(; receiver) # Inserts normalized measured receive coils into KomaMRI.
    fov = Float64.(map_fov) # Matches Phantom coordinate precision.

    voxel_count = prod(voxel_grid) # Counts optimized real density parameters.
    voxel_axis_x, voxel_axis_y = centered_axis.(fov[1:2], voxel_grid[1:2]) # Defines unknown-density voxel centers.
    simulation_grid = voxel_grid .* subspin_grid # Defines the finer physical spin grid.
    simulation_axes = centered_axis.(fov, simulation_grid) # Places all simulation spins in the FOV.
    spin_points = vec(collect(Iterators.product(simulation_axes...))) # Enumerates Cartesian spin coordinates.
    spin_x = first.(spin_points) # Extracts spin x coordinates.
    spin_y = getindex.(spin_points, 2) # Extracts spin y coordinates.
    spin_z = last.(spin_points) # Extracts spin z coordinates.
    total_spin_count = length(spin_points) # Sizes all spin-property arrays.
    object_name = "$(resolution) x $(resolution) fully sampled in-vivo brain reconstruction" # Labels the validation Phantom.
    physio = CardiacSignal(; heart_rate=1) # Supplies deterministic trigger timing required by the sequence.
    sim_params = Dict{String,Any}( # Configures direct KomaMRI validation.
        "gpu" => true, # Runs Bloch validation on the available GPU.
        "precision" => "f32", # Matches compiled AD precision.
        "sim_method" => Bloch(), # Uses the full Bloch forward method.
        "return_type" => "mat", # Returns samples by coil as a matrix.
    ) # Finishes simulation configuration.
    stencil = bilinear_stencil(voxel_axis_x, voxel_axis_y, spin_x, spin_y) # Precomputes voxel-to-spin interpolation indices and weights.
    stencil = merge(stencil, (; wx=Float32.(stencil.wx), wy=Float32.(stencil.wy))) # Matches interpolation weights to AD precision.
    receiver_values = ComplexF32.(get_sens(receiver, spin_x, spin_y, spin_z)) # Interpolates normalized complex coil maps at every spin.
    params = (; # Packages fixed validation and encoding inputs.
        seq, # Stores the measured pulse sequence.
        scanner, # Stores the normalized multicoil scanner.
        sim_params, # Stores KomaMRI validation settings.
        physio, # Stores resolved trigger input.
        b, # Stores unscaled measured multicoil image data.
        image_sample_indices, # Stores the image-only simulation range.
        voxel_grid, # Stores unknown-density dimensions.
        voxel_axis_x, # Stores density x coordinates.
        voxel_axis_y, # Stores density y coordinates.
        spin_x, # Stores simulation spin x coordinates.
        spin_y, # Stores simulation spin y coordinates.
        spin_z, # Stores simulation spin z coordinates.
        total_spin_count, # Stores the number of simulated spins.
        simulation_spins_per_voxel, # Stores density-conservation scaling.
        brain_T1, # Stores the chosen T1.
        brain_T2, # Stores the chosen T2.
        brain_T2s, # Stores the chosen T2*.
        maximum_iterations, # Stores the optimization limit.
        λ₀, # Stores the initial step.
        iteration_directory, # Stores the result path.
        object_name, # Stores the Phantom label.
        stencil, # Stores voxel-to-spin interpolation.
        simulation_grid, # Stores FFT grid dimensions.
        fov, # Stores physical dimensions in metres.
        voxel_count, # Stores the number of unknowns.
    ) # Finishes the fixed parameter tuple.

    fft_model = fft_encoding(params, receiver_values) # Builds the differentiable encoding from sequence and coils.

    probe = Float32.(range(0.1, 1; length=voxel_count)) # Creates a nontrivial real validation density.
    reference_signal = Array(koma_signal(probe, params)) # Computes its direct KomaMRI signal.
    fft_probe = fft_signal(ComplexF32.(probe), fft_model) # Computes the differentiable model signal.
    relative_forward_error = norm(fft_probe - reference_signal) / # Measures disagreement with direct KomaMRI.
                             max(norm(reference_signal), eps(Float32)) # Normalizes error safely.
    println("FFT forward relative error: ", relative_forward_error) # Reports physical-forward agreement.
    relative_forward_error < 5e-4 || error("FFT forward does not match KomaMRI") # Rejects an inaccurate differentiable model.

    forward_inner_product = dot(fft_probe, b) # Computes the forward side of the adjoint identity.
    adjoint_inner_product = dot(ComplexF32.(probe), fft_adjoint(b, fft_model)) # Computes the Hermitian side using conjugated coils.
    relative_adjoint_error = abs(forward_inner_product - adjoint_inner_product) / # Measures failure of dot(Ax,b)=dot(x,A^H b).
                             max(abs(forward_inner_product), eps(Float32)) # Normalizes the identity error safely.
    println("FFT adjoint relative error: ", relative_adjoint_error) # Reports adjoint correctness.
    relative_adjoint_error < 5e-5 || error("FFT adjoint does not match its forward") # Rejects an incorrect Hermitian operator.

    controls = (; voxel_grid, maximum_iterations, λ₀, proton_density_range, iteration_directory) # Keeps host-side optimization and plotting settings.
    model_device = (; # Transfers the differentiable operator to Reactant.
        sample_indices=Reactant.to_rarray(fft_model.sample_indices), # Transfers ADC-to-FFT bin indices.
        sample_weight=Reactant.to_rarray(fft_model.sample_weight), # Transfers temporal and phase weights.
        receiver_values=Reactant.to_rarray(fft_model.receiver_values), # Transfers normalized complex coil maps.
        stencil=map(Reactant.to_rarray, fft_model.stencil), # Transfers interpolation indices and weights.
        simulation_grid=fft_model.simulation_grid, # Keeps static FFT dimensions.
        simulation_spins_per_voxel=fft_model.simulation_spins_per_voxel, # Keeps density-conservation scaling.
        b=Reactant.to_rarray(b), # Transfers raw measured data without scaling.
    ) # Finishes the compiled model tuple.
    x = Reactant.to_rarray(zeros(Float32, voxel_count)) # Initializes real density at zero.
    compiled_gradient = Reactant.@compile sync=true f_gradient(x, model_device) # Compiles reverse loss and gradient evaluation.
    compiled_loss = Reactant.@compile sync=true f(x, model_device) # Compiles final loss-only evaluation.
    initial_state = compiled_gradient(x, model_device) # Evaluates zero-density loss and gradient.
    analytic_gradient = -2 .* real.(fft_adjoint(b, fft_model)) # Computes 2Re(A^H(Ax-b)) at x=0.
    relative_gradient_error = norm(Array(last(initial_state)) - analytic_gradient) / # Compares Enzyme with the exact linear-model gradient.
                              max(norm(analytic_gradient), eps(Float32)) # Normalizes gradient error safely.
    println("AD gradient relative error: ", relative_gradient_error) # Reports differentiation correctness.
    relative_gradient_error < 5e-4 || error("AD gradient does not match FFT adjoint") # Rejects an incorrect AD gradient.
    return reconstruct( # Starts compiled density optimization.
        x, # Supplies zero-density initialization.
        controls, # Supplies iteration and plotting controls.
        model_device, # Supplies the compiled forward operator and data.
        compiled_gradient, # Supplies compiled loss-plus-gradient evaluation.
        compiled_loss, # Supplies compiled final loss evaluation.
        initial_state, # Reuses the validated initial loss and gradient.
    ) # Returns the optimized density.
end # Completes one AD resolution.
