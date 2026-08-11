using FiniteDiff, KomaMRI, MRICoilSensitivities # Loads finite differences, simulation, reconstruction, and ESPIRiT.
using Interpolations: Flat, linear_interpolation # Interpolates voxel density onto simulation spins.
using KomaMRIBase: ArbitraryCoilSens, Phantom # Builds measured coils and the simulated object.
using LinearAlgebra: norm # Measures gradient and iterate changes.
@eval using $(Sys.isapple() ? :Metal : :CUDA) # Selects the available GPU package.

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count) # Places voxel centers symmetrically in the FOV.

function run_finite_full_reconstruction(voxel_grid) # Runs one requested finite-difference resolution.
    archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS) # Uses the default archive unless one is supplied.
    fully_sampled_mrd_file = joinpath(archive_directory, "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd") # Locates ESPIRiT calibration data.
    sequence_file = joinpath(archive_directory, "seq/hard_epi_20interleaves_5avg_fatsat.seq") # Locates the matching fully sampled sequence.
    resolution = first(voxel_grid) # Names outputs by the in-plane resolution.
    iteration_directory = joinpath(@__DIR__, "FiniteFull") # Keeps results beside FiniteDiff.
    mkpath(iteration_directory) # Creates the result folder if needed.

    recon_size = (128, 128) # Matches the acquired image matrix.
    subspin_grid = (5, 2, 1) # Resolves each unknown voxel with ten spins.
    simulation_spins_per_voxel = prod(subspin_grid) # Converts voxel density to per-spin density.
    navigator_count = 3 # Skips the three phase-correction navigator profiles.
    brain_T1 = 1.0 # Sets the simulated longitudinal relaxation time in seconds.
    brain_T2 = 0.1 # Sets the simulated transverse relaxation time in seconds.
    brain_T2s = 0.05 # Sets the simulated effective transverse relaxation time in seconds.
    finite_difference_step = cbrt(eps(Float32)) # Uses the standard first-derivative Float32 step scale.
    maximum_iterations = 20 # Bounds reconstruction work and saved frames.
    λ₀ = 1e-6 # Sets the first gradient-descent step.
    proton_density_range = (0f0, 1.2f-4) # Fixes black at zero and one scale for every image.

    seq = read_seq(sequence_file) # Loads the exact physical pulse sequence.
    acquired_lines = 0:(recon_size[2] - 1) # Uses every phase-encode line in the fully sampled acquisition.

    raw_measured = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file)) # Loads raw fully sampled multicoil profiles.
    image_profile_indices = (navigator_count + 1):(navigator_count + length(acquired_lines)) # Separates image profiles from navigators once.
    image_profiles = raw_measured.profiles[image_profile_indices] # Keeps only profiles used by the density loss.
    b = ComplexF32.(reduce(vcat, profile.data for profile in image_profiles)) # Stacks measured samples by time and coil without scaling.
    image_sample_offset = navigator_count * size(first(image_profiles).data, 1) # Counts simulated navigator samples preceding image data.
    image_sample_indices = (image_sample_offset + 1):(image_sample_offset + size(b, 1)) # Selects matching simulated image samples.
    adc_blocks = findall(block -> is_ADC_on(seq[block]), 1:length(seq)) # Finds sequence blocks that acquire data.
    seq = seq[1:adc_blocks[navigator_count + length(acquired_lines)]] # Stops after the final measured image readout.

    raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file)) # Loads fully sampled calibration profiles.
    raw_reference.profiles = raw_reference.profiles[(navigator_count + 1):(navigator_count + recon_size[2])] # Removes reference navigators and keeps 128 lines.
    acq_reference = AcquisitionData(raw_reference) # Converts raw profiles to the ESPIRiT input representation.
    acq_reference.traj[1].circular = false # Prevents a circular mask from discarding Cartesian corners.
    sensitivity_maps = espirit(acq_reference, (6, 6), 30, recon_size; eigThresh_1=0.02, eigThresh_2=0.0) # Estimates complex coil eigenvector maps from calibration data.
    sensitivity_maps ./= max.(sqrt.(sum(abs2, sensitivity_maps; dims=4)), eps(Float32)) # Enforces sum_c abs2(S_c)=1 at each supported pixel.

    map_fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3 # Converts the reference FOV from millimetres to metres.
    map_x = collect(LinRange(-map_fov[1] / 2, map_fov[1] / 2, recon_size[1])) # Assigns x coordinates to sensitivity pixels.
    map_y = collect(LinRange(-map_fov[2] / 2, map_fov[2] / 2, recon_size[2])) # Assigns y coordinates to sensitivity pixels.
    map_z = Float32[-map_fov[3] / 2, 0, map_fov[3] / 2] # Supplies a three-plane z grid for 2D maps.
    receiver = ArbitraryCoilSens(map_x, map_y, map_z, repeat(sensitivity_maps, 1, 1, length(map_z), 1)) # Makes the normalized complex maps the receive model.
    scanner = Scanner(; receiver) # Inserts measured receive coils into KomaMRI.
    fov = Float64.(map_fov) # Matches Phantom coordinate precision.

    voxel_count = prod(voxel_grid) # Counts optimized real density parameters.
    voxel_axis_x, voxel_axis_y = centered_axis.(fov[1:2], voxel_grid[1:2]) # Defines unknown-density voxel centers.
    simulation_grid = voxel_grid .* subspin_grid # Defines the finer physical spin grid.
    simulation_axes = centered_axis.(fov, simulation_grid) # Places all simulation spins in the FOV.
    spin_points = vec(collect(Iterators.product(simulation_axes...))) # Enumerates Cartesian spin coordinates.
    spin_x = first.(spin_points) # Extracts spin x coordinates.
    spin_y = getindex.(spin_points, 2) # Extracts spin y coordinates.
    spin_z = last.(spin_points) # Extracts spin z coordinates.

    sim_params = Dict{String,Any}("gpu" => true, "precision" => "f32", "sim_method" => Bloch(), "return_type" => "mat") # Requests complex Float32 GPU Bloch signals.
    physio = CardiacSignal(; heart_rate=1) # Supplies deterministic trigger timing required by the sequence.
    params = (; seq, scanner, sim_params, physio, b, image_sample_indices, voxel_grid, voxel_axis_x, voxel_axis_y, spin_x, spin_y, spin_z, simulation_spins_per_voxel, brain_T1, brain_T2, brain_T2s, finite_difference_step, maximum_iterations, λ₀, proton_density_range, iteration_directory, object_name="$(resolution) x $(resolution) fully sampled in-vivo brain reconstruction") # Groups fixed forward-model and optimizer inputs.

    function simulation_object(x) # Converts one density guess into a KomaMRI Phantom.
        interpolation = linear_interpolation((params.voxel_axis_x, params.voxel_axis_y), reshape(x, params.voxel_grid[1:2]); extrapolation_bc=Flat()) # Interpolates the coarse density grid continuously.
        spin_count = length(params.spin_x) # Sizes every spin property consistently.
        return Phantom(; name=params.object_name, x=params.spin_x, y=params.spin_y, z=params.spin_z, ρ=interpolation.(params.spin_x, params.spin_y) ./ params.simulation_spins_per_voxel, T1=fill(params.brain_T1, spin_count), T2=fill(params.brain_T2, spin_count), T2s=fill(params.brain_T2s, spin_count)) # Builds spins whose summed density equals the voxel guess.
    end # Completes Phantom construction.

    function simulate_image(x) # Evaluates the direct physical multicoil forward model.
        signal = simulate(simulation_object(x), params.seq, params.scanner; sim_params=params.sim_params, physio=params.physio, verbose=false) # Simulates the guessed object through sequence and coils.
        return @view signal[params.image_sample_indices, :] # Excludes simulated navigator samples from the loss.
    end # Completes the physical forward evaluation.

    save_density_image(x, name, title) = savefig(plot_image(reshape(x, params.voxel_grid[1:2]); title, zmin=first(params.proton_density_range), zmax=last(params.proton_density_range)), joinpath(params.iteration_directory, name)) # Saves every iterate on the same zero-based scale.

    function reconstruct(x) # Minimizes measured-versus-simulated squared error.
        objective(x) = sum(abs2, simulate_image(x) - params.b) # Defines f(x)=norm(A(x)-b)^2 without data scaling.
        save_density_image(x, "iteration_00.png", "Iteration 0") # Records the zero-density initialization.
        ∇fₖ = FiniteDiff.finite_difference_gradient(objective, x; relstep=params.finite_difference_step) # Estimates the first full numerical gradient.
        xₖ₋₁ = copy(x) # Stores the previous density for step adaptation.
        ∇fₖ₋₁ = ∇fₖ # Stores the previous gradient for step adaptation.
        λₖ = params.λ₀ # Starts from the configured step length.
        θₖ = Inf # Allows the first adaptive upper bound to be unconstrained.

        for iteration in 1:params.maximum_iterations # Performs the requested optimization iterations.
            if iteration > 1 # Reuses the already computed gradient only in iteration one.
                ∇fₖ = FiniteDiff.finite_difference_gradient(objective, x; relstep=params.finite_difference_step) # Recomputes the numerical gradient at the current density.
                Δxₖ = norm(x - xₖ₋₁) # Measures the latest density displacement.
                Δ∇fₖ = norm(∇fₖ - ∇fₖ₋₁) # Measures the latest gradient displacement.
                λₖ₋₁, θₖ₋₁ = λₖ, θₖ # Preserves the previous adaptive state.
                λₖ = min(sqrt(1 + θₖ₋₁) * λₖ₋₁, iszero(Δ∇fₖ) ? λₖ₋₁ : Δxₖ / (2Δ∇fₖ)) # Applies the reference adaptive step rule.
                θₖ = λₖ / λₖ₋₁ # Records step growth for the next bound.
            end # Completes adaptive step selection.

            ∇fₖ_norm = norm(∇fₖ) # Measures convergence and reports gradient size.
            iszero(∇fₖ_norm) && break # Stops if the objective is stationary.
            xₖ₋₁ .= x # Saves the current density before updating it.
            ∇fₖ₋₁ = ∇fₖ # Saves the current gradient before replacing it.
            x .-= λₖ .* ∇fₖ # Takes one gradient-descent step.

            save_density_image(x, "iteration_$(lpad(iteration, 2, '0')).png", "Iteration $iteration") # Saves the updated density with a sortable name.
            println("Iteration $iteration: loss = $(objective(x)), gradient = $∇fₖ_norm, λ = $λₖ") # Reports numerical progress.
        end # Completes all finite-difference iterations.

        save_density_image(x, "reconstructed_density.png", "Reconstructed brain density") # Saves the final density on the shared scale.
        println("Final loss: ", objective(x)) # Reports final data inconsistency.
        println("Density extrema: ", extrema(x)) # Reveals negative values and display saturation.
        println("Saved images to: ", params.iteration_directory) # Reports the exact output folder.
        return x # Returns the optimized density vector.
    end # Completes finite-difference optimization.

    return reconstruct(zeros(voxel_count)) # Reconstructs from a zero-density initialization.
end # Completes one finite-difference resolution.
