module PrepareOpenMOLLIADInputs

using FFTW
using KomaMRI
using LinearAlgebra: I, tr
using SHA: sha256
using Statistics: quantile

const Twix = KomaMRI.KomaMRIFiles.MRIFiles
const DMA_LENGTH_MASK = UInt32(0x01ffffff)
const READOUT_SAMPLES = 192
const ACQUIRED_PROFILES = 95
const CONTRASTS = 8
const COILS = 30
const STARTUP_PROFILES = 11
const FULL_PHASE_LINES = 144
const OUTPUT_SIZE = (192, 144)
const GRAPPA_READOUT_OFFSETS = -2:2
const GRAPPA_PHASE_OFFSETS = (-1, 1)
const GRAPPA_REGULARIZATION = 1f-3

read_struct(io, T) = read!(io, Ref{T}())[]

function raid_entries(filename)
    open(filename) do io
        header = read_struct(io, Twix.MrParcRaidFileHeader)
        [read_struct(io, Twix.MrParcRaidFileEntry) for _ in 1:header.count_]
    end
end

function data_scans(filename, entry, samples_per_profile, coil_count)
    scans = NamedTuple[]
    open(filename) do io
        seek(io, entry.off_)
        header_length = read(io, UInt32)
        seek(io, entry.off_ + header_length)
        measurement_end = entry.off_ + entry.len_
        while position(io) + sizeof(Twix.sScanHeader) <= measurement_end
            scan_start = position(io)
            header = read_struct(io, Twix.sScanHeader)
            dma_length = Int(header.ulFlagsAndDMALength & DMA_LENGTH_MASK)
            dma_length < sizeof(Twix.sScanHeader) && break
            flags = UInt64(header.aulEvalInfoMask[1]) |
                (UInt64(header.aulEvalInfoMask[2]) << 32)
            samples = Int(header.ushSamplesInScan)
            channels = Int(header.ushUsedChannels)
            if samples == samples_per_profile && channels == coil_count &&
                (flags & 0x8) != 0
                data = Matrix{ComplexF32}(undef, samples, channels)
                channel_ids = Vector{Int}(undef, channels)
                for channel in 1:channels
                    channel_start = position(io)
                    channel_header = read_struct(io, Twix.sChannelHeader)
                    channel_length = Int(channel_header.ulTypeAndChannelLength >> 8)
                    channel_ids[channel] = Int(channel_header.ulChannelId)
                    read!(io, view(data, :, channel))
                    seek(io, channel_start + channel_length)
                end
                push!(scans, (;
                    timestamp=Int(header.ulTimeStamp),
                    flags,
                    channel_ids,
                    header,
                    data,
                ))
            end
            seek(io, scan_start + dma_length)
            isodd(flags) && break
        end
    end
    scans
end

function sequence_layout(sequence)
    definitions = sequence.DEF
    readout_samples = round(Int, definitions["Nx"])
    profiles_per_image = round(Int, definitions["ADCEventsPerImage"])
    startup_profiles = round(Int, definitions["StartupADCCount"])
    phase_lines = round(Int, definitions["Ny"])
    contrasts = round(Int, definitions["NumImages"])
    images_per_inversion = round.(Int, definitions["ImagesPerInversion"])
    readout_samples == READOUT_SAMPLES || error("Sequence/readout sample mismatch")
    profiles_per_image == ACQUIRED_PROFILES || error("Sequence/profile-count mismatch")
    startup_profiles == STARTUP_PROFILES || error("Sequence/startup-profile mismatch")
    phase_lines == FULL_PHASE_LINES || error("Sequence/phase-matrix mismatch")
    contrasts == CONTRASTS || error("Sequence/contrast-count mismatch")
    sum(images_per_inversion) == CONTRASTS || error("Invalid inversion grouping")

    _, adc_kspace = get_kspace(sequence)
    size(adc_kspace, 1) == readout_samples * profiles_per_image * contrasts ||
        error("Sequence ADC count does not match its definitions")
    fov_phase = definitions["FOV"][2]
    first_image_ky = [
        adc_kspace[(profile - 1) * readout_samples + readout_samples ÷ 2, 2]
        for profile in 1:profiles_per_image
    ]
    pattern = round.(Int,
        first_image_ky[(startup_profiles + 1):end] .* fov_phase .+
        (phase_lines + 1) / 2,
    )
    length(unique(pattern)) == length(pattern) || error("Repeated encoded phase line")
    all((1 .<= pattern) .& (pattern .<= phase_lines)) ||
        error("Encoded phase line is out of range")
    pattern
end

function entry_header(filename, entry)
    open(filename) do io
        seek(io, entry.off_)
        length = Int(read(io, UInt32))
        String(read(io, length - sizeof(UInt32)))
    end
end

function protocol_value(header, key)
    found = match(Regex("\\Q$key\\E\\s*=\\s*([-+0-9.eE]+)"), header)
    isnothing(found) && error("Missing Siemens protocol value: $key")
    parse(Float32, only(found.captures))
end

scan_position(scan) = Float32[
    scan.header.sSliceData.sSlicePosVec.flSag,
    scan.header.sSliceData.sSlicePosVec.flCor,
    scan.header.sSliceData.sSlicePosVec.flTra,
]
scan_directions(scan) = Float32.(KomaMRI.KomaMRIBase.rotation_matrix(
    QuaternionRot(scan.header.sSliceData.aflQuaternion...),
))
centered_axis(width, count) = Float32.(range(
    -width / 2 + width / (2count); step=width / count, length=count,
))

function adjustment_sensitivity_maps(filename, entry, imaging_scan, image_size, image_fov)
    header = entry_header(filename, entry)
    matrix = Tuple(round(Int, protocol_value(header, key)) for key in (
        "sKSpace.lBaseResolution", "sKSpace.lPhaseEncodingLines",
        "sKSpace.lPartitions",
    ))
    fov = Float32[protocol_value(header, key) for key in (
        "sSliceArray.asSlice[0].dReadoutFOV",
        "sSliceArray.asSlice[0].dPhaseFOV",
        "sSliceArray.asSlice[0].dThickness",
    )]
    coil_count = length(imaging_scan.channel_ids)
    scans = data_scans(filename, entry, 2matrix[1], coil_count)
    kspace = zeros(ComplexF32, 2matrix[1], matrix[2], matrix[3], coil_count)
    for scan in scans
        line = Int(scan.header.sLC.ushLine) + 1
        partition = Int(scan.header.sLC.ushPartition) + 1
        kspace[:, line, partition, :] .= scan.data
    end
    volumes = fftshift(ifft(ifftshift(kspace, (1, 2, 3)), (1, 2, 3)), (1, 2, 3))
    first_readout = fld(size(volumes, 1) - matrix[1], 2) + 1
    readout = first_readout:(first_readout + matrix[1] - 1)
    channels = [only(findall(==(id), scans[1].channel_ids)) for id in imaging_scan.channel_ids]
    volumes = Array(@view volumes[readout, :, :, channels])
    rss = sqrt.(sum(abs2, volumes; dims=4))
    volumes ./= max.(rss, 0.01f0 * maximum(rss))

    x = repeat(centered_axis(image_fov[1], image_size[1]), image_size[2])
    y = repeat(centered_axis(image_fov[2], image_size[2]); inner=image_size[1])
    positions = scan_position(imaging_scan) .+
        scan_directions(imaging_scan)[:, 1] .* x' .+
        scan_directions(imaging_scan)[:, 2] .* y'
    coordinates = scan_directions(scans[1])' *
        (positions .- scan_position(scans[1]))
    receiver = ArbitraryCoilSens(
        centered_axis.(fov, size(volumes)[1:3])..., volumes,
    )
    values = get_sens(receiver, eachrow(coordinates)...)
    maps = reshape(values, image_size..., coil_count)
    rss = sqrt.(sum(abs2, maps; dims=3))
    maps ./ max.(rss, 0.01f0 * maximum(rss))
end

function contiguous_phase_region(pattern)
    breaks = findall(!=(1), diff(pattern))
    starts = [1; breaks .+ 1]
    stops = [breaks; length(pattern)]
    regions = [pattern[first:last] for (first, last) in zip(starts, stops)]
    regions[argmax(length.(regions))]
end

function calibration_targets(acs, phase_offsets, missing_remainder)
    [
        line for line in acs
        if mod(line, 2) == missing_remainder &&
            all(
                offset -> first(acs) <= line + offset <= last(acs),
                phase_offsets,
            )
    ]
end

function fill_features!(features, kspace, readout, phase, contrast,
    readout_offsets, phase_offsets)
    feature = 1
    for phase_offset in phase_offsets, readout_offset in readout_offsets,
        coil in axes(kspace, 4)
        features[feature] = kspace[
            readout + readout_offset,
            phase + phase_offset,
            contrast,
            coil,
        ]
        feature += 1
    end
    nothing
end

function grappa_weights(kspace, contrasts, target_lines,
    readout_offsets, phase_offsets)
    readout_range = (1 - first(readout_offsets)):(
        size(kspace, 1) - last(readout_offsets)
    )
    feature_count = length(readout_offsets) * length(phase_offsets) *
        size(kspace, 4)
    row_count = length(contrasts) * length(target_lines) * length(readout_range)
    source = Matrix{ComplexF32}(undef, row_count, feature_count)
    target = Matrix{ComplexF32}(undef, row_count, size(kspace, 4))
    features = Vector{ComplexF32}(undef, feature_count)
    row = 1
    for contrast in contrasts, phase in target_lines, readout in readout_range
        fill_features!(
            features, kspace, readout, phase, contrast,
            readout_offsets, phase_offsets,
        )
        source[row, :] .= features
        target[row, :] .= kspace[readout, phase, contrast, :]
        row += 1
    end
    gram = Matrix{ComplexF64}(source' * source)
    cross = Matrix{ComplexF64}(source' * target)
    regularization = GRAPPA_REGULARIZATION * real(tr(gram)) / size(gram, 1)
    ComplexF32.((gram + regularization * I) \ cross)
end

function grappa_error(kspace, weights, contrast, target_lines,
    readout_offsets, phase_offsets)
    readout_range = (1 - first(readout_offsets)):(
        size(kspace, 1) - last(readout_offsets)
    )
    features = Vector{ComplexF32}(undef, size(weights, 1))
    prediction = Vector{ComplexF32}(undef, size(weights, 2))
    squared_error = 0.0
    squared_reference = 0.0
    for phase in target_lines, readout in readout_range
        fill_features!(
            features, kspace, readout, phase, contrast,
            readout_offsets, phase_offsets,
        )
        prediction .= vec(transpose(features) * weights)
        reference = @view kspace[readout, phase, contrast, :]
        squared_error += sum(abs2, prediction .- reference)
        squared_reference += sum(abs2, reference)
    end
    sqrt(squared_error / squared_reference)
end

function interpolate_grappa!(kspace, pattern)
    acquired = falses(size(kspace, 2))
    acquired[pattern] .= true
    missing_lines = findall(.!acquired)
    missing_remainder = mod(first(missing_lines), 2)
    acs_targets = findall(acquired .&
        (mod.(eachindex(acquired), 2) .== missing_remainder))
    acs = (first(acs_targets) - 1):last(acs_targets)
    length(acs) == 24 || error("Expected a 24-line contiguous ACS region")

    interior_targets = calibration_targets(
        acs, GRAPPA_PHASE_OFFSETS, missing_remainder,
    )
    validation_weights = grappa_weights(
        kspace, 1:7, interior_targets,
        GRAPPA_READOUT_OFFSETS, GRAPPA_PHASE_OFFSETS,
    )
    validation_nrmse = grappa_error(
        kspace, validation_weights, 8, interior_targets,
        GRAPPA_READOUT_OFFSETS, GRAPPA_PHASE_OFFSETS,
    )
    interior_weights = grappa_weights(
        kspace, axes(kspace, 3), interior_targets,
        GRAPPA_READOUT_OFFSETS, GRAPPA_PHASE_OFFSETS,
    )
    boundary_weights = grappa_weights(
        kspace, axes(kspace, 3), interior_targets,
        0:0, GRAPPA_PHASE_OFFSETS,
    )

    upper_phase_offsets = (-3, -1)
    upper_targets = calibration_targets(
        acs, upper_phase_offsets, missing_remainder,
    )
    upper_weights = grappa_weights(
        kspace, axes(kspace, 3), upper_targets,
        GRAPPA_READOUT_OFFSETS, upper_phase_offsets,
    )
    upper_boundary_weights = grappa_weights(
        kspace, axes(kspace, 3), upper_targets,
        0:0, upper_phase_offsets,
    )

    interior_readout = (1 - first(GRAPPA_READOUT_OFFSETS)):(
        size(kspace, 1) - last(GRAPPA_READOUT_OFFSETS)
    )
    boundary_readout = setdiff(axes(kspace, 1), interior_readout)
    features = Vector{ComplexF32}(undef, size(interior_weights, 1))
    prediction = Vector{ComplexF32}(undef, size(interior_weights, 2))
    boundary_features = Vector{ComplexF32}(undef, size(boundary_weights, 1))
    for contrast in axes(kspace, 3), phase in missing_lines
        phase_offsets = phase == size(kspace, 2) ?
            upper_phase_offsets : GRAPPA_PHASE_OFFSETS
        weights = phase == size(kspace, 2) ? upper_weights : interior_weights
        edge_weights = phase == size(kspace, 2) ?
            upper_boundary_weights : boundary_weights
        for readout in interior_readout
            fill_features!(
                features, kspace, readout, phase, contrast,
                GRAPPA_READOUT_OFFSETS, phase_offsets,
            )
            prediction .= vec(transpose(features) * weights)
            kspace[readout, phase, contrast, :] .= prediction
        end
        for readout in boundary_readout
            fill_features!(
                boundary_features, kspace, readout, phase, contrast,
                0:0, phase_offsets,
            )
            prediction .= vec(transpose(boundary_features) * edge_weights)
            kspace[readout, phase, contrast, :] .= prediction
        end
    end
    (; acs, validation_nrmse, reconstructed_lines=length(missing_lines))
end

function reconstruct_contrasts(scans, pattern, sensitivity_maps)
    length(scans) == ACQUIRED_PROFILES * CONTRASTS ||
        error("Expected 760 imaging profiles, found $(length(scans))")
    all(scan.channel_ids == scans[1].channel_ids for scan in scans) ||
        error("Receive-channel order changes within the measurement")

    acquired = Array{ComplexF32}(
        undef, READOUT_SAMPLES, ACQUIRED_PROFILES, CONTRASTS, COILS,
    )
    for contrast in 1:CONTRASTS, profile in 1:ACQUIRED_PROFILES
        acquired[:, profile, contrast, :] .=
            scans[(contrast - 1) * ACQUIRED_PROFILES + profile].data
    end

    encoded = zeros(
        ComplexF32, READOUT_SAMPLES, FULL_PHASE_LINES, CONTRASTS, COILS,
    )
    encoded[:, pattern, :, :] .= acquired[:, (STARTUP_PROFILES + 1):end, :, :]
    grappa = interpolate_grappa!(encoded, pattern)
    coil_images = fftshift(ifft(ifftshift(encoded, (1, 2)), (1, 2)), (1, 2))

    support_magnitude = sqrt.(dropdims(
        sum(abs2, @view(coil_images[:, :, 5, :]); dims=3); dims=3,
    ))
    denominator = sum(abs2, sensitivity_maps; dims=3) .+ eps(Float32)
    combined = Array{ComplexF32}(
        undef, READOUT_SAMPLES, FULL_PHASE_LINES, CONTRASTS,
    )
    for contrast in 1:CONTRASTS
        contrast_coils = coil_images[:, :, contrast, :]
        combined[:, :, contrast] .= dropdims(sum(
            conj.(sensitivity_maps) .* contrast_coils; dims=3,
        ); dims=3) ./ dropdims(denominator; dims=3)
    end
    reference = @view combined[:, :, 5]
    reference_phase = reference ./ max.(abs.(reference), floatmin(Float32))
    signed = real.(conj.(reshape(reference_phase, size(reference)..., 1)) .* combined)
    signed, support_magnitude, grappa
end

function save_pgm(path, image)
    scale = quantile(vec(image), 0.995)
    pixels = round.(UInt8, 255 .* clamp.(image ./ scale, 0, 1))
    open(path, "w") do io
        write(io, "P5\n$(size(image, 2)) $(size(image, 1))\n255\n")
        write(io, vec(permutedims(pixels)))
    end
    return nothing
end

function prepare(raw_file, sequence_file, output_directory)
    mkpath(output_directory)
    sequence = read_seq(sequence_file)
    pattern = sequence_layout(sequence)
    entries = raid_entries(raw_file)
    length(entries) >= 2 || error("The Twix file has no openMOLLI measurement")
    scans = data_scans(raw_file, entries[2], READOUT_SAMPLES, COILS)
    sensitivity_maps = adjustment_sensitivity_maps(
        raw_file, entries[1], first(scans), OUTPUT_SIZE,
        1000f0 .* Float32.(sequence.DEF["FOV"][1:2]),
    )
    signed_contrasts, support_magnitude, grappa =
        reconstruct_contrasts(scans, pattern, sensitivity_maps)

    rows = round.(Int, range(1, size(signed_contrasts, 1); length=OUTPUT_SIZE[1]))
    columns = round.(Int, range(1, size(signed_contrasts, 2); length=OUTPUT_SIZE[2]))
    signed_contrasts = signed_contrasts[rows, columns, :]
    support_magnitude = support_magnitude[rows, columns]
    support = support_magnitude .>= 0.05f0 * maximum(support_magnitude)

    write(joinpath(output_directory, "signed_contrasts.f32"), signed_contrasts)
    write(joinpath(output_directory, "coil_sensitivities.c32"), sensitivity_maps)
    write(joinpath(output_directory, "support_magnitude.f32"), support_magnitude)
    write(joinpath(output_directory, "support.u8"), UInt8.(support))
    cp(sequence_file, joinpath(output_directory, basename(sequence_file)); force=true)
    save_pgm(
        joinpath(output_directory, "contrast_mosaic.pgm"),
        hcat((abs.(signed_contrasts[:, :, contrast]) for contrast in 1:CONTRASTS)...),
    )

    center_profile = STARTUP_PROFILES +
        round(Int, sequence.DEF["CenterKSpaceOrderIndex"]) + 1
    center_timestamps = [
        scans[(contrast - 1) * ACQUIRED_PROFILES + center_profile].timestamp
        for contrast in 1:CONTRASTS
    ]
    center_gaps = 2.5f-3 .* diff(center_timestamps)
    write(joinpath(output_directory, "center_gaps.f64"), Float64.(center_gaps))
    write(
        joinpath(output_directory, "metadata.txt"),
        """
        raw_data = $raw_file
        sequence_file = $sequence_file
        sequence_sha256 = $(bytes2hex(sha256(read(sequence_file))))
        sequence_name = $(sequence.DEF["Name"])
        sequence_duration_seconds = $(dur(sequence))
        sequence_blocks = $(length(sequence))
        imaging_profiles = $(length(scans))
        contrast_count = $CONTRASTS
        profiles_per_contrast = $ACQUIRED_PROFILES
        startup_profiles_per_contrast = $STARTUP_PROFILES
        encoded_profiles_per_contrast = $(length(pattern))
        encoded_phase_pattern = $pattern
        GRAPPA_interpolation = true
        GRAPPA_acceleration = 2
        GRAPPA_ACS_lines = $(collect(grappa.acs))
        GRAPPA_readout_offsets = $(collect(GRAPPA_READOUT_OFFSETS))
        GRAPPA_phase_offsets = $(collect(GRAPPA_PHASE_OFFSETS))
        GRAPPA_regularization = $GRAPPA_REGULARIZATION
        GRAPPA_reconstructed_phase_lines = $(grappa.reconstructed_lines)
        GRAPPA_heldout_contrast_NRMSE = $(grappa.validation_nrmse)
        coil_sensitivity_source = $(String(UInt8[x for x in entries[1].protName_ if x != 0]))
        coil_phase_reference_contrast = 5
        fully_sampled_data = none
        readout_samples = $READOUT_SAMPLES
        receive_channels = $COILS
        output_size = $(OUTPUT_SIZE[1])x$(OUTPUT_SIZE[2])
        supported_pixels = $(count(support))
        center_timestamps = $center_timestamps
        center_gaps_seconds = $center_gaps
        """,
    )
    println("Prepared openMOLLI AD inputs in $output_directory")
    (; signed_contrasts, sensitivity_maps, support_magnitude, support,
       center_gaps, grappa)
end

export prepare

end


if abspath(PROGRAM_FILE) == (@__FILE__)
    raw_file = isempty(ARGS) ?
        "/Users/alyakurt/Downloads/cardiac-multi/meas_MID00420_FID37841_openMOLLIst.dat" :
        ARGS[1]
    sequence_file = length(ARGS) < 2 ?
        "/Users/alyakurt/Downloads/OpenMOLLI_LCD_S.seq" : ARGS[2]
    output_directory = length(ARGS) < 3 ?
        "/Users/alyakurt/Desktop/komaMRI/.tmp/openmolli_ad_inputs" : ARGS[3]
    PrepareOpenMOLLIADInputs.prepare(raw_file, sequence_file, output_directory)
end
