module FormantAnalyzer

using DataFrames, Statistics, Distributions

export extract_formants, extract_f0_and_harmonics, select_window, extract_formants2

# This module is responsible for extracting the formants and harmonics from given peak data

function extract_formants(onset_vectors::Vector{Any}, tolerance::Float64, min_distance::Float64 = 0.01)
    harmonics_output = []

    for (onset, points) in onset_vectors
        harmonics = extract_f0_and_harmonics(onset_vectors, 0.05)[1][2]
        harmonic_frequencies = [h[1] for h in harmonics]

        F1 = points[argmax(map(x -> x[2], points))]
        detected_formants = [F1]

        filtered_points = []
        for point in points
            if point in detected_formants
                push!(filtered_points, point)
                push!(detected_formants, point)
                continue
            end
            # idea is to include both harmonic distance and formant distance
            # but in a naive way, version 3 does this better (based on LPC)
            is_far_from_harmonics = all(abs(point[1] - h) > tolerance for h in harmonic_frequencies)
            is_far_from_formants = all(abs(point[1] - formant[1]) >= min_distance for formant in detected_formants)
            is_above_F1 = point[1] >= F1[1]

            if is_far_from_harmonics && is_far_from_formants && is_above_F1
                push!(filtered_points, point)
                push!(detected_formants, point)
            end
        end

        push!(harmonics_output, (onset, filtered_points))

    end

    return harmonics_output
end

function extract_naive_formants(onset_vectors::Vector{Any})
    naive_formants_output = []

    for (onset, points) in onset_vectors
        # sort, so we can select the strongest peaks
        sorted_points = sort(points, by=x -> x[2], rev=true)

        detected_formants = []

        for point in sorted_points
            # Stop if we already have 4 candidates
            if length(detected_formants) >= 4
                break
            end

            # add if higher frequency then last formant and higher value
            if isempty(detected_formants) || point[1] > detected_formants[end][1]
                push!(detected_formants, point)
            end
        end

        push!(naive_formants_output, (onset, detected_formants))
    end

    return naive_formants_output
end

function extract_formants2(processed_peaks::Vector{Any}, tolerance::Float64=0.1, min_formant_distance::Float64=0.025)
    formants_output = []

    for (window_start, peaks) in processed_peaks

        if isempty(peaks)
            println("Window starting at $window_start is empty. Skipping.")
            push!(formants_output, (window_start, []))
            continue
        end

        sorted_peaks = sort(peaks, by=x->x[1])

        # f0 - in dataset consistently the first peak (because not too much noise)
        F0 = sorted_peaks[1]

        # seems to be quite consistent in the dataset, F1 = strongest peak
        F1 = sorted_peaks[argmax([p[2] for p in sorted_peaks])]

        detected_formants = [F1]

        remaining_peaks = [p for p in sorted_peaks if p != F0 && p != F1]

        sort!(remaining_peaks, by=x->x[2], rev=true)


        for peak in remaining_peaks
            # Skip if too close to a harmonic (n*F0 frequency)
            is_harmonic = false
            for n in 2:5  # Check up to 5th harmonic
                harmonic_freq = n * F0[1]
                if abs(peak[1] - harmonic_freq) < tolerance
                    #println("Peak $(peak .* 8000) dismissed as harmonic of F0.")
                    is_harmonic = true
                    continue
                end
            end

            is_close_to_formant = any(abs(peak[1] - f[1]) < min_formant_distance for f in detected_formants)
            if is_close_to_formant
                #println("Peak $(peak .* 8000) dismissed as too close to an existing formant.")
            end

            # assumption that formant is not close to a harmonic and not close to another formant
            if !is_harmonic && !is_close_to_formant
                #println("Peak $(peak .* 8000) added as a formant.")
                push!(detected_formants, peak)

                # 4 formants enough in range of 5000 Hz
                if length(detected_formants) >= 4
                    #println("Detected sufficient formants (F1-F4). Stopping further processing.")
                    break
                end
            end
        end

        # Sort final formants by frequency and prepare output
        final_formants = sort(detected_formants, by=x->x[1])
        push!(formants_output, (window_start, final_formants))
    end

    return formants_output
end

function extract_formants3(
    processed_peaks::Vector{Any},
    tolerance::Float64 = 0.1, # 500 Hz - tolerance is used to check if a peak is a harmonic
    min_formant_distance::Float64 = 0.025 # +/- 100 Hz - minimum distance between formants
)
    formant_ranges = [
        (0.06, 0.18),  # F1 (300 - 900)
        (0.18, 0.5),   # F2 (900 - 2500)
        (0.5, 0.7),    # F3 (2500 - 3500)
        (0.7, 0.9)     # F4 (3500 - 4500)
    ]

    previous_formants = nothing
    formants_output = []

    for (window_start, peaks) in processed_peaks
        if isempty(peaks)
            push!(formants_output, (window_start, []))
            previous_formants = nothing
            continue
        end

        sorted_peaks = sort(peaks, by=x -> x[1])
        estimated_f0 = sorted_peaks[1][1]

        # F1 = strongest amplitude peak
        F1_index = argmax([p[2] for p in sorted_peaks])
        F1_peak = sorted_peaks[F1_index]
        F1_freq = F1_peak[1]

        candidates = []

        # Based on LPC
        for (i_peak, peak) in enumerate(sorted_peaks)
            freq, amp = peak # (it is not always amp, depends on what was passed)
            is_harmonic = any(abs(freq - n * estimated_f0) < tolerance for n in 2:5)

            # Estimate local bandwidth (distance to neighboring peaks)
            left = i_peak > 1 ? sorted_peaks[i_peak - 1][1] : freq - 0.05
            right = i_peak < length(sorted_peaks) ? sorted_peaks[i_peak + 1][1] : freq + 0.05
            bandwidth = max(right - left, 1e-4)  # avoid div-by-zero
            sharpness = amp / bandwidth          # like Q-factor

            for (i, (low, high)) in enumerate(formant_ranges)
                # F1 accepts any peak in range; F2–F4 only if > F1 freq
                if (i == 1 || freq > F1_freq) && low <= freq <= high
                    center = (low + high) / 2
                    dist_score = 1.0 - abs(freq - center) / (high - low)
                    harmonic_penalty = is_harmonic ? 0.5 : 1.0
                    continuity_bonus = previous_formants !== nothing && i <= length(previous_formants) ?
                        exp(-abs(freq - previous_formants[i][1]) / 0.1) : 1.0

                    # Final score includes sharpness (like LPC picks high-Q resonances)
                    score = amp * dist_score * harmonic_penalty * continuity_bonus * sharpness
                    push!(candidates, (i, freq, amp, score))
                end
            end
        end

        # Select best candidate per formant group
        selected_formants = Dict{Int, Tuple{Float64, Float64}}()

        for i in 1:length(formant_ranges)
            formant_candidates = filter(c -> c[1] == i, candidates)
            if !isempty(formant_candidates)
                best_idx = findmax(c -> c[4], formant_candidates)[2]
                selected_formants[i] = (formant_candidates[best_idx][2], formant_candidates[best_idx][3])
            end
        end

        ordered = sort(collect(values(selected_formants)), by=x -> x[1])
        push!(formants_output, (window_start, ordered))
        previous_formants = ordered
    end

    return formants_output
end



function extract_f0_and_harmonics(onset_vectors::Vector{Any}, tolerance::Float64, max_harmonics::Int = 10)
    harmonics_output = []

    for (onset, points) in onset_vectors
        lowest_point = points[argmin(map(x -> x[1], points))]
        f0 = lowest_point[1]
        harmonics = [lowest_point]

        for n in 2:max_harmonics
            harmonic_freq = f0 * n
            found_harmonic = false

            for point in points
                freq = point[1]
                if abs(freq - harmonic_freq) <= tolerance
                    push!(harmonics, point)
                    found_harmonic = true
                    break
                end
            end

            if !found_harmonic
                break
            end
        end

        push!(harmonics_output, (onset, harmonics))
    end

    return harmonics_output
end

function select_best_formants(onset_vectors::Vector{Any}, bin_size::Float64 = 0.01)
    # Combine all formants into a single list
    all_formants = [formant[1] for (_, peaks) in onset_vectors for formant in peaks]

    # Bin the formants
    bins = Dict{Float64, Int}()
    for formant in all_formants
        bin = round(formant / bin_size) * bin_size
        bins[bin] = get(bins, bin, 0) + 1
    end

    # Sort bins by count (descending)
    sorted_bins = sort(collect(bins), by=x -> x[2], rev=true)

    # Select formants ensuring they are well spread
    selected_formants = []
    for (bin, _) in sorted_bins
        if all(abs(bin - f) > 0.1 for f in selected_formants)
            push!(selected_formants, bin)
        end
        # Stop after selecting 4 formants
        if length(selected_formants) >= 4
            break
        end
    end

    # Return the result as a single tuple
    onset_start = onset_vectors[1][1]  # Use the first onset as the start
    sort!(selected_formants, by=x -> x[1])  # Sort selected formants by frequency
    return selected_formants
end


function select_window(df::DataFrame, begin_onset::Int64, end_onset::Int64)
    window = filter(row -> row[:onset] >= begin_onset && row[:onset] <= end_onset, df)
    return window
end

end