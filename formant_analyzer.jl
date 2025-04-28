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

function extract_formants2(processed_peaks::Vector{Any}, tolerance::Float64=0.1, min_formant_distance::Float64=0.025)
    formants_output = []

    for (window_start, peaks) in processed_peaks

        # Skip empty windows
        if isempty(peaks)
            println("Window starting at $window_start is empty. Skipping.")
            push!(formants_output, (window_start, []))
            continue
        end

        # Sort peaks by frequency (ascending)
        sorted_peaks = sort(peaks, by=x->x[1])

        # Identify F0 - lowest frequency peak (assuming it's present)
        F0 = sorted_peaks[1]

        # Identify F1 - strongest amplitude peak
        F1 = sorted_peaks[argmax([p[2] for p in sorted_peaks])]

        # Initialize formants with F1
        detected_formants = [F1]

        # Process remaining peaks for additional formants (F2, F3, etc.)
        remaining_peaks = [p for p in sorted_peaks if p != F0 && p != F1]

        # Sort remaining by amplitude (descending) for processing
        sort!(remaining_peaks, by=x->x[2], rev=true)

        for peak in remaining_peaks
            # Skip if too close to a harmonic (n*F0 frequency)
            is_harmonic = false
            for n in 2:5  # Check up to 5th harmonic
                harmonic_freq = n * F0[1]
                if abs(peak[1] - harmonic_freq) < tolerance
                    println("Peak $(peak .* 8000) dismissed as harmonic of F0.")
                    is_harmonic = true
                    break
                end
            end

            # Skip if too close to existing formants
            is_close_to_formant = any(abs(peak[1] - f[1]) < min_formant_distance for f in detected_formants)
            if is_close_to_formant
                println("Peak $(peak .* 8000) dismissed as too close to an existing formant.")
            end

            # Potential formant if not harmonic and not near existing formant
            if !is_harmonic && !is_close_to_formant
                println("Peak $(peak .* 8000) added as a formant.")
                push!(detected_formants, peak)

                # Stop if we've found reasonable number of formants (F1-F4)
                if length(detected_formants) >= 4
                    println("Detected sufficient formants (F1-F4). Stopping further processing.")
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


function select_window(df::DataFrame, begin_onset::Int64, end_onset::Int64)
    window = filter(row -> row[:onset] >= begin_onset && row[:onset] <= end_onset, df)
    return window
end

end