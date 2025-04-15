module FormantAnalyzer

using DataFrames, Statistics, Distributions

export extract_formants, extract_f0_and_harmonics, select_window

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