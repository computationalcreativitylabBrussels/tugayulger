module FormantScorer

using DataFrames, Statistics

export score_points, calculate_scores

function score_f0(points::Vector{Any})
    scores = zeros(length(points))
    f0_index = argmin([point[1] for point in points])
    scores[f0_index] = 1.0
    return scores
end

function score_f1(points::Vector{Any}, column::Symbol=:power)
    scores = zeros(length(points))
    max_power_index = argmax([point[2] for point in points])
    strongest_frequency = points[max_power_index][1]

    for (i, point) in enumerate(points)
        distance = abs(point[1] - strongest_frequency)
        scores[i] = 1 - (distance / strongest_frequency)
    end

    return scores
end

function score_harmonic(points::Vector{Any}, f0::Float64)
    scores = zeros(length(points))
    for (i, point) in enumerate(points)

        closest_harmonic_distance = minimum(abs(point[1] - n * f0) for n in 1:10)

        scores[i] = 1 -  (closest_harmonic_distance / f0)
    end
    return scores
end

function calculate_scores(onset_vectors::Vector{Any}, tolerance::Float64, column::Symbol=:power)
    scored_output = []

    for (onset, points) in onset_vectors
        f0_scores = score_f0(points)
        f1_scores = score_f1(points, column)
        f0 = points[argmax(f0_scores)][1]  # Get the frequency of f0
        harmonic_scores = score_harmonic(points, f0)

        # Combine scores into a vector of tuples
        scores = [
            (
                points[i][1],  # Frequency
                points[i][2],  # Power
                f0_scores[i],
                f1_scores[i],
                harmonic_scores[i]
            ) for i in 1:length(points)
        ]

        push!(scored_output, (onset, scores))
    end

    return scored_output
end

function print_scores(scored_data, normalization_factors)
    println("Scores for Points:")
    min_value, max_value, min_frequency, max_frequency = normalization_factors

    for (onset, scores) in scored_data
        println("Onset (s): ", onset / 16000)
        for score in scores
            # Ensure score is a tuple and access elements by index
            normalized_frequency = score[1]
            normalized_power = score[2]
            f0_score = score[3]
            f1_score = score[4]
            harmonic_score = score[5]

            # Reconstruct original values
            original_frequency = normalized_frequency * (max_frequency - min_frequency) + min_frequency
            original_power = normalized_power * (max_value - min_value) + min_value

            # Print the reconstructed values and scores
            println("  Frequency (Hz): ", original_frequency,
                    ", Power: ", original_power,
                    ", f0 Score: ", f0_score,
                    ", F1 Score: ", f1_score,
                    ", Harmonic Score: ", harmonic_score)
        end
    end
end

-

end