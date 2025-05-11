module EnvelopeAnalysis

using DataFrames, Statistics, Distributions, JuMP, Ipopt, Polynomials, Plots
export analyze_peaks

function analyze_peaks(onset_vectors::Vector{Any})
    all_local_maxima = []

    for (onset, peaks) in onset_vectors
        # Extract and normalize frequencies and amplitudes
        frequencies = [peak[1] for peak in peaks]
        amplitudes = [peak[2] for peak in peaks]

        # Normalize to [-1, 1]
        min_freq = minimum(frequencies)
        max_freq = maximum(frequencies)
        norm_freq = 2 .* (frequencies .- min_freq) ./ (max_freq - min_freq) .- 1
        norm_freq = frequencies

        degree = 9
        n_points = length(norm_freq)

        # Build constrained optimization model
        model = Model(Ipopt.Optimizer)
        set_silent(model)

        @variable(model, a[1:(degree+1)])

        # Polynomial evaluated at x: p(x) = a[1] + a[2]x + a[3]x^2 + ... + a[10]x^9
        function poly_eval(x)
            return sum(a[j] * x^(j - 1) for j in 1:(degree + 1))
        end

        # Enforce that poly lies above all data points
        for i in 1:n_points
            @constraint(model, poly_eval(norm_freq[i]) >= amplitudes[i])
        end

        # Objective: minimize total polynomial "area" (or roughness)
        @objective(model, Min, sum(poly_eval(norm_freq[i]) for i in 1:n_points))

        optimize!(model)
        coeffs = value.(a)
        poly = Polynomial(coeffs)

        # Plot
        scatter(norm_freq, amplitudes, label="Data", color=:blue)
        plot!(poly, -1, 1, label="Fitted Envelope (deg=9)", color=:red, linewidth=2)
        title!("Envelope Fit at Onset $onset")
        xlabel!("Normalized Frequency")
        ylabel!("Amplitude")
        savefig("envelope_fit_$(onset).png")

        # Extract local maxima = roots of derivative
        dpoly = derivative(poly)
        crit_pts = real.(roots(dpoly))
        crit_pts = filter(x -> -1 ≤ x ≤ 1, crit_pts)

        # Keep local maxima only (second derivative < 0)
        ddpoly = derivative(dpoly)
        maxima = filter(x -> ddpoly(x) < 0, crit_pts)
        maxima = sort(maxima)

        # Rescale back to original frequency domain
        formant_freqs = [(x + 1) * (max_freq - min_freq) / 2 + min_freq for x in maxima]

        push!(all_local_maxima, (onset=onset, formants=formant_freqs))
    end

    return all_local_maxima
end

end
