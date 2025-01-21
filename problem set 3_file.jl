#Prblem 1
X = 50.0 
c = 0.5   
q = 0.15
p_min = 10.0
p_max = 100.0
N = 50
prices = range(p_min, p_max, length=100)
probability_density = 1.0 / (p_max - p_min)


function bellman_equation(N)
    v = zeros(N + 1) 
    σ_approach = zeros(Bool, N)  
    σ_buy = zeros(Bool, N, length(prices)) 
    for n in N:-1:1
        expected_value_approach = sum(probability_density * max(X - price, v[n + 1] - c) for price in prices)
        v_approach = expected_value_approach - c
        v_terminate = v[n + 1]
        if v_approach > v_terminate
            σ_approach[n] = true
            v[n] = v_approach
        else
            σ_approach[n] = false
            v[n] = v_terminate
        end
        for (i, price) in enumerate(prices)
            σ_buy[n, i] = (X - price) >= v[n]
        end
    end

    return v, σ_approach, σ_buy
end


function compute_purchase_probability(σ_buy, N)
    prob_buy = 0.0
    for n in 1:N
        prob_buy += sum(σ_buy[n, :]) * probability_density
    end
    return prob_buy
end

function compute_expected_price(σ_buy, N)
    expected_price = 0.0
    for n in 1:N
        for (i, price) in enumerate(prices)
            if σ_buy[n, i]
                expected_price += price * probability_density
            end
        end
    end
    return expected_price
end

function compute_expected_vendors(σ_approach, N)
    return sum(σ_approach)
end


v, σ_approach, σ_buy = bellman_equation(N)
probability_of_buy = compute_purchase_probability(σ_buy, N)
expected_price = compute_expected_price(σ_buy, N)
expected_vendors = compute_expected_vendors(σ_approach, N)

println("a) Probability that Basil will buy the orchid: $probability_of_buy")
println("b) Expected price that Basil will pay: $expected_price")
println("c) Expected number of vendors Basil will approach: $expected_vendors")

#Problem 2
using Plots


β = 0.95  
c = 10.0
wages = 10:0.5:20
π_w = fill(1 / length(wages), length(wages))

function solve_value_functions(p)
    V_U = zeros(length(wages))
    V_E = zeros(length(wages))
    tol = 1e-6
    diff = 1.0
    while diff > tol
        V_U_new = [maximum([V_E[i], c + β * sum(V_U .* π_w)]) for i in 1:length(wages)]
        V_E_new = [w + β * ((1 - p) * V_E[i] + p * sum(V_U .* π_w)) for (i, w) in enumerate(wages)]
        diff = maximum(abs.(V_U_new .- V_U)) + maximum(abs.(V_E_new .- V_E))
        V_U, V_E = V_U_new, V_E_new
    end
    return V_U, V_E
end


function reservation_wage(p)
    V_U, V_E = solve_value_functions(p)
    for (i, w) in enumerate(wages)
        if c + β * sum(V_U .* π_w) <= V_E[i]
            return w
        end
    end
    return maximum(wages)  
end


function acceptance_probability(w_star)
    return sum(π_w[i] for (i, w) in enumerate(wages) if w >= w_star)
end


function expected_unemployment_duration(q)
    return 1 / q
end


p_values = 0:0.1:0.9
reservation_wages = [reservation_wage(p) for p in p_values]
acceptance_probs = [acceptance_probability(w_star) for w_star in reservation_wages]
expected_durations = [expected_unemployment_duration(q) for q in acceptance_probs]

plot(p_values, reservation_wages, xlabel="Separation Probability (p)", ylabel="Reservation Wage (w*)", label="Reservation Wage", legend=:top)
savefig("reservation_wage_plot.png")

plot(p_values, acceptance_probs, xlabel="Separation Probability (p)", ylabel="Acceptance Probability (q)", label="Acceptance Probability", legend=:top)
savefig("acceptance_probability_plot.png")

plot(p_values, expected_durations, xlabel="Separation Probability (p)", ylabel="Expected Duration", label="Expected Duration of Unemployment", legend=:top)
savefig("expected_duration_plot.png")



#Problem 3
β = 0.95      
α = 0.3
delta = 0.05
f(k) = k^α
k_star = ((β * α) / (1 - β * (1 - delta)))^(1 / (1 - α))
function compute_convergence_time(k0, γ, epsilon)
    k_t = k0
    t = 0
    while abs(k_star - k_t) > epsilon
        c_t = (1 - β) * k_t^α 
        k_t = β * k_t^α + (1 - delta) * k_t - c_t
        t += 1
    end
    return t
end


function generate_table(k0, γ_values, epsilon)
    println("γ\tConvergence Time")
    println("--------------------------")
    for γ in γ_values
        t_convergence = compute_convergence_time(k0, γ, epsilon)
        println("$γ\t$t_convergence")
    end
end


function plot_dynamics(k0, γ_values, T)
    plots = [] 
    labels = ["γ = 0.5", "γ = 1.0", "γ = 2.0"]
    capital_levels = []
    outputs = []
    investments_ratios = []
    consumptions_ratios = []

    for (i, γ) in enumerate(γ_values)
        k_t = k0
        capitals, outputs_t, investments, consumptions = [], [], [], []
        for t in 1:T
            y_t = f(k_t) 
            c_t = (1 - β) * y_t
            i_t = β * y_t - c_t
            push!(capitals, k_t)
            push!(outputs_t, y_t)
            push!(investments, i_t / y_t)
            push!(consumptions, c_t / y_t)
            k_t = β * y_t + (1 - delta) * k_t - c_t
        end
        push!(capital_levels, capitals)
        push!(outputs, outputs_t)
        push!(investments_ratios, investments)
        push!(consumptions_ratios, consumptions)
    end


    p1 = plot(1:T, capital_levels, label=labels, title="Capital Levels", xlabel="Time", ylabel="Capital")
    p2 = plot(1:T, outputs, label=labels, title="Outputs", xlabel="Time", ylabel="Output")
    p3 = plot(1:T, investments_ratios, label=labels, title="Investment to Output Ratio", xlabel="Time", ylabel="Ratio")
    p4 = plot(1:T, consumptions_ratios, label=labels, title="Consumption to Output Ratio", xlabel="Time", ylabel="Ratio")


    plot(p1, p2, p3, p4, layout=(2, 2), size=(800, 600))
end


k0 = 0.5 * k_star  
γ_values = [0.5, 1.0, 2.0] 
epsilon = 0.5 * (k_star - k0) 
T = 50 


println("Convergence Table:")
generate_table(k0, γ_values, epsilon)


println("Generating plots...")
plot_dynamics(k0, γ_values, T)




#Problem 4
using LinearAlgebra
P = [0.5 0.3 0.2;
     0.2 0.7 0.1;
     0.3 0.3 0.4]
function σ(X_t, Z_t)
    if Z_t == 1
        return 0
    elseif Z_t == 2
        return X_t
    elseif Z_t == 3 && X_t ≤ 4
        return X_t + 1
    elseif Z_t == 3 && X_t == 5
        return 3
    else
        error("Invalid state!")
    end
end



X = 0:5
Z = 1:3 

function compute_joint_transition_matrix(X, Z, P)
    num_states_X = length(X)
    num_states_Z = length(Z)
    joint_size = num_states_X * num_states_Z
    T = zeros(Float64, joint_size, joint_size)

    state_to_index = Dict((x, z) => (z - 1) * num_states_X + x + 1 for x in X, z in Z)

    for x in X
        for z in Z
            for zp in Z
                next_x = σ(x, z)
                if next_x in X
                    row = state_to_index[(x, z)]
                    col = state_to_index[(next_x, zp)]
                    T[row, col] += P[z, zp]
                end
            end
        end
    end
    return T
end

function stationary_distribution(T)
    eigvals, eigvecs = eigen(T')
    stationary = eigvecs[:, argmax(real(eigvals))]
    stationary /= sum(stationary)  
    return stationary
end

function marginal_distribution(stationary, X, Z)
    num_states_X = length(X)
    num_states_Z = length(Z)
    marginal = zeros(Float64, num_states_X)

    for x in X
        for z in Z
            idx = (z - 1) * num_states_X + x + 1
            marginal[x + 1] += stationary[idx]
        end
    end
    return marginal
end

function expected_value_X(marginal, X)
    return sum(x * marginal[x + 1] for x in X)
end


T_joint = compute_joint_transition_matrix(X, Z, P)
stationary = stationary_distribution(T_joint)
marginal_X = marginal_distribution(stationary, X, Z)
expected_X = expected_value_X(marginal_X, X)

println("Joint Transition Matrix (T):")
println(T_joint)
println("\nStationary Distribution (Joint):")
println(stationary)
println("\nMarginal Distribution of X_t:")
println(marginal_X)
println("\nExpected Value of X_t:")
println(expected_X)

