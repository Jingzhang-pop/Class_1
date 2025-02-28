

using QuantEcon, Distributions, Plots, Optim, Interpolations, LinearAlgebra, Statistics, Roots


# -------------------------- Model Parameters --------------------------
β    = 0.96       # discount factor
γ    = 2.0        # CRRA coefficient
ϕ    = 0.0        # borrowing constraint (a' ≥ -ϕ)
ρ    = 0.9        # persistence of productivity
σ    = 0.4        # std of productivity shocks
α    = 0.467       # capital share
δ    = 0.1        # depreciation rate
A    = 1.0        # total factor productivity
L    = 1.0        # labor supply (normalized)

#--------------------- Productivity Process ---------------------
N_z = 5  # number of productivity states
mc = tauchen(N_z, ρ, sqrt(σ), 0.0)
λ_z_vec = stationary_distributions(mc)[1]
# Normalize so that E[z] = 1:
z_vec = exp.(mc.state_values)
z_vec = z_vec ./ sum(z_vec .* λ_z_vec)
P_z = mc.p

# --------------------- Asset Grid ---------------------
N_a   = 150
a_min = -ϕ
a_max = 150.0
rescaler = range(0, 1, length=N_a) .^ 5
a_vec = a_min .+ rescaler*(a_max - a_min)

# --------------------- Utility Function ---------------------
u(c) = (c^(1-γ) - 1)/(1-γ)

# --------------------- Household Problem ---------------------
function bellman_operator(v, r, τ, λ_tax)
    N_a = length(a_vec)
    N_z = length(z_vec)
    v_new = zeros(N_a, N_z)
    policy_index = zeros(Int, N_a, N_z)
    for iz in 1:N_z
        z = z_vec[iz]
        net_income = (1-τ) * z^(1-λ_tax)
        for ia in 1:N_a
            a = a_vec[ia]
            max_val = -Inf
            max_ind = 1
            for ia_next in 1:N_a
                a_next = a_vec[ia_next]
                c = (1+r)*a + net_income - a_next
                val = c > 0 ? u(c) : -Inf
                if c > 0
                    cont = 0.0
                    for iz_next in 1:N_z
                        cont += P_z[iz, iz_next]*v[ia_next, iz_next]
                    end
                    val += β * cont
                end
                if val > max_val
                    max_val = val
                    max_ind = ia_next
                end
            end
            v_new[ia, iz] = max_val
            policy_index[ia, iz] = max_ind
        end
    end
    return v_new, policy_index
end

function solve_household(r, τ, λ_tax; tol=1e-8, maxiter=1000)
    N_a = length(a_vec)
    N_z = length(z_vec)
    v = zeros(N_a, N_z)
    iter = 0
    diff = Inf
    policy_index = ones(Int, N_a, N_z)
    while diff > tol && iter < maxiter
        v_new, policy_index_new = bellman_operator(v, r, τ, λ_tax)
        diff = maximum(abs.(v_new - v))
        v = v_new
        policy_index = policy_index_new
        iter += 1
    end
    return v, policy_index, iter, diff
end

function policy_function(policy_index)
    N_a, N_z = size(policy_index)
    policy = zeros(N_a, N_z)
    for i in 1:N_a, j in 1:N_z
        policy[i, j] = a_vec[policy_index[i, j]]
    end
    return policy
end

# --------------------- Transition Matrix & Stationary Distribution ---------------------
function transition_matrix(policy_index)
    N_a = length(a_vec)
    N_z = length(z_vec)
    N = N_a * N_z
    Q = zeros(N, N)
    for iz in 1:N_z
        for ia in 1:N_a
            idx = (iz-1)*N_a + ia
            ia_next = policy_index[ia, iz]
            for iz_next in 1:N_z
                idx_next = (iz_next-1)*N_a + ia_next
                Q[idx, idx_next] += P_z[iz, iz_next]
            end
        end
    end
    return Q
end

function stationary_distribution(policy_index)
    Q = transition_matrix(policy_index)
    Q_power = Q^10000
    dist = Q_power[1, :]
    N_a = length(a_vec)
    N_z = length(z_vec)
    dist_matrix = reshape(dist, N_a, N_z)
    λ_a = sum(dist_matrix, dims=2)
    return dist_matrix, λ_a
end

function aggregate_assets(policy_index)
    pol = policy_function(policy_index)
    dist, λ_a = stationary_distribution(policy_index)
    A_agg = sum(pol .* dist)
    return A_agg, dist, λ_a
end

# --------------------- Firm Problem ---------------------
function solve_firm(r)
    K = (α*A/(r+δ))^(1/(1-α))
    w = (1-α)*A*K^α
    return K, w
end

# --------------------- Market Equilibrium ---------------------
function market_residual(r, τ, λ_tax)
    K_firm, w = solve_firm(r)
    v, policy_index, iter, diff = solve_household(r, τ, λ_tax)
    A_agg, _, _ = aggregate_assets(policy_index)
    return A_agg - K_firm, w, v, policy_index
end

# --------------------- Lorenz Curve ---------------------
function lorenz_curve(a_vec, λ_a)
    weights = vec(λ_a)
    total_pop = sum(weights)
    total_assets = sum(a_vec .* weights)
    cum_pop = cumsum(weights) ./ total_pop
    cum_assets = cumsum(a_vec .* weights) ./ total_assets
    return cum_pop, cum_assets
end

# --------------------- Gini Coefficient ---------------------
function gini_coefficient(x, weights)
    μ = sum(x .* weights)
    n = length(x)
    diff_sum = 0.0
    for i in 1:n
        for j in 1:n
            diff_sum += abs(x[i] - x[j]) * weights[i] * weights[j]
        end
    end
    return diff_sum / (2 * μ)
end

# --------------------- Calibration for Flat Tax (λ = 0) ---------------------
r_target = 0.04
K_target, w_target = solve_firm(r_target)
Y = A * K_target^α
τ_flat = 0.2 * Y   # Flat tax: G = τ_flat, with G/Y = 0.2
println("=== Flat Tax Case (λ = 0) ===")
println("r_target = $(r_target), K_target = $(K_target), w = $(w_target), Y = $(Y)")
println("Flat tax rate τ = $(τ_flat)")

res_flat, w_calc, v_flat, policy_index_flat = market_residual(r_target, τ_flat, 0.0)
println("Asset market residual (flat): ", res_flat)

# --------------------- Calibration for Progressive Tax (λ = 0.15) ---------------------
λ_prog = 0.15
E_z_pow = sum(z_vec.^(1-λ_prog) .* λ_z_vec)
τ_prog = 1 - (τ_flat / E_z_pow)
println("\n=== Progressive Tax Case (λ = $(λ_prog)) ===")
println("Progressive tax rate τ = $(τ_prog)")
G_prog = 1 - (1-τ_prog)*E_z_pow
println("Government revenue (progressive) G = $(G_prog)  [should equal flat revenue τ_flat = $(τ_flat)]")

res_prog, w_calc_prog, v_prog, policy_index_prog = market_residual(r_target, τ_prog, λ_prog)
println("Asset market residual (progressive): ", res_prog)

# --------------------- Aggregate Assets ---------------------
A_agg_flat, dist_flat, λ_a_flat = aggregate_assets(policy_index_flat)
A_agg_prog, dist_prog, λ_a_prog = aggregate_assets(policy_index_prog)
println("\nAggregate assets: Flat = $(A_agg_flat), Progressive = $(A_agg_prog)")
println("Firm capital supply (K_target) = $(K_target)")

# --------------------- Compute Gini Coefficients ---------------------
weights_flat = vec(λ_a_flat) ./ sum(vec(λ_a_flat))
weights_prog = vec(λ_a_prog) ./ sum(vec(λ_a_prog))
gini_assets_flat = gini_coefficient(a_vec, weights_flat)
gini_assets_prog = gini_coefficient(a_vec, weights_prog)

# flat: ỹ = (1-τ_flat)*z, progressive: ỹ = (1-τ_prog)*z^(1-λ_prog)
marginal_z_flat = vec(sum(dist_flat, dims=1))
marginal_z_flat = marginal_z_flat ./ sum(marginal_z_flat)
marginal_z_prog = vec(sum(dist_prog, dims=1))
marginal_z_prog = marginal_z_prog ./ sum(marginal_z_prog)
income_flat = (1-τ_flat) .* z_vec
income_prog = (1-τ_prog) .* (z_vec.^(1-λ_prog))
gini_income_flat = gini_coefficient(income_flat, marginal_z_flat)
gini_income_prog = gini_coefficient(income_prog, marginal_z_prog)

println("\nGini Coefficients:")
println("Assets: Flat = $(gini_assets_flat), Progressive = $(gini_assets_prog)")
println("After-tax Income: Flat = $(gini_income_flat), Progressive = $(gini_income_prog)")

# --------------------- Plotting ---------------------
plot_v_flat = plot(title="Value Function (Flat Tax)", xlabel="Assets", ylabel="Value")
for iz in 1:length(z_vec)
    plot!(plot_v_flat, a_vec, v_flat[:, iz], label="z state $(iz)")
end

plot_v_prog = plot(title="Value Function (Progressive Tax)", xlabel="Assets", ylabel="Value")
for iz in 1:length(z_vec)
    plot!(plot_v_prog, a_vec, v_prog[:, iz], label="z state $(iz)")
end

# policy function
policy_flat = policy_function(policy_index_flat)
policy_prog = policy_function(policy_index_prog)
plot_policy_flat = plot(title="Policy Function (Flat Tax)", xlabel="Assets", ylabel="Next-period Assets")
for iz in 1:length(z_vec)
    plot!(plot_policy_flat, a_vec, policy_flat[:, iz], label="z state $(iz)")
end
plot!(plot_policy_flat, a_vec, a_vec, linestyle=:dash, color=:black)

plot_policy_prog = plot(title="Policy Function (Progressive Tax)", xlabel="Assets", ylabel="Next-period Assets")
for iz in 1:length(z_vec)
    plot!(plot_policy_prog, a_vec, policy_prog[:, iz], label="z state $(iz)")
end
plot!(plot_policy_prog, a_vec, a_vec, linestyle=:dash, color=:black)

# stationary_distribution
_, λ_a_flat_mat = stationary_distribution(policy_index_flat)
_, λ_a_prog_mat = stationary_distribution(policy_index_prog)
marginal_flat = vec(λ_a_flat_mat)
marginal_prog = vec(λ_a_prog_mat)

plot_dist_flat = plot(a_vec, marginal_flat, xlabel="Assets", ylabel="Density",
    title="Asset Distribution (Flat Tax)", label="")
plot_dist_prog = plot(a_vec, marginal_prog, xlabel="Assets", ylabel="Density",
    title="Asset Distribution (Progressive Tax)", label="")

# lorenz curve
cum_pop_flat, cum_assets_flat = lorenz_curve(a_vec, λ_a_flat_mat)
cum_pop_prog, cum_assets_prog = lorenz_curve(a_vec, λ_a_prog_mat)
plot_lorenz_flat = plot(cum_pop_flat, cum_assets_flat, xlabel="Cumulative Population",
    ylabel="Cumulative Assets", title="Lorenz Curve (Flat Tax)", label="Lorenz Curve", lw=2)
plot!(plot_lorenz_flat, [0,1], [0,1], linestyle=:dash, color=:black, label="45° line")

plot_lorenz_prog = plot(cum_pop_prog, cum_assets_prog, xlabel="Cumulative Population",
    ylabel="Cumulative Assets", title="Lorenz Curve (Progressive Tax)", label="Lorenz Curve", lw=2)
plot!(plot_lorenz_prog, [0,1], [0,1], linestyle=:dash, color=:black, label="45° line")

# Gini Coefficients
gini_assets = [gini_assets_flat, gini_assets_prog]
gini_income = [gini_income_flat, gini_income_prog]
tax_regime = ["Flat Tax (λ=0)", "Progressive Tax (λ=0.15)"]

plot_gini_assets = bar(tax_regime, gini_assets, title="Gini Coefficient for Assets",
    xlabel="Tax Regime", ylabel="Gini", legend=false)
plot_gini_income = bar(tax_regime, gini_income, title="Gini Coefficient for After-tax Income",
    xlabel="Tax Regime", ylabel="Gini", legend=false)

# show all 
display(plot_v_flat)
display(plot_v_prog)
display(plot_policy_flat)
display(plot_policy_prog)
display(plot_dist_flat)
display(plot_dist_prog)
display(plot_lorenz_flat)
display(plot_lorenz_prog)
display(plot_gini_assets)
display(plot_gini_income)
