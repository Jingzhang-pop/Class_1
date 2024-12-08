#peoblem 1
function iterative_solver(f, x0, α = 0.0; ϵ = 1e-6, maxiter = 1000)
    x_current = x0
    residuals = []
    guesses = []
    flag = 1

    for n in 1:maxiter
        g_x = f(x_current) + x_current
        x_next = (1 - α) * g_x + α * x_current
        push!(guesses, x_next)
        res = abs(x_next - x_current)
        push!(residuals, res)

        if res < ϵ
            flag = 0
            return flag, x_next, x_next, res, guesses, residuals
        end

        x_current = x_next
    end

    return flag, NaN, NaN, NaN, guesses, residuals
end
f(x) = (x + 1)^(1/3) - x
result = iterative_solver(f, 1.0, 0.0; ϵ = 1e-6, maxiter = 1000)
println(result)
h(x) = x^3 - x - 1
result_h = iterative_solver(h, 1.0, 0.0; ϵ = 1e-6, maxiter = 1000)
println(result_h)

# α: A scaling parameter for f(x) to control step size (default 1.0).
#- Larger α can make the iterations faster but risk overshooting the solution.
#- Smaller α can slow convergence but ensure stability for some functions.


#problem 2

using LinearAlgebra


function exact_solution(α, β)
    x5 = 1
    x4 = x5 + 1  
    x3 = x4
    x2 = x3
    x1 = α + β * x5 + (α - β) * x4
    return [x1, x2, x3, x4, x5]
end


function solve_system(α, β)
    
    A = [1  -1  0   α-β  β;
        0   1  -1  0    0;
        0   0   1  -1   0;
        0   0   0   1  -1;
        0   0   0   0   1]
    b = [α, 0, 0, 0, 1]


    x_exact = exact_solution(α, β)

  
    x_numeric = A \ b


    cond_num = cond(A)

    residual = norm(A * x_numeric - b) / norm(b)

    return x_exact, x_numeric, cond_num, residual
end


function create_table(α, β_values)
    println("β\tExact x1\tNumeric x1\tCondition Number\tRelative Residual")
    for β in β_values
        x_exact, x_numeric, cond_num, residual = solve_system(α, β)
        println("$β\t$(x_exact[1])\t$(x_numeric[1])\t$(cond_num)\t$(residual)")
    end
end


α = 0.1
β_values = [1, 10, 100, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9, 10^10, 10^11, 10^12]
create_table(α, β_values)

#problem 3
using Roots
function NPV(r,C)
    T = length(C) - 1
    return sum(C[t + 1] / (1 + r)^t for t in 0:T)
end
function internal_rate(C)
    if all(C .>= 0)  || all(C .<=0)
        return "Warning: Cash flows do not have a sign change; IRR may not exist."
    end
    f(r) = NPV(r, C)
    try
        irr = find_zerp(f, (0.0, 1.0), Bisection())
        return irr
    catch
        return "Warning: Could not find a valid root for IRR."
    end
    
end

C = [-1000, 300, 400, 500]
println("The IRR is: ", internal_rate(C))
C1= [-5, 0, 0, 2.5, 5]
println("The IRR is: ", internal_rate(C1))

#problem 4

import Pkg; Pkg.add("Ipopt")
using Plots
using JuMP
using Ipopt

function production_function(a, σ, x1, x2)
    ρ = (σ - 1) / σ
    return (a * x1^ρ + (1 - a) * x2^ρ)^(1 / ρ)
end

function contour_plot_production_function(a, σ, x1_range, x2_range)
    ρ = (σ - 1) / σ
    f(x1, x2) = (a * x1^ρ + (1 - a) * x2^ρ)^(1 / ρ)

    x1_vals = range(x1_range[1], x1_range[2], length=100)
    x2_vals = range(x2_range[1], x2_range[2], length=100)
    z_vals = [f(x1, x2) for x1 in x1_vals, x2 in x2_vals]

    contour(x1_vals, x2_vals, z_vals, title="Production Function (σ = $σ)",
            xlabel="x1", ylabel="x2", fill=true, levels=20)
end

function cost_minimization(a, σ, w1, w2, y)
    ρ = (σ - 1) / σ
    model = Model(Ipopt.Optimizer)
    @variable(model, x1 >= 0)
    @variable(model, x2 >= 0)
    @objective(model, Min, w1 * x1 + w2 * x2)
    @constraint(model, (a * x1^ρ + (1 - a) * x2^ρ)^(1 / ρ) == y)
    optimize!(model)
    x1_val = value(x1)
    x2_val = value(x2)
    cost = w1 * x1_val + w2 * x2_val
    return cost, x1_val, x2_val
end

function plot_cost_and_inputs(a, σ_values, w1, w2, y)
    costs = []
    x1_vals = []
    x2_vals = []

    for σ in σ_values
        cost, x1, x2 = cost_minimization(a, σ, w1, w2, y)
        push!(costs, cost)
        push!(x1_vals, x1)
        push!(x2_vals, x2)
    end

    p1 = plot(σ_values, costs, label="Cost", xlabel="σ", ylabel="Cost", lw=2, title="Cost Function")
    p2 = plot(σ_values, x1_vals, label="x1 (Input Demand)", xlabel="σ", ylabel="x1", lw=2, title="Input x1")
    p3 = plot(σ_values, x2_vals, label="x2 (Input Demand)", xlabel="σ", ylabel="x2", lw=2, title="Input x2")

    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
end

function main()
    a = 0.5                
    σ_values = [0.25, 1, 4] 
    x1_range = (0.1, 5)   
    x2_range = (0.1, 5)  
    w1, w2 = 1.0, 1.0   
    y = 1.0        

    println()
    for σ in σ_values
        contour_plot_production_function(a, σ, x1_range, x2_range)
    end

    println()
    for σ in σ_values
        cost, x1, x2 = cost_minimization(a, σ, w1, w2, y)
        println("σ=$σ: Cost=$cost, x1=$x1, x2=$x2")
    end

    println()
    plot_cost_and_inputs(a, σ_values, w1, w2, y)
end

main()
