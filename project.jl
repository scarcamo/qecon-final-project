###############################################################################
#                           Heterogeneous Agent Model
#             Tax Reform – Comparing λ = 0.0 (flat tax) vs. λ = 0.15 (progressive)
#
# This code calibrates an Aiyagari‐style model with a tax function
# T(y) = y – (1-τ)*(y/ȳ)^(1-λ)*ȳ so that post‐tax labor income is 
# ỹ = (1-τ)*(y/ȳ)^(1-λ)*ȳ.
#
# In the baseline (λ = 0) we set w = 1, r = 0.04 and τ = 0.3. 
# The firm’s parameters (α, δ, A) are calibrated via:
#   r = α*A*K^(α-1) - δ   and   δK/(A*K^α) = 0.2.
# We then choose β such that the household asset market clears.
#
# In the reform economy (λ = 0.15) the calibrated parameters (β,γ,ρ,σ,α,δ,A,φ)
# remain fixed, but r, w and τ are determined endogenously (with government revenue 
# to output fixed at 0.2). We then compute equilibrium statistics and plot value/policy
# functions, asset distributions and Lorenz curves.
#
###############################################################################

using Parameters, Interpolations, Plots, QuantEcon, LinearAlgebra, Statistics, Roots

#########################
# --- Utility Functions
#########################

# CRRA utility and its derivative
function u(c, γ)
    return γ == 1 ? log(c) : (c^(1-γ) - 1)/(1-γ)
end

function u_prime(c, γ)
    return c^(-γ)
end

function u_prime_inv(mu, γ)
    return mu^(-1/γ)
end

##################################
# --- Lorenz and Gini Functions
##################################

function lorenz_curve(x::Vector{Float64}, weights::Vector{Float64})
    # Sort x and associated weights
    sorted_idx = sortperm(x)
    x_sorted = x[sorted_idx]
    w_sorted = weights[sorted_idx]
    cum_pop = cumsum(w_sorted) / sum(w_sorted)
    cum_x = cumsum(x_sorted .* w_sorted) / sum(x_sorted .* w_sorted)
    return cum_pop, cum_x
end

function gini_coefficient(x::Vector{Float64}, weights::Vector{Float64})
    sorted_idx = sortperm(x)
    x_sorted = x[sorted_idx]
    w_sorted = weights[sorted_idx]
    # Use formula: G = 1 - 2 * (weighted cumulative share)
    B = sum(w_sorted .* cumsum(x_sorted .* w_sorted)) / (sum(w_sorted)*sum(x_sorted .* w_sorted))
    return 1 - 2*B
end

#########################
# --- Model Structures
#########################

# Household (agent) problem with taxes
@with_kw struct HAProblemTax
    # Preferences
    γ::Float64 = 2.0          # CRRA curvature (γ = 2)
    β::Float64 = 0.96         # Discount factor (to be calibrated in λ=0 economy)
    ϕ::Float64 = 0.0          # Borrowing constraint (φ = 0)

    # Productivity (idiosyncratic) process: AR(1) in logs
    ρ::Float64 = 0.9          # Persistence
    σ_ϵ::Float64 = 0.4        # Std. dev. of shock
    N_z::Int = 5              # Number of discrete productivity states
    mc_z = tauchen(5, 0.9, 0.4, 0.0)  # Discretization using Tauchen's method
    λ_z = stationary_distributions(tauchen(5,0.9,0.4,0.0))[1]
    # Normalize z so that E[z] = 1
    z_vec::Vector{Float64} = exp.(tauchen(5,0.9,0.4,0.0).state_values) ./ 
                              sum(exp.(tauchen(5,0.9,0.4,0.0).state_values) .* 
                              stationary_distributions(tauchen(5,0.9,0.4,0.0))[1])

    # Asset grid
    a_min::Float64 = 0.0
    a_max::Float64 = 40.0
    N_a::Int = 500
    a_grid::Vector{Float64} = collect(range(0.0, 40.0, length=500))
end

# Firm parameters calibrated from λ = 0 economy.
@with_kw struct FirmParams
    α::Float64 = 1/3         # Capital share (from US labor share data)
    δ::Float64 = 0.06        # Depreciation (calibrated so that δK/(Y)=0.2)
    A::Float64 = 0.8775      # Productivity parameter (from firm FOC)
end

# Given r, solve for firm equilibrium variables (with L = 1)
function solve_firm(r, firm::FirmParams)
    α = firm.α; δ = firm.δ; A = firm.A
    K = ((α*A)/(r+δ))^(1/(1-α))
    w = (1-α)*A*K^α
    Y = A*K^α  # Output (with L = 1)
    return K, w, Y
end

#########################
# --- Household Problem Solver (VFI)
#########################

"""
    solve_household(problem, r, w, τ, λ_tax; tol, maxiter)

Solves the household's Bellman equation with post‐tax income:
   c = (1-τ)*w*z^(1-λ_tax) + (1+r)*a - a′,
subject to a′ ≥ a_min. Returns value function V, policy functions for a′ and c,
the stationary distribution λ_dist over (a,z), and aggregate assets.
"""
function solve_household(problem::HAProblemTax, r, w, τ, λ_tax; tol=1e-6, maxiter=1000)
    a_grid = problem.a_grid
    N_a = length(a_grid)
    z_vec = problem.z_vec
    N_z = length(z_vec)
    β = problem.β
    γ = problem.γ
    
    # Precompute post-tax income for each z (note: with normalization E[z]=1, ȳ = w)
    income = [(1-τ)*w*(z^(1-λ_tax)) for z in z_vec]
    
    # Initialize value function V and policies
    V = zeros(N_a, N_z)
    policy_a = zeros(N_a, N_z)
    policy_c = zeros(N_a, N_z)
    
    # Initial guess: consume everything if choosing lowest asset
    for i in 1:N_a, j in 1:N_z
        c = income[j] + (1+r)*a_grid[i] - a_grid[1]
        V[i,j] = c > 0 ? u(c, γ) : -Inf
    end
    
    diff = 1.0
    iter = 0
    while diff > tol && iter < maxiter
        V_new = similar(V)
        for j in 1:N_z
            for i in 1:N_a
                best_val = -Inf
                best_a_prime = a_grid[1]
                best_c = 1e-10
                for k in 1:N_a
                    a_prime = a_grid[k]
                    c = income[j] + (1+r)*a_grid[i] - a_prime
                    if c <= 0
                        val = -Inf
                    else
                        # Expected continuation value using Tauchen's transition matrix
                        EV = 0.0
                        for j_next in 1:N_z
                            EV += problem.mc_z.p[j, j_next] * V[k, j_next]
                        end
                        val = u(c, γ) + β * EV
                    end
                    if val > best_val
                        best_val = val
                        best_a_prime = a_prime
                        best_c = c
                    end
                end
                V_new[i,j] = best_val
                policy_a[i,j] = best_a_prime
                policy_c[i,j] = best_c
            end
        end
        diff = maximum(abs.(V_new - V))
        V = V_new
        iter += 1
    end
    println("Household VFI converged in $iter iterations with diff = $diff")
    
    # --- Compute Stationary Distribution over (a,z)
    N = N_a * N_z
    Q = zeros(N, N)
    for j in 1:N_z
        for i in 1:N_a
            idx = (j-1)*N_a + i
            # Locate a′ chosen: here we assume it falls on grid (or use nearest-neighbor)
            a_prime = policy_a[i,j]
            k = searchsortedfirst(a_grid, a_prime)
            for j_next in 1:N_z
                idx_next = (j_next-1)*N_a + k
                Q[idx, idx_next] += problem.mc_z.p[j, j_next]
            end
        end
    end
    dist = ones(1, N) / N
    for it in 1:10000
        dist = dist * Q
    end
    λ_dist = reshape(dist, (N_a, N_z))
    agg_assets = sum(λ_dist .* repeat(a_grid, 1, N_z))
    
    return V, policy_a, policy_c, λ_dist, agg_assets
end

#########################
# --- Calibrating β in the λ = 0 Economy
#########################

# In the flat tax (λ = 0) baseline, we set:
r0 = 0.04         # given interest rate
τ0 = 0.3          # tax rate (flat tax: T(y) = τ*y)
# Firm: wage is set to 1 by calibration
firm = FirmParams()  # uses α = 1/3, δ = 0.06, A = 0.8775

K_target, w_target, Y_target = solve_firm(r0, firm)
println("Baseline firm: K = $K_target, w = $w_target, Y = $Y_target")

# Define a function returning excess asset demand (aggregate assets - K_target)
function market_clearing_beta(β_guess, problem::HAProblemTax, r, w, τ, λ_tax, K_target)
    prob = HAProblemTax(; β=β_guess, ρ=problem.ρ, σ_ϵ=problem.σ_ϵ, N_z=problem.N_z,
                        a_min=problem.a_min, a_max=problem.a_max, N_a=problem.N_a, a_grid=problem.a_grid,
                        z_vec=problem.z_vec)
    _, _, _, _, agg_assets = solve_household(prob, r, w, τ, λ_tax)
    return agg_assets - K_target
end

# Create a baseline problem (λ = 0) with default parameters
problem0 = HAProblemTax()
# Calibrate β so that aggregate assets equal K_target
β_calibrated = find_zero(β -> market_clearing_beta(β, problem0, r0, w_target, τ0, 0.0, K_target), (0.8, 0.99))
println("Calibrated β (λ=0): $β_calibrated")

# Update problem0 with the calibrated β
problem0 = HAProblemTax(; β=β_calibrated, ρ=problem0.ρ, σ_ϵ=problem0.σ_ϵ, N_z=problem0.N_z,
                         a_min=problem0.a_min, a_max=problem0.a_max, N_a=problem0.N_a,
                         a_grid=problem0.a_grid, z_vec=problem0.z_vec)

# Solve household problem for baseline (λ = 0)
V0, policy_a0, policy_c0, λ_dist0, agg_assets0 = solve_household(problem0, r0, w_target, τ0, 0.0)
println("Baseline (λ=0): Aggregate household assets = $agg_assets0, Target = $K_target")

# Compute marginal distributions
λ_a0 = sum(λ_dist0, dims=2)  # asset distribution over grid
# After-tax labor income for each productivity state: ỹ = (1-τ0)*w*z
income0 = [ (1-τ0)*w_target*z for z in problem0.z_vec ]
# Gini for assets and income (using grid weights from the stationary distribution)
gini_assets0 = gini_coefficient(problem0.a_grid, vec(λ_a0))
gini_income0 = gini_coefficient(income0, problem0.λ_z)
println("Baseline Gini coefficients: Assets = $(gini_assets0), Income = $(gini_income0)")

#########################
# --- Equilibrium in the λ = 0.15 Economy
#########################

# In the progressive tax case we set λ_tax = 0.15 and keep all other parameters fixed.
λ_tax_reform = 0.15

"""
For the reform economy, government revenue is given by:
   G = w*(1 - (1-τ)*E[z^(1-λ_tax_reform)]),
and we require G/Y = 0.2.
Thus, τ is determined by:
   τ = 1 - (1 - (0.2*Y)/w) / m,    where m = E[z^(1-λ_tax_reform)].
"""
function market_excess_r(r, problem::HAProblemTax, firm::FirmParams, λ_tax)
    K, w, Y = solve_firm(r, firm)
    m = sum(problem.z_vec.^(1-λ_tax) .* problem.λ_z)
    τ = 1 - (1 - (0.2*Y)/w) / m
    # Solve household problem with these prices and tax parameters
    _, _, _, _, agg_assets = solve_household(problem, r, w, τ, λ_tax)
    excess = agg_assets - K
    return excess, τ, K, w, Y, agg_assets
end

# Find equilibrium r (using problem0 with calibrated β) such that market clears
r_eq = find_zero(r -> market_excess_r(r, problem0, firm, λ_tax_reform)[1], 0.02)

excess, τ_eq, K_eq, w_eq, Y_eq, agg_assets_eq = market_excess_r(r_eq, problem0, firm, λ_tax_reform)
println("Reform Economy (λ=0.15): Equilibrium r = $r_eq, w = $w_eq, τ = $τ_eq")
println("Reform Economy: K = $K_eq, Y = $Y_eq, Aggregate household assets = $agg_assets_eq")
println("Reform Capital-to-Output ratio: $(K_eq/Y_eq)")

# Compute distributions and Gini coefficients for reform economy
_, _, _, _, _, agg_assets_temp, V_reform, policy_a_reform, policy_c_reform, λ_dist_reform = 
    let r_temp = r_eq
        V, pa, pc, λd, agg = solve_household(problem0, r_eq, w_eq, τ_eq, λ_tax_reform)
        (V, pa, pc, λd, agg)
    end

V_reform, policy_a_reform, policy_c_reform, λ_dist_reform, agg_assets_temp = solve_household(problem0, r_eq, w_eq, τ_eq, λ_tax_reform)


λ_a_reform = sum(λ_dist_reform, dims=2)
income_reform = [ (1-τ_eq)*w_eq*(z^(1-λ_tax_reform)) for z in problem0.z_vec ]
gini_assets_reform = gini_coefficient(problem0.a_grid, vec(λ_a_reform))
gini_income_reform = gini_coefficient(income_reform, problem0.λ_z)
println("Reform Gini coefficients: Assets = $(gini_assets_reform), Income = $(gini_income_reform)")

#########################
# --- Plots
#########################

# Plot value functions for baseline vs. reform for a mid productivity state (e.g., 3rd state)
plot(problem0.a_grid, V0[:,3], label="V (λ=0)", lw=2, xlabel="Assets", ylabel="Value", title="Value Functions")
plot!(problem0.a_grid, V_reform[:,3], label="V (λ=0.15)", lw=2)

# Plot policy functions (asset savings) for baseline vs. reform for the lowest and mid productivity states
p1 = plot(problem0.a_grid, policy_a0[:,1], label="Policy a' (λ=0, low z)", lw=2, xlabel="Assets", ylabel="a'", title="Policy Functions")
plot!(problem0.a_grid, policy_a0[:,3], label="Policy a' (λ=0, mid z)", lw=2)
plot!(problem0.a_grid, policy_a_reform[:,1], label="Policy a' (λ=0.15, low z)", lw=2, linestyle=:dash)
plot!(problem0.a_grid, policy_a_reform[:,3], label="Policy a' (λ=0.15, mid z)", lw=2, linestyle=:dash)

# Plot marginal distribution of assets (summing over productivity)
p2 = plot(problem0.a_grid, vec(λ_a0), label="λ(a) (λ=0)", lw=2, xlabel="Assets", ylabel="Density", title="Asset Distribution")
plot!(problem0.a_grid, vec(λ_a_reform), label="λ(a) (λ=0.15)", lw=2, linestyle=:dash)

# Plot Lorenz curves for after-tax labor income
cum_pop0, cum_income0 = lorenz_curve(income0, problem0.λ_z)
cum_pop_reform, cum_income_reform = lorenz_curve(income_reform, problem0.λ_z)
p3 = plot(cum_pop0, cum_income0, label="Lorenz Income (λ=0)", lw=2, xlabel="Cumulative Population", ylabel="Cumulative Income", title="Lorenz Curves")
plot!(cum_pop_reform, cum_income_reform, label="Lorenz Income (λ=0.15)", lw=2, linestyle=:dash)

# Plot Lorenz curves for assets
λ_a0_vec = vec(λ_a0)
pops_a0 = cumsum(λ_a0_vec) / sum(λ_a0_vec)
lorenz_a0 = cumsum(problem0.a_grid .* λ_a0_vec) / sum(problem0.a_grid .* λ_a0_vec)
λ_a_reform_vec = vec(λ_a_reform)
pops_a_reform = cumsum(λ_a_reform_vec) / sum(λ_a_reform_vec)
lorenz_a_reform = cumsum(problem0.a_grid .* λ_a_reform_vec) / sum(problem0.a_grid .* λ_a_reform_vec)
p4 = plot(pops_a0, lorenz_a0, label="Lorenz Assets (λ=0)", lw=2, xlabel="Cumulative Population", ylabel="Cumulative Assets", title="Lorenz Curves for Assets")
plot!(pops_a_reform, lorenz_a_reform, label="Lorenz Assets (λ=0.15)", lw=2, linestyle=:dash)

#########################
# --- Summary Output
#########################

println("\n--- Summary Statistics ---")
println("Baseline (λ=0):")
println("  r = $r0, w = $w_target, τ = $τ0")
println("  Capital-to-Output ratio: $(K_target/Y_target)")
println("  Gini: Assets = $(gini_assets0), Income = $(gini_income0)")
println("\nReform (λ=0.15):")
println("  r = $(r_eq), w = $(w_eq), τ = $(τ_eq)")
println("  Capital-to-Output ratio: $(K_eq/Y_eq)")
println("  Gini: Assets = $(gini_assets_reform), Income = $(gini_income_reform)")
