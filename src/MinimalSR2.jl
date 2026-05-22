module MinimalSR2

# This file is intentionally broad pseudocode. It sketches a second MinimalSR
# architecture where the engine loop is policy-neutral and Basic/PySR behavior
# is expressed only by hyperparameters plus a fixed set of callback functions.

# ─── Core data ──────────────────────────────────────────────────────────────

abstract type AbstractNode end

struct Individual
    tree::AbstractNode
    loss::Float64
    cost::Float64
    complexity::Int
    birth::Int
    parent_ref::Union{Int, Nothing}
end

mutable struct Population
    members::Vector{Individual}
end

mutable struct Archive
    members::Vector{Individual}
end

mutable struct SearchStats
    data::Dict{Symbol, Any}
end

mutable struct EngineState
    populations::Vector{Population}
    archive::Archive
    stats::SearchStats
    rng
    birth_counter::Int
    eval_count::Int
end

Base.@kwdef struct Hyperparameters
    population_size::Int = 100
    populations::Int = 1
    niterations::Int = 100
    ncycles_per_iteration::Int = 10
    maxsize::Int = 30
    maxdepth::Int = 10
    topn::Int = 10
    crossover_probability::Float64 = 0.0
    tournament_selection_n::Int = 2
    tournament_selection_p::Float64 = 1.0
    archive_fraction_replaced::Float64 = 0.0
    migration_fraction_replaced::Float64 = 0.0
    annealing::Bool = false
    alpha::Float64 = 1.0
    should_simplify::Bool = true
    should_optimize_constants::Bool = true
    optimizer_iterations::Int = 8
    optimize_probability::Float64 = 0.0
    use_frequency::Bool = false
    use_frequency_in_tournament::Bool = false
    adaptive_parsimony_scaling::Float64 = 0.0
    random_state::Int = 0
    max_evals::Union{Int, Nothing} = nothing
end

Base.@kwdef struct MinimalSRPolicy
    loss_function::Function
    survival::Function
    selection::Function
    mutation::Function
    acceptance::Function
    crossover::Function
    update_archive::Function
    update_population::Function
    update_stats::Function
end

Base.@kwdef struct MinimalSRConfig
    h::Hyperparameters = Hyperparameters()
    policy::MinimalSRPolicy
end

# ─── Policy callback contracts ──────────────────────────────────────────────

# loss_function(tree, X, y, state, config) -> (; loss, cost, complexity) | nothing
# survival(population, candidates, state, config) -> Population
# selection(population, state, config) -> Individual
# mutation(parent, state, config) -> tree | nothing
# acceptance(parent, child, state, config) -> Bool
# crossover(parent_a, parent_b, state, config) -> tree | nothing
# update_archive(archive, populations, state, config) -> Archive
# update_population(archive, populations, state, config) -> Vector{Population}
# update_stats(populations, state, config) -> SearchStats

# These contracts are intentionally state-rich. The callbacks can ignore state
# for Basic behavior, or use stats/frequencies/archive for PySR. Any temperature
# or annealing state is policy-specific data owned by `update_stats`.

# ─── Generic engine ─────────────────────────────────────────────────────────

function fit_minimal_sr2(X, y, variable_names; config::MinimalSRConfig)
    state = initialize_state(X, y, variable_names, config)
    initialize_populations!(state, X, y, config)
    state.archive = config.policy.update_archive(state.archive, state.populations, state, config)

    for iteration in 1:config.h.niterations
        has_budget(state, config) || break
        state.stats = config.policy.update_stats(state.populations, state, config)

        for pop_index in eachindex(state.populations)
            for _cycle in 1:config.h.ncycles_per_iteration
                regularized_cycle!(state, pop_index, X, y, config)
                state.stats = config.policy.update_stats(state.populations, state, config)
                has_budget(state, config) || break
            end
            has_budget(state, config) || break
        end

        optimize_and_simplify_populations!(state.populations, X, y, state, config)
        state.archive = config.policy.update_archive(state.archive, state.populations, state, config)
        state.populations = config.policy.update_population(
            state.archive, state.populations, state, config
        )
        state.stats = config.policy.update_stats(state.populations, state, config)
    end

    return format_result(state.archive, state, config)
end

function regularized_cycle!(state::EngineState, pop_index::Int, X, y, config::MinimalSRConfig)
    h = config.h
    p = config.policy
    population = state.populations[pop_index]
    ncycles = cycles_for_population(population, h)

    for _ in 1:ncycles
        has_budget(state, config) || break

        parent = p.selection(population, state, config)
        child_tree = if should_crossover(state.rng, h)
            other_parent = p.selection(population, state, config)
            p.crossover(parent, other_parent, state, config)
        else
            p.mutation(parent, state, config)
        end
        child_tree === nothing && continue

        child = make_individual(child_tree, X, y, state, config)
        child === nothing && continue

        if p.acceptance(parent, child, state, config)
            state.populations[pop_index] = p.survival(population, [child], state, config)
            population = state.populations[pop_index]
        end
    end

    return state
end

function optimize_and_simplify_populations!(populations, X, y, state, config)
    # Common loop step for both Basic and PySR variants. Hyperparameters decide
    # whether this does nothing, simplification only, constant optimization only,
    # or both. This is intentionally not a policy callback.
end

# ─── Basic config sketch ────────────────────────────────────────────────────

function basic_loss_function(tree, X, y, state, config)
    # Evaluate tree on X; return MSE plus complexity-derived cost.
end

function basic_selection(population, state, config)
    # n=2 tournament; pick the lower-cost individual.
end

function basic_survival(population, candidates, state, config)
    # Append candidates; keep top-k / population_size by fitness.
end

function basic_mutation(parent, state, config)
    # Replace one random node with a terminal or small random subtree.
end

function basic_crossover(parent_a, parent_b, state, config)
    # Swap random subtrees.
end

function basic_acceptance(parent, child, state, config)
    # Always accept valid children; survival decides whether they remain.
    return true
end

function basic_update_archive(archive, populations, state, config)
    # All-time top-k by loss/cost.
end

function basic_update_population(archive, populations, state, config)
    # No archive migration or population rewrite.
    return populations
end

function basic_update_stats(populations, state, config)
    # No running statistics.
    return state.stats
end

function basic_config(; kwargs...)
    h = Hyperparameters(;
        tournament_selection_n=2,
        tournament_selection_p=1.0,
        crossover_probability=0.0,
        archive_fraction_replaced=0.0,
        migration_fraction_replaced=0.0,
        should_simplify=true,
        should_optimize_constants=true,
        use_frequency=false,
        use_frequency_in_tournament=false,
        kwargs...,
    )
    policy = MinimalSRPolicy(;
        loss_function=basic_loss_function,
        survival=basic_survival,
        selection=basic_selection,
        mutation=basic_mutation,
        acceptance=basic_acceptance,
        crossover=basic_crossover,
        update_archive=basic_update_archive,
        update_population=basic_update_population,
        update_stats=basic_update_stats,
    )
    return MinimalSRConfig(; h, policy)
end

# ─── PySR-compatible config sketch ──────────────────────────────────────────

function pysr_loss_function(tree, X, y, state, config)
    # MSE with PySR-compatible cost/parsimony handling.
end

function pysr_selection(population, state, config)
    # Adaptive-parsimony tournament selection. Uses normalized complexity
    # frequencies in state.stats when use_frequency_in_tournament is enabled.
end

function pysr_survival(population, candidates, state, config)
    # Age-regularized survival: insert accepted candidate, remove oldest member
    # outside any protected set needed by the selection/survival contract.
end

function pysr_mutation(parent, state, config)
    # One mega-mutation callback:
    # 1. condition PySR mutation weights on parent tree and hyperparameters
    # 2. sample mutation type once
    # 3. retry that same mutation up to N times for validity
    # 4. return the first valid tree or nothing
end

function pysr_crossover(parent_a, parent_b, state, config)
    # PySR-compatible subtree crossover, including validity retries if needed.
end

function pysr_acceptance(parent, child, state, config)
    # Annealing factor from cost delta, alpha, and temperature stored in stats.
    # Frequency factor old_frequency / new_frequency when use_frequency is set.
end

function pysr_update_archive(archive, populations, state, config)
    # Maintain best-by-complexity, then expose Pareto frontier by complexity/loss.
end

function pysr_update_population(archive, populations, state, config)
    # Applies both PySR-style subpopulation migration and archive/HOF migration.
    # Because this callback sees all populations, the generic engine does not need
    # separate migration branches.
end

function pysr_update_stats(populations, state, config)
    # Update running complexity frequencies, normalize them, move the approximate
    # window, and track counters needed by PySR-compatible selection/acceptance.
    # This is also where PySR-specific temperature/annealing state is advanced.
    # Stats may be global, per-population, or both, depending on what parity needs.
end

function pysr_config(; kwargs...)
    h = Hyperparameters(;
        tournament_selection_n=3,
        tournament_selection_p=0.982,
        crossover_probability=0.2,
        archive_fraction_replaced=0.0614,
        migration_fraction_replaced=0.00036,
        annealing=false,
        alpha=3.17,
        should_simplify=true,
        should_optimize_constants=true,
        optimize_probability=0.14,
        use_frequency=true,
        use_frequency_in_tournament=true,
        adaptive_parsimony_scaling=1040.0,
        kwargs...,
    )
    policy = MinimalSRPolicy(;
        loss_function=pysr_loss_function,
        survival=pysr_survival,
        selection=pysr_selection,
        mutation=pysr_mutation,
        acceptance=pysr_acceptance,
        crossover=pysr_crossover,
        update_archive=pysr_update_archive,
        update_population=pysr_update_population,
        update_stats=pysr_update_stats,
    )
    return MinimalSRConfig(; h, policy)
end

# ─── Open design questions ──────────────────────────────────────────────────

# Resolved design choices:
#
# 1. The engine is multi-population by default. `update_population` and
#    `update_stats` receive the full `Vector{Population}` so PySR can express
#    subpopulation migration and archive/HOF migration without special engine
#    branches. Basic config sets `populations=1` or makes these callbacks no-op.
#
# 2. Simplification and constant optimization are common loop stages controlled
#    by hyperparameters, not policy callbacks. Both Basic and PySR pass through
#    the same `optimize_and_simplify_populations!` hook.
#
# 3. `update_archive` runs after the local evolution plus common optimize/
#    simplify step. `update_population` runs after archive update, then
#    `update_stats` observes the post-migration populations. If exact MiniSR
#    parity requires pre-migration stats, the least invasive adjustment is to
#    move `update_stats` before `update_population` in the generic loop.
#
# 4. Acceptance remains parent-child, before survival. Survival receives only
#    accepted children and decides which population members remain. Temperature
#    is not a generic engine concept; PySR acceptance reads it from `state.stats`,
#    and PySR `update_stats` decides how and when to advance it.
#
# Still open:
#
# - Archive representation. Basic only needs top-k all-time; PySR likely wants
#   best-by-complexity plus a rendered Pareto frontier. `Archive` should probably
#   become a structured object instead of just `members::Vector{Individual}`.

end # module MinimalSR2
