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
    population::Population
    archive::Archive
    stats::SearchStats
    rng
    birth_counter::Int
    eval_count::Int
end

Base.@kwdef struct Hyperparameters
    population_size::Int = 100
    niterations::Int = 100
    ncycles_per_iteration::Int = 10
    maxsize::Int = 30
    maxdepth::Int = 10
    topn::Int = 10
    crossover_probability::Float64 = 0.0
    tournament_selection_n::Int = 2
    tournament_selection_p::Float64 = 1.0
    archive_fraction_replaced::Float64 = 0.0
    annealing::Bool = false
    alpha::Float64 = 1.0
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
# acceptance(parent, child, temperature, state, config) -> Bool
# crossover(parent_a, parent_b, state, config) -> tree | nothing
# update_archive(archive, population, state, config) -> Archive
# update_population(archive, population, state, config) -> Population
# update_stats(population, state, config) -> SearchStats

# These contracts are intentionally state-rich. The callbacks can ignore state
# for Basic behavior, or use stats/temperature/frequencies/archive for PySR.

# ─── Generic engine ─────────────────────────────────────────────────────────

function fit_minimal_sr2(X, y, variable_names; config::MinimalSRConfig)
    state = initialize_state(X, y, variable_names, config)
    initialize_population!(state, X, y, config)
    state.archive = config.policy.update_archive(state.archive, state.population, state, config)

    for iteration in 1:config.h.niterations
        has_budget(state, config) || break
        temperature_schedule = temperatures_for_iteration(iteration, config)

        for temperature in temperature_schedule
            regularized_cycle!(state, X, y, temperature, config)
            has_budget(state, config) || break
        end

        state.archive = config.policy.update_archive(state.archive, state.population, state, config)
        state.population = config.policy.update_population(
            state.archive, state.population, state, config
        )
        state.stats = config.policy.update_stats(state.population, state, config)
    end

    return format_result(state.archive, state, config)
end

function regularized_cycle!(state::EngineState, X, y, temperature::Float64, config::MinimalSRConfig)
    h = config.h
    p = config.policy
    ncycles = cycles_for_population(state.population, h)

    for _ in 1:ncycles
        has_budget(state, config) || break

        parent = p.selection(state.population, state, config)
        child_tree = if should_crossover(state.rng, h)
            other_parent = p.selection(state.population, state, config)
            p.crossover(parent, other_parent, state, config)
        else
            p.mutation(parent, state, config)
        end
        child_tree === nothing && continue

        child = make_individual(child_tree, X, y, state, config)
        child === nothing && continue

        if p.acceptance(parent, child, temperature, state, config)
            state.population = p.survival(state.population, [child], state, config)
        end
    end

    return state
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

function basic_acceptance(parent, child, temperature, state, config)
    # Always accept valid children; survival decides whether they remain.
    return true
end

function basic_update_archive(archive, population, state, config)
    # All-time top-k by loss/cost.
end

function basic_update_population(archive, population, state, config)
    # No archive migration or population rewrite.
    return population
end

function basic_update_stats(population, state, config)
    # No running statistics.
    return state.stats
end

function basic_config(; kwargs...)
    h = Hyperparameters(;
        tournament_selection_n=2,
        tournament_selection_p=1.0,
        crossover_probability=0.0,
        archive_fraction_replaced=0.0,
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

function pysr_acceptance(parent, child, temperature, state, config)
    # Annealing factor from cost delta and alpha.
    # Frequency factor old_frequency / new_frequency when use_frequency is set.
end

function pysr_update_archive(archive, population, state, config)
    # Maintain best-by-complexity, then expose Pareto frontier by complexity/loss.
end

function pysr_update_population(archive, population, state, config)
    # Migrate/replace a configured fraction of the population from archive.
    # If we need subpopulation-to-subpopulation migration, this callback likely
    # needs either all populations or a richer PopulationManager state.
end

function pysr_update_stats(population, state, config)
    # Update running complexity frequencies, normalize them, move the approximate
    # window, and track counters needed by PySR-compatible selection/acceptance.
end

function pysr_config(; kwargs...)
    h = Hyperparameters(;
        tournament_selection_n=3,
        tournament_selection_p=0.982,
        crossover_probability=0.2,
        archive_fraction_replaced=0.0614,
        annealing=false,
        alpha=3.17,
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

# 1. Should update_population see only one population, or all populations?
#    PySR has both archive/HOF migration and inter-subpopulation migration.
#
# 2. Should update_archive return a full archive object plus a public frontier,
#    or should Archive itself store both best-by-complexity and displayed rows?
#
# 3. Should update_stats run before or after update_population? PySR currently
#    counts the evolved population before archive migration in the MiniSR sketch.
#
# 4. Should constant optimization and simplification be separate callbacks, or
#    should they live inside mutation/update_population/update_stats? They are
#    not in the requested nine functions, but PySR compatibility needs a place
#    for them.
#
# 5. Should acceptance happen before survival, as sketched, or should survival
#    own acceptance by receiving all candidate children? PySR acceptance is a
#    parent-child decision, while top-k survival can be population-wide.

end # module MinimalSR2
