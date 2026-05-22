module MinimalSR2

import ..MinimalSR

const MS = MinimalSR
const Node = MS.Node
const Individual = MS.Individual
const Population = Vector{Individual}

# ─── State/config ───────────────────────────────────────────────────────────

abstract type AbstractArchiveState end
abstract type AbstractSearchStatsState end

mutable struct BasicArchiveState <: AbstractArchiveState
    members::Vector{Individual}
end

BasicArchiveState() = BasicArchiveState(Individual[])

struct BasicStatsState <: AbstractSearchStatsState end

mutable struct PySRArchiveState <: AbstractArchiveState
    best_by_complexity::Dict{Int, Individual}
    frontier::Vector{Individual}
end

PySRArchiveState() = PySRArchiveState(Dict{Int, Individual}(), Individual[])

mutable struct PySRStatsState <: AbstractSearchStatsState
    per_population::Vector{MS.RunningSearchStatistics}
    current_temperature::Float64
    counted_population_cycles::Vector{Int}
end

function PySRStatsState(cfg::MS.EngineConfig)
    return PySRStatsState(
        [MS.RunningSearchStatistics(cfg.maxsize) for _ in 1:max(1, cfg.populations)],
        1.0,
        zeros(Int, max(1, cfg.populations)),
    )
end

mutable struct EngineState{
    A<:AbstractArchiveState,
    S<:AbstractSearchStatsState,
}
    engine::MS.RegularizedEvolutionEngine
    populations::Vector{Population}
    archive::A
    stats::S
    current_iteration::Int
    current_population::Int
    current_inner_cycle::Int
    completed_population_cycles::Vector{Int}
end

Base.@kwdef struct MinimalSRPolicy
    init_archive::Function
    init_stats::Function
    loss_function::Function
    survival::Function
    selection::Function
    mutation::Function
    acceptance::Function
    crossover::Function
    update_archive!::Function
    update_population::Function
    update_stats!::Function
end

Base.@kwdef struct MinimalSRConfig
    engine_config::MS.EngineConfig
    policy::MinimalSRPolicy
end

engine(state::EngineState) = state.engine
current_stats(state::EngineState{<:AbstractArchiveState, PySRStatsState}) =
    state.stats.per_population[state.current_population]

archive_members(archive::BasicArchiveState) = archive.members
archive_members(archive::PySRArchiveState) = archive.frontier

function engine_config_from_kwargs(; kwargs...)
    return MS.EngineConfig(; kwargs...)
end

function engine_config_from_namedtuple(nt::NamedTuple)
    engine_fields = Set(fieldnames(MS.EngineConfig))
    kwargs = Dict{Symbol, Any}()
    for key in keys(nt)
        key in engine_fields || continue
        kwargs[key] = nt[key]
    end
    return MS.EngineConfig(; kwargs...)
end

function initialize_state(X, y, variable_names, config::MinimalSRConfig)
    _ = variable_names
    eng = MS.RegularizedEvolutionEngine(Matrix{Float64}(X), Float64.(vec(y)), config.engine_config)
    archive = config.policy.init_archive(config)
    stats = config.policy.init_stats(config)
    return EngineState(
        eng,
        Population[],
        archive,
        stats,
        0,
        1,
        1,
        Int[],
    )
end

function initialize_populations!(state::EngineState, _X, _y, _config::MinimalSRConfig)
    n = max(1, state.engine.cfg.populations)
    state.populations = [MS.initialize_population(state.engine) for _ in 1:n]
    state.completed_population_cycles = zeros(Int, n)
    return nothing
end

has_budget(state::EngineState, _config::MinimalSRConfig) = MS.has_budget(state.engine)

function cycles_for_population(population::Population, cfg::MS.EngineConfig)
    return Int(ceil(length(population) / max(1, cfg.tournament_selection_n)))
end

should_crossover(rng, cfg::MS.EngineConfig, population::Population) =
    length(population) >= 2 && rand(rng) <= cfg.crossover_probability

function make_individual(tree::Node, _X, _y, state::EngineState, config::MinimalSRConfig;
                         parent_ref::Union{Int, Nothing}=nothing)
    out = config.policy.loss_function(tree, state, config)
    out === nothing && return nothing
    loss, cost, complexity = out
    return Individual(
        tree,
        loss,
        cost,
        complexity,
        MS.next_birth!(state.engine),
        MS.next_ref!(state.engine),
        parent_ref,
    )
end

function trees_from_crossover_result(result)
    result === nothing && return Node[]
    result isa Node && return Node[result]
    result isa Tuple && return Node[tree for tree in result if tree isa Node]
    result isa AbstractVector && return Node[tree for tree in result if tree isa Node]
    return Node[]
end

# ─── Generic engine ─────────────────────────────────────────────────────────

function fit_minimal_sr2(X_in, y_in, variable_names_in; config::MinimalSRConfig)
    X = MS.as_matrix(X_in)
    y = MS.as_vector(y_in)
    variable_names = MS.as_strings(variable_names_in)
    state = initialize_state(X, y, variable_names, config)
    initialize_populations!(state, X, y, config)
    config.policy.update_archive!(state.archive, state.populations, state, config)

    for iteration in 1:config.engine_config.niterations
        has_budget(state, config) || break
        state.current_iteration = iteration

        for pop_index in eachindex(state.populations)
            has_budget(state, config) || break
            state.current_population = pop_index

            for inner in 1:max(1, config.engine_config.ncycles_per_iteration)
                has_budget(state, config) || break
                state.current_inner_cycle = inner
                config.policy.update_stats!(state.populations, state, config)
                regularized_cycle!(state, pop_index, X, y, config)
            end

            has_budget(state, config) && optimize_and_simplify_population!(
                state.populations[pop_index], state, config
            )
            state.completed_population_cycles[pop_index] += 1
            config.policy.update_archive!(state.archive, state.populations, state, config)
            config.policy.update_stats!(state.populations, state, config)
            state.populations = config.policy.update_population(
                state.archive, state.populations, state, config
            )
        end
    end

    return format_result(state.archive, state, variable_names)
end

function regularized_cycle!(state::EngineState, pop_index::Int, X, y, config::MinimalSRConfig)
    cfg = config.engine_config
    policy = config.policy
    population = state.populations[pop_index]

    for _ in 1:cycles_for_population(population, cfg)
        has_budget(state, config) || break

        if should_crossover(state.engine.rng, cfg, population)
            parent_a = policy.selection(population, state, config)
            parent_b = policy.selection(population, state, config)
            result = policy.crossover(parent_a, parent_b, state, config)
            candidates = Individual[]
            for (tree, parent) in zip(trees_from_crossover_result(result), (parent_a, parent_b))
                child = make_individual(tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                push!(candidates, child)
            end
            isempty(candidates) && cfg.skip_mutation_failures && continue
            state.populations[pop_index] = policy.survival(population, candidates, state, config)
        else
            parent = policy.selection(population, state, config)
            child_tree = policy.mutation(parent, state, config)
            if child_tree === nothing
                cfg.skip_mutation_failures && continue
                replacement = MS.spawn_from_existing(state.engine, parent)
                state.populations[pop_index] = policy.survival(
                    population, [replacement], state, config
                )
            else
                child = make_individual(child_tree, X, y, state, config; parent_ref=parent.ref)
                child === nothing && return state
                if policy.acceptance(parent, child, state, config)
                    state.populations[pop_index] = policy.survival(
                        population, [child], state, config
                    )
                elseif !cfg.skip_mutation_failures
                    replacement = MS.spawn_from_existing(state.engine, parent)
                    state.populations[pop_index] = policy.survival(
                        population, [replacement], state, config
                    )
                end
            end
        end
        population = state.populations[pop_index]
    end
    return state
end

function optimize_and_simplify_population!(
    population::Population, state::EngineState, _config::MinimalSRConfig
)
    return MS.optimize_and_simplify!(state.engine, population)
end

function format_result(
    archive::AbstractArchiveState, state::EngineState, variable_names::Vector{String}
)
    rows = Vector{Dict{String, Any}}()
    members = archive_members(archive)
    if isempty(members)
        best = state.populations[1][1]
        for pop in state.populations, member in pop
            member.cost < best.cost && (best = member)
        end
        members = [best]
    end
    for member in sort(members, by=m -> m.complexity)
        eqn = MS.node_string(member.tree)
        for i in reverse(eachindex(variable_names))
            eqn = replace(eqn, Regex("\\bx$(i - 1)\\b") => variable_names[i])
        end
        push!(rows, Dict("complexity" => member.complexity, "loss" => member.loss, "equation" => eqn))
    end
    return Dict("rows" => rows, "n_evals" => state.engine.eval_count)
end

# ─── Shared callbacks ───────────────────────────────────────────────────────

mse_loss_function(tree::Node, state::EngineState, _config::MinimalSRConfig) =
    MS.evaluate_candidate(state.engine, tree)

always_accept(_parent::Individual, _child::Individual, _state::EngineState, _config::MinimalSRConfig) =
    true

function subtree_swap_crossover(parent_a::Individual, parent_b::Individual, state::EngineState,
                                _config::MinimalSRConfig)
    for _attempt in 1:10
        t1, t2 = MS.default_crossover(state.engine, parent_a.tree, parent_b.tree)
        MS.valid_tree(state.engine, t1) && MS.valid_tree(state.engine, t2) && return (t1, t2)
    end
    return nothing
end

# ─── Basic policy ───────────────────────────────────────────────────────────

function basic_selection(population::Population, state::EngineState, config::MinimalSRConfig)
    idx = MS.simple_tournament_select(population, config.engine_config, state.engine.rng)
    return population[idx]
end

function basic_survival(population::Population, candidates::Vector{Individual},
                        _state::EngineState, config::MinimalSRConfig)
    output = copy(population)
    MS.topk_survive!(output, candidates, config.engine_config)
    return output
end

function basic_mutation(parent::Individual, state::EngineState, _config::MinimalSRConfig)
    return MS.default_replace_subtree_mutation(state.engine, parent)
end

init_basic_archive(_config::MinimalSRConfig) = BasicArchiveState()
init_basic_stats(_config::MinimalSRConfig) = BasicStatsState()

function basic_update_archive!(archive::BasicArchiveState, populations::Vector{Population},
                               _state::EngineState, config::MinimalSRConfig)
    combined = copy(archive.members)
    for pop in populations
        append!(combined, copy.(pop))
    end
    sort!(combined, by=m -> (m.loss, m.cost, m.complexity, m.birth))
    empty!(archive.members)
    seen = Set{String}()
    for member in combined
        isfinite(member.loss) || continue
        key = MS.node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(archive.members, copy(member))
        length(archive.members) >= max(1, config.engine_config.topn) && break
    end
    return nothing
end

basic_update_population(_archive::BasicArchiveState, populations::Vector{Population},
                        _state::EngineState, _config::MinimalSRConfig) = populations

basic_update_stats!(_populations::Vector{Population}, _state::EngineState,
                    _config::MinimalSRConfig) = nothing

function basic_policy()
    return MinimalSRPolicy(;
        init_archive=init_basic_archive,
        init_stats=init_basic_stats,
        loss_function=mse_loss_function,
        survival=basic_survival,
        selection=basic_selection,
        mutation=basic_mutation,
        acceptance=always_accept,
        crossover=subtree_swap_crossover,
        update_archive! = basic_update_archive!,
        update_population=basic_update_population,
        update_stats! = basic_update_stats!,
    )
end

function default_minimal_config(; kwargs...)
    nt = MS.default_minimal_config(; kwargs...)
    cfg = engine_config_from_namedtuple(nt)
    return MinimalSRConfig(; engine_config=cfg, policy=basic_policy())
end

fit_default_sr(args...; kwargs...) =
    fit_minimal_sr2(args...; config=default_minimal_config(; kwargs...))

# ─── PySR-compatible policy ─────────────────────────────────────────────────

function pysr_selection(population::Population, state::EngineState, config::MinimalSRConfig)
    stats = current_stats(state)
    idx = MS.tournament_select(population, stats, config.engine_config, state.engine.rng)
    return population[idx]
end

function pysr_survival(population::Population, candidates::Vector{Individual},
                       state::EngineState, _config::MinimalSRConfig)
    output = copy(population)
    used = Set{Int}()
    for candidate in candidates
        isempty(output) && break
        victim = MS.oldest_survival(output, state.engine.rng, used)
        push!(used, victim)
        output[victim] = candidate
    end
    return output
end

function pysr_mutation(parent::Individual, state::EngineState, _config::MinimalSRConfig)
    return MS.pysr_weighted_mutation(state.engine, parent)
end

function pysr_acceptance(parent::Individual, child::Individual, state::EngineState,
                         _config::MinimalSRConfig)
    return MS.accept_candidate(
        state.engine, parent, child, current_stats(state), state.stats.current_temperature
    )
end

init_pysr_archive(_config::MinimalSRConfig) = PySRArchiveState()
init_pysr_stats(config::MinimalSRConfig) = PySRStatsState(config.engine_config)

function pysr_update_archive!(archive::PySRArchiveState, populations::Vector{Population},
                              _state::EngineState, config::MinimalSRConfig)
    for pop in populations, member in pop
        c = member.complexity
        1 <= c <= config.engine_config.maxsize || continue
        best = get(archive.best_by_complexity, c, nothing)
        if isnothing(best) || member.loss < best.loss
            archive.best_by_complexity[c] = copy(member)
        end
    end
    archive.frontier = MS.calculate_pareto_frontier_from_dict(archive.best_by_complexity)
    return nothing
end

function pysr_update_population(archive::PySRArchiveState, populations::Vector{Population},
                                state::EngineState, _config::MinimalSRConfig)
    MS.default_migration(state.engine, populations, state.current_population, archive.frontier)
    return populations
end

function pysr_update_stats!(populations::Vector{Population}, state::EngineState,
                            config::MinimalSRConfig)
    cfg = config.engine_config
    stats = current_stats(state)

    MS.normalize!(stats)
    if cfg.annealing && cfg.ncycles_per_iteration > 1
        denom = max(1, cfg.ncycles_per_iteration - 1)
        state.stats.current_temperature = 1.0 - (state.current_inner_cycle - 1) / denom
    else
        state.stats.current_temperature = 1.0
    end
    state.engine.current_temperature = clamp(state.stats.current_temperature, 0.0, 1.0)

    pop_idx = state.current_population
    completed = state.completed_population_cycles[pop_idx]
    if completed > state.stats.counted_population_cycles[pop_idx]
        pop = populations[state.current_population]
        for member in pop
            MS.update_size!(stats, member.complexity)
        end
        MS.move_window!(stats)
        state.stats.counted_population_cycles[pop_idx] = completed
    end
    return nothing
end

function pysr_policy()
    return MinimalSRPolicy(;
        init_archive=init_pysr_archive,
        init_stats=init_pysr_stats,
        loss_function=mse_loss_function,
        survival=pysr_survival,
        selection=pysr_selection,
        mutation=pysr_mutation,
        acceptance=pysr_acceptance,
        crossover=subtree_swap_crossover,
        update_archive! = pysr_update_archive!,
        update_population=pysr_update_population,
        update_stats! = pysr_update_stats!,
    )
end

function pysr_compat_config(; kwargs...)
    nt = MS.pysr_compat_config(; kwargs...)
    cfg = engine_config_from_namedtuple(nt)
    return MinimalSRConfig(; engine_config=cfg, policy=pysr_policy())
end

fit_pysr_compat_sr(args...; kwargs...) =
    fit_minimal_sr2(args...; config=pysr_compat_config(; kwargs...))

end # module MinimalSR2
