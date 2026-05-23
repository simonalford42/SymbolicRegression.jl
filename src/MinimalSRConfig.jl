# Default and PySR-compatible policy configurations for MinimalSR.jl.

# ─── Shared policy helpers ───────────────────────────────────────────────────

function minimal_random_subtree(engine::RegularizedEvolutionEngine)
    rand(engine.rng) < 0.5 && return random_terminal(engine)
    arity = sample_operator_arity(engine; max_added_nodes=2)
    arity <= 0 && return random_terminal(engine)
    op = sample_operator(engine, arity)
    arity == 1 && return OpNode(op, random_terminal(engine), nothing)
    return OpNode(op, random_terminal(engine), random_terminal(engine))
end

function minimal_replace_subtree_mutation(engine::RegularizedEvolutionEngine, tree::Node)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    isempty(nodes) && return tree
    _, parent, side = rand(engine.rng, nodes)
    subtree = minimal_random_subtree(engine)
    return replace_subtree(tree, parent, side, subtree)
end

function default_replace_subtree_mutation(engine::RegularizedEvolutionEngine, parent::Individual)
    for _attempt in 1:10
        proposal = minimal_replace_subtree_mutation(engine, parent.tree)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function simple_tournament_select(population::Vector{Individual}, cfg::EngineConfig, rng)
    n = length(population)
    k = min(max(1, cfg.tournament_selection_n), n)
    candidate_idx = sample_indices(rng, n, k)
    costs = [population[i].cost for i in candidate_idx]
    return candidate_idx[argmin(costs)]
end

function topk_survive!(population::Vector{Individual}, offspring::Vector{Individual}, cfg::EngineConfig)
    isempty(offspring) && return nothing
    combined = vcat(copy.(population), copy.(offspring))
    sort!(combined, by=m -> (m.cost, m.loss, m.complexity, m.birth))
    keep = min(length(population), cfg.population_size, length(combined))
    empty!(population)
    append!(population, combined[1:keep])
    return nothing
end

mse_loss_function(tree::Node, state::EngineState, _config::MinimalSRConfig) =
    MS.evaluate_candidate(state.engine, tree)

always_accept(
    _parent::Individual, _child::Individual, _state::EngineState, _config::MinimalSRConfig
) = true

function subtree_swap_crossover(parent_a::Individual, parent_b::Individual, state::EngineState,
                                _config::MinimalSRConfig)
    for _attempt in 1:10
        t1, t2 = MS.default_crossover(state.engine, parent_a.tree, parent_b.tree)
        MS.valid_tree(state.engine, t1) && MS.valid_tree(state.engine, t2) && return (t1, t2)
    end
    return nothing
end

# ─── Default policy ──────────────────────────────────────────────────────────

mutable struct BasicPolicyState <: AbstractPolicyState
    archive::Vector{Individual}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
end

function BasicPolicyState(cfg::MS.EngineConfig)
    return BasicPolicyState(Individual[], false, zeros(Int, max(1, cfg.populations)))
end

result_members(policy_state::BasicPolicyState) = policy_state.archive

function default_minimal_kwargs(; kwargs...)
    defaults = (
        binary_operators=String["+", "-", "*", "/"],
        unary_operators=String[],
        constants=Float64[],
        constraints=Dict{String, Any}(),
        nested_constraints=Dict{String, Any}(),
        population_size=100,
        populations=1,
        niterations=100,
        ncycles_per_iteration=1,
        maxsize=30,
        maxdepth=10,
        parsimony=0.0,
        tournament_selection_n=5,
        tournament_selection_p=1.0,
        crossover_probability=0.1,
        skip_mutation_failures=true,
        use_frequency=false,
        use_frequency_in_tournament=false,
        adaptive_parsimony_scaling=0.0,
        annealing=false,
        alpha=1.0,
        perturbation_factor=0.1,
        probability_negate_constant=0.0,
        migration=false,
        hof_migration=false,
        fraction_replaced=0.0,
        fraction_replaced_hof=0.0,
        topn=10,
        should_optimize_constants=false,
        optimize_probability=0.0,
        optimizer_iterations=8,
        optimizer_nrestarts=1,
        should_simplify=false,
    )
    return merge(defaults, (; kwargs...))
end

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

init_basic_state(config::MinimalSRConfig) = BasicPolicyState(config.engine_config)

function basic_update_state!(populations::Vector{Population}, state::EngineState{BasicPolicyState},
                             config::MinimalSRConfig)
    policy_state = state.policy_state
    pop_indices = if !policy_state.archive_initialized
        collect(eachindex(populations))
    else
        [
            i for i in eachindex(populations) if
            state.completed_population_cycles[i] > policy_state.archive_counted_population_cycles[i]
        ]
    end
    isempty(pop_indices) && return nothing

    combined = copy(policy_state.archive)
    for i in pop_indices
        append!(combined, copy.(populations[i]))
    end
    sort!(combined, by=m -> (m.loss, m.cost, m.complexity, m.birth))
    empty!(policy_state.archive)
    seen = Set{String}()
    for member in combined
        isfinite(member.loss) || continue
        key = MS.node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(policy_state.archive, copy(member))
        length(policy_state.archive) >= max(1, config.engine_config.topn) && break
    end
    for i in pop_indices
        policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
    end
    policy_state.archive_initialized = true
    return nothing
end

basic_update_population(_policy_state::BasicPolicyState, populations::Vector{Population},
                        _state::EngineState, _config::MinimalSRConfig) = populations

function basic_policy()
    return MinimalSRPolicy(;
        init_state=init_basic_state,
        loss_function=mse_loss_function,
        survival=basic_survival,
        selection=basic_selection,
        mutation=basic_mutation,
        acceptance=always_accept,
        crossover=subtree_swap_crossover,
        update_population=basic_update_population,
        update_state! = basic_update_state!,
    )
end

function default_minimal_config(; kwargs...)
    nt = default_minimal_kwargs(; kwargs...)
    cfg = engine_config_from_namedtuple(nt)
    return MinimalSRConfig(; engine_config=cfg, policy=basic_policy())
end

fit_default_sr(args...; kwargs...) =
    fit_minimal_sr(args...; config=default_minimal_config(; kwargs...))

# ─── PySR-compatible policy ──────────────────────────────────────────────────

mutable struct PySRPolicyState <: AbstractPolicyState
    best_by_complexity::Dict{Int, Individual}
    frontier::Vector{Individual}
    per_population_stats::Vector{MS.RunningSearchStatistics}
    current_temperature::Float64
    counted_population_cycles::Vector{Int}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
    last_logged_cycle::Int
end

function PySRPolicyState(cfg::MS.EngineConfig)
    n = max(1, cfg.populations)
    return PySRPolicyState(
        Dict{Int, Individual}(),
        Individual[],
        [MS.RunningSearchStatistics(cfg.maxsize) for _ in 1:n],
        1.0,
        zeros(Int, n),
        false,
        zeros(Int, n),
        -2,
    )
end

current_stats(state::EngineState{PySRPolicyState}) =
    state.policy_state.per_population_stats[state.current_population]

result_members(policy_state::PySRPolicyState) = policy_state.frontier

const PYSR_COMPAT_MUTATION_NAMES = [
    :add_node,
    :insert_node,
    :delete_node,
    :do_nothing,
    :mutate_constant,
    :mutate_operator,
    :mutate_feature,
    :swap_operands,
    :rotate_tree,
    :randomize,
    :simplify,
    :optimize,
]

const PYSR_COMPAT_MUTATION_WEIGHTS = Dict{Symbol, Float64}(
    :add_node => 2.47,
    :insert_node => 0.0112,
    :delete_node => 0.87,
    :do_nothing => 0.273,
    :mutate_constant => 0.0346,
    :mutate_operator => 0.293,
    :mutate_feature => 0.1,
    :swap_operands => 0.198,
    :rotate_tree => 4.26,
    :randomize => 0.000502,
    :simplify => 0.00209,
    :optimize => 0.0,
)

function conditioned_pysr_mutation_weights(engine::RegularizedEvolutionEngine, tree::Node)
    nodes = nodes_with_parent(tree)
    leaves = leaf_nodes(tree)
    n_constants = count(n -> n isa ConstNode, leaves)
    names = copy(PYSR_COMPAT_MUTATION_NAMES)
    w = Dict(name => max(0.0, PYSR_COMPAT_MUTATION_WEIGHTS[name]) for name in names)
    if isleaf(tree)
        w[:mutate_operator] = 0.0
        w[:swap_operands] = 0.0
        w[:delete_node] = 0.0
        w[:simplify] = 0.0
        if tree isa VarNode
            w[:optimize] = 0.0
            w[:mutate_constant] = 0.0
        else
            w[:mutate_feature] = 0.0
        end
    end
    if !any(n isa OpNode && !isnothing(n.right) for (n, _, _) in nodes)
        w[:swap_operands] = 0.0
    end
    w[:mutate_constant] *= min(8, n_constants) / 8.0
    engine.n_features <= 1 && (w[:mutate_feature] = 0.0)
    tree_size(tree) >= engine.cfg.maxsize && ((w[:add_node] = 0.0); (w[:insert_node] = 0.0))
    !engine.cfg.should_simplify && (w[:simplify] = 0.0)
    return names, w
end

function sample_pysr_mutation_choice(engine::RegularizedEvolutionEngine, tree::Node)
    names, w = conditioned_pysr_mutation_weights(engine, tree)
    weights = [w[n] for n in names]
    total = sum(weights)
    total <= 0 && return :do_nothing
    weights ./= total
    return weighted_choice(engine.rng, names, weights)
end

function pysr_weighted_mutation(engine::RegularizedEvolutionEngine, parent::Individual)
    mutation = sample_pysr_mutation_choice(engine, parent.tree)
    for _attempt in 1:10
        proposal = default_mutation(engine, parent.tree, mutation)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function pysr_compat_kwargs(; kwargs...)
    defaults = (
        binary_operators=String["+", "-", "*", "/"],
        unary_operators=String["sin", "cos", "exp", "log", "sqrt", "square"],
        constants=Float64[],
        constraints=Dict{String, Any}(),
        nested_constraints=Dict{String, Any}(),
        population_size=27,
        populations=31,
        niterations=100,
        ncycles_per_iteration=380,
        maxsize=30,
        maxdepth=16,
        parsimony=0.0,
        tournament_selection_n=15,
        tournament_selection_p=0.982,
        crossover_probability=0.0259,
        skip_mutation_failures=true,
        use_frequency=true,
        use_frequency_in_tournament=true,
        adaptive_parsimony_scaling=1040.0,
        annealing=false,
        alpha=3.17,
        perturbation_factor=0.129,
        probability_negate_constant=0.00743,
        migration=true,
        hof_migration=true,
        fraction_replaced=0.00036,
        fraction_replaced_hof=0.0614,
        topn=12,
        should_optimize_constants=true,
        optimize_probability=0.14,
        optimizer_iterations=8,
        optimizer_nrestarts=2,
        optimizer_f_calls_limit=10_000,
        should_simplify=true,
    )
    return merge(defaults, (; kwargs...))
end

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
        state.engine, parent, child, current_stats(state), state.policy_state.current_temperature
    )
end

init_pysr_state(config::MinimalSRConfig) = PySRPolicyState(config.engine_config)

function pysr_update_state!(populations::Vector{Population}, state::EngineState{PySRPolicyState},
                            config::MinimalSRConfig)
    policy_state = state.policy_state
    was_archive_initialized = policy_state.archive_initialized
    archive_pop_indices = if !policy_state.archive_initialized
        collect(eachindex(populations))
    else
        [
            i for i in eachindex(populations) if
            state.completed_population_cycles[i] > policy_state.archive_counted_population_cycles[i]
        ]
    end
    if !isempty(archive_pop_indices)
        for i in archive_pop_indices, member in populations[i]
            c = member.complexity
            1 <= c <= config.engine_config.maxsize || continue
            best = get(policy_state.best_by_complexity, c, nothing)
            if isnothing(best) || member.loss < best.loss
                policy_state.best_by_complexity[c] = copy(member)
            end
        end
        policy_state.frontier = MS.calculate_pareto_frontier_from_dict(
            policy_state.best_by_complexity
        )
        for i in archive_pop_indices
            policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
        end
        policy_state.archive_initialized = true
        if !isempty(config.engine_config.log_file)
            cycle = was_archive_initialized ?
                (state.current_iteration - 1) * length(populations) + (state.current_population - 1) :
                -1
            if cycle > policy_state.last_logged_cycle
                MS.write_hof_log(state.engine, cycle, policy_state.frontier)
                policy_state.last_logged_cycle = cycle
            end
        end
    end

    cfg = config.engine_config
    stats = current_stats(state)

    MS.normalize!(stats)
    if cfg.annealing && cfg.ncycles_per_iteration > 1
        denom = max(1, cfg.ncycles_per_iteration - 1)
        policy_state.current_temperature = 1.0 - (state.current_inner_cycle - 1) / denom
    else
        policy_state.current_temperature = 1.0
    end
    state.engine.current_temperature = clamp(policy_state.current_temperature, 0.0, 1.0)

    pop_idx = state.current_population
    completed = state.completed_population_cycles[pop_idx]
    if completed > policy_state.counted_population_cycles[pop_idx]
        pop = populations[state.current_population]
        for member in pop
            MS.update_size!(stats, member.complexity)
        end
        MS.move_window!(stats)
        policy_state.counted_population_cycles[pop_idx] = completed
    end
    return nothing
end

function pysr_update_population(policy_state::PySRPolicyState, populations::Vector{Population},
                                state::EngineState, _config::MinimalSRConfig)
    MS.default_migration(state.engine, populations, state.current_population, policy_state.frontier)
    return populations
end

function pysr_policy()
    return MinimalSRPolicy(;
        init_state=init_pysr_state,
        loss_function=mse_loss_function,
        survival=pysr_survival,
        selection=pysr_selection,
        mutation=pysr_mutation,
        acceptance=pysr_acceptance,
        crossover=subtree_swap_crossover,
        update_population=pysr_update_population,
        update_state! = pysr_update_state!,
    )
end

function pysr_compat_config(; kwargs...)
    nt = pysr_compat_kwargs(; kwargs...)
    cfg = engine_config_from_namedtuple(nt)
    return MinimalSRConfig(; engine_config=cfg, policy=pysr_policy())
end

fit_pysr_compat_sr(args...; kwargs...) =
    fit_minimal_sr(args...; config=pysr_compat_config(; kwargs...))
