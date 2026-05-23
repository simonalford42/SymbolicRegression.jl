# BasicSR policy configuration for SkeletonSR.jl.

module BasicSRConfig

using Random: rand, randperm

using ..SkeletonSR:
    AbstractPolicyState,
    EngineConfig,
    EngineState,
    Individual,
    Node,
    OpNode,
    Population,
    SkeletonSRConfig,
    SkeletonSRPolicy,
    engine_config_from_namedtuple,
    evaluate_tree,
    fit_skeleton_sr,
    node_string,
    nodes_with_parent,
    random_terminal,
    replace_subtree,
    sample_operator,
    sample_operator_arity,
    valid_tree

Base.@kwdef struct BasicSROptions
    tournament_selection_n::Int = 5
    crossover_probability::Float64 = 0.1
    skip_mutation_failures::Bool = true
    topn::Int = 10
    parsimony::Float64 = 0.0
end

# ─── BasicSR policy ──────────────────────────────────────────────────────────

mutable struct BasicSRState <: AbstractPolicyState
    options::BasicSROptions
    archive::Vector{Individual}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
end

function BasicSRState(cfg::EngineConfig, options::BasicSROptions)
    return BasicSRState(options, Individual[], false, zeros(Int, max(1, cfg.populations)))
end

function basic_sr_kwargs(; kwargs...)
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
        crossover_probability=0.1,
        skip_mutation_failures=true,
        topn=10,
    )
    return merge(defaults, (; kwargs...))
end

function basic_loss_function(tree::Node, complexity::Int, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred) .& (abs.(pred) .< 1e12))
        return (Inf, Inf)
    end
    loss = sum((engine.y .- pred) .^ 2) / length(engine.y)
    cost = loss
    isfinite(cost) || return (Inf, Inf)
    return (loss, cost + state.policy_state.options.parsimony * complexity)
end

function basic_survival(
    population::Population,
    candidates::Vector{Individual},
    _state::EngineState,
    config::SkeletonSRConfig,
)
    output = copy(population)
    isempty(candidates) && return output
    combined = vcat(copy.(population), copy.(candidates))
    sort!(combined, by=m -> (m.cost, m.loss, m.complexity, m.birth))
    keep = min(length(population), config.engine_config.population_size, length(combined))
    empty!(output)
    append!(output, combined[1:keep])
    return output
end

function basic_selection(population::Population, state::EngineState, _config::SkeletonSRConfig)
    n = length(population)
    k = min(max(1, state.policy_state.options.tournament_selection_n), n)
    candidate_idx = randperm(state.engine.rng, n)[1:k]
    costs = [population[i].cost for i in candidate_idx]
    return population[candidate_idx[argmin(costs)]]
end

function basic_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    for _attempt in 1:10
        proposal = copy(parent.tree)
        nodes = nodes_with_parent(proposal)
        isempty(nodes) && return proposal
        _, parent_node, side = rand(engine.rng, nodes)

        subtree = if rand(engine.rng) < 0.5
            random_terminal(engine)
        else
            arity = sample_operator_arity(engine; max_added_nodes=2)
            if arity <= 0
                random_terminal(engine)
            else
                op = sample_operator(engine, arity)
                if arity == 1
                    OpNode(op, random_terminal(engine), nothing)
                else
                    OpNode(op, random_terminal(engine), random_terminal(engine))
                end
            end
        end

        proposal = replace_subtree(proposal, parent_node, side, subtree)
        valid_tree(engine, proposal) && return proposal
    end
    return nothing
end

function basic_acceptance(
    _parent::Individual, _child::Individual, _state::EngineState, _config::SkeletonSRConfig
)
    return true
end

function basic_crossover(
    parent_a::Individual,
    parent_b::Individual,
    state::EngineState,
    _config::SkeletonSRConfig,
)
    for _attempt in 1:10
        t1 = copy(parent_a.tree)
        t2 = copy(parent_b.tree)
        node1, p1, s1 = rand(state.engine.rng, nodes_with_parent(t1))
        node2, p2, s2 = rand(state.engine.rng, nodes_with_parent(t2))
        t1 = replace_subtree(t1, p1, s1, copy(node2))
        t2 = replace_subtree(t2, p2, s2, copy(node1))
        valid_tree(state.engine, t1) && valid_tree(state.engine, t2) && return (t1, t2)
    end
    return nothing
end

function basic_cycles_per_population(
    population::Population, state::EngineState, _config::SkeletonSRConfig
)
    return Int(ceil(length(population) / max(1, state.policy_state.options.tournament_selection_n)))
end

function basic_should_crossover(population::Population, state::EngineState, _config::SkeletonSRConfig)
    return length(population) >= 2 &&
        rand(state.engine.rng) <= state.policy_state.options.crossover_probability
end

function basic_skip_mutation_failures(state::EngineState, _config::SkeletonSRConfig)
    return state.policy_state.options.skip_mutation_failures
end

function basic_postprocess_population!(
    _population::Population, _state::EngineState, _config::SkeletonSRConfig
)
    return nothing
end

function basic_update_population(
    _policy_state::BasicSRState,
    populations::Vector{Population},
    _state::EngineState,
    _config::SkeletonSRConfig,
)
    return populations
end

function basic_update_archive!(
    populations::Vector{Population}, state::EngineState{BasicSRState}, _config::SkeletonSRConfig
)
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
        key = node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(policy_state.archive, copy(member))
        length(policy_state.archive) >= max(1, policy_state.options.topn) && break
    end
    for i in pop_indices
        policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
    end
    policy_state.archive_initialized = true
    return nothing
end

function basic_policy(options::BasicSROptions)
    return SkeletonSRPolicy(;
        init_state=config -> BasicSRState(config.engine_config, options),
        loss_function=basic_loss_function,
        survival=basic_survival,
        selection=basic_selection,
        mutation=basic_mutation,
        acceptance=basic_acceptance,
        crossover=basic_crossover,
        cycles_per_population=basic_cycles_per_population,
        should_crossover=basic_should_crossover,
        skip_mutation_failures=basic_skip_mutation_failures,
        postprocess_population! = basic_postprocess_population!,
        update_population=basic_update_population,
        update_state! = basic_update_archive!,
    )
end

function basic_sr_config(; kwargs...)
    nt = basic_sr_kwargs(; kwargs...)
    cfg = engine_config_from_namedtuple(nt)
    options = BasicSROptions(;
        tournament_selection_n=nt.tournament_selection_n,
        crossover_probability=nt.crossover_probability,
        skip_mutation_failures=nt.skip_mutation_failures,
        topn=nt.topn,
        parsimony=nt.parsimony,
    )
    return SkeletonSRConfig(; engine_config=cfg, policy=basic_policy(options))
end

fit_basic_sr(args...; kwargs...) =
    fit_skeleton_sr(args...; config=basic_sr_config(; kwargs...))

end
