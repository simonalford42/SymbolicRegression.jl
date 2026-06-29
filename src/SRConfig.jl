# SR policy configuration for SkeletonSR.jl.
#
# This module is a copy of BasicSRConfig.jl that serves as the canvas for
# meta-evolution via evolve_fullsr.py: the LLM proposes new versions of the
# eight policy functions (loss_function, survival, selection, mutation,
# acceptance, crossover, update_population, update_state!) and the runtime
# loads them dynamically. BasicSRConfig.jl is preserved unchanged so the
# evolution loop always has a known-good seed to fall back to.

module SRConfig

using Random: rand, randperm

using ..SkeletonSR:
    AbstractPolicyState,
    ConstNode,
    EngineConfig,
    EngineState,
    EvolutionEngine,
    Individual,
    Node,
    OpNode,
    Population,
    SkeletonSRConfig,
    SkeletonSRPolicy,
    VarNode,
    evaluate_tree,
    fit_skeleton_sr,
    node_string,
    nodes_with_parent,
    random_terminal,
    replace_subtree,
    sample_operator,
    sample_operator_arity,
    skeleton_sr_config,
    valid_tree,
    weighted_choice

# ─── SR policy ──────────────────────────────────────────────────────────────

mutable struct SRState <: AbstractPolicyState
    archive::Vector{Individual}
    archive_initialized::Bool
    archive_counted_population_cycles::Vector{Int}
end

function SRState(cfg::EngineConfig)
    return SRState(Individual[], false, zeros(Int, max(1, cfg.populations)))
end

function sr_loss_function(tree::Node, complexity::Int, state::EngineState, _config::SkeletonSRConfig)
    engine = state.engine
    pred = Vector{Float64}(evaluate_tree(tree, engine.X))
    if length(pred) != length(engine.y) || !all(isfinite.(pred) .& (abs.(pred) .< 1e12))
        return (Inf, Inf)
    end
    loss = sum((engine.y .- pred) .^ 2) / length(engine.y)
    cost = loss
    isfinite(cost) || return (Inf, Inf)
    return (loss, cost)
end

function sr_survival(
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

function sr_selection(population::Population, state::EngineState, _config::SkeletonSRConfig)
    n = length(population)
    k = min(15, n)
    candidate_idx = randperm(state.engine.rng, n)[1:k]
    costs = [population[i].cost for i in candidate_idx]
    return population[candidate_idx[argmin(costs)]]
end

function sr_mutation(parent::Individual, state::EngineState, _config::SkeletonSRConfig)
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

function sr_acceptance(
    _parent::Individual, _child::Individual, _state::EngineState, _config::SkeletonSRConfig
)
    return true
end

function sr_crossover(
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

function sr_update_population(
    _policy_state::SRState,
    populations::Vector{Population},
    _state::EngineState,
    _config::SkeletonSRConfig,
)
    return populations
end

function sr_update_archive!(
    populations::Vector{Population}, state::EngineState{SRState}, _config::SkeletonSRConfig
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
        length(policy_state.archive) >= 10 && break
    end
    for i in pop_indices
        policy_state.archive_counted_population_cycles[i] = state.completed_population_cycles[i]
    end
    policy_state.archive_initialized = true
    return nothing
end

function sr_policy()
    return SkeletonSRPolicy(;
        init_state=config -> SRState(config.engine_config),
        loss_function=sr_loss_function,
        survival=sr_survival,
        selection=sr_selection,
        mutation=sr_mutation,
        acceptance=sr_acceptance,
        crossover=sr_crossover,
        update_population=sr_update_population,
        update_state! = sr_update_archive!,
    )
end

fit_sr(args...; kwargs...) =
    fit_skeleton_sr(args...; config=skeleton_sr_config(sr_policy(); kwargs...))

end
