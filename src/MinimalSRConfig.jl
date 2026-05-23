# Default and PySR-compatible policy configurations for MinimalSR.jl.

# ─── Running search statistics ──────────────────────────────────────────────

mutable struct RunningSearchStatistics
    frequencies::Vector{Float64}
    normalized_frequencies::Vector{Float64}
    window_size::Int
end

function RunningSearchStatistics(maxsize::Int; window_size::Int=100_000)
    freqs = ones(Float64, maxsize)
    RunningSearchStatistics(freqs, freqs ./ sum(freqs), window_size)
end

function update_size!(stats::RunningSearchStatistics, size::Int)
    if 1 <= size <= length(stats.frequencies)
        stats.frequencies[size] += 1.0
    end
end

function move_window!(stats::RunningSearchStatistics)
    smallest_frequency_allowed = 1.0
    max_loops = 1000
    total = sum(stats.frequencies)
    total <= stats.window_size && return
    difference = total - stats.window_size
    num_loops = 0
    while difference > 0
        indices = findall(>(smallest_frequency_allowed), stats.frequencies)
        isempty(indices) && break
        num_remaining = length(indices)
        max_subtract = minimum(stats.frequencies[indices]) - smallest_frequency_allowed
        amount = min(difference / num_remaining, max_subtract)
        stats.frequencies[indices] .-= amount
        total_subtracted = amount * num_remaining
        difference -= total_subtracted
        num_loops += 1
        if num_loops > max_loops || total_subtracted < 1e-6
            break
        end
    end
end

function normalize!(stats::RunningSearchStatistics)
    total = sum(stats.frequencies)
    if total <= 0
        stats.normalized_frequencies .= 1.0 / length(stats.frequencies)
    else
        stats.normalized_frequencies .= stats.frequencies ./ total
    end
end

# ─── PySR-compatible low-level operators ────────────────────────────────────

sample_indices(rng, n::Int, k::Int) = randperm(rng, n)[1:min(k, n)]

# Knuth's algorithm, matches SymbolicRegression.jl/src/Utils.jl:poisson_sample.
function poisson_sample(rng::AbstractRNG, λ::Float64)
    iszero(λ) && return 0
    k, p, L = 0, 1.0, exp(-λ)
    while p > L
        k += 1
        p *= rand(rng)
    end
    return k - 1
end

function tournament_select(population::Vector{Individual}, stats::RunningSearchStatistics,
                           cfg::EngineConfig, rng)
    n = length(population)
    k = min(cfg.tournament_selection_n, n)
    candidate_idx = sample_indices(rng, n, k)
    adjusted_costs = Float64[]
    for idx in candidate_idx
        member = population[idx]
        cost = member.cost
        if cfg.use_frequency_in_tournament && 1 <= member.complexity <= cfg.maxsize
            freq = stats.normalized_frequencies[member.complexity]
            cost *= exp(clamp(cfg.adaptive_parsimony_scaling * freq, -50.0, 50.0))
        end
        push!(adjusted_costs, cost)
    end
    order = sortperm(adjusted_costs)
    p = cfg.tournament_selection_p
    p >= 1.0 && return candidate_idx[order[1]]
    weights = [p * ((1 - p) ^ i) for i in 0:(k - 1)]
    weights ./= sum(weights)
    place = weighted_choice(rng, 1:k, weights)
    return candidate_idx[order[place]]
end

function oldest_survival(population::Vector{Individual}, rng, exclude_indices::Set{Int})
    candidates = [i for i in eachindex(population) if !(i in exclude_indices)]
    isempty(candidates) && return rand(rng, eachindex(population))
    births = [population[i].birth for i in candidates]
    return candidates[argmin(births)]
end

function default_mutation(engine::RegularizedEvolutionEngine, tree::Node, mutation::Symbol)
    tree = copy(tree)
    nodes = nodes_with_parent(tree)
    leaves = leaf_nodes(tree)
    constants = [n for n in leaves if n isa ConstNode]
    mutation === :do_nothing && return tree

    if mutation === :mutate_constant
        if !isempty(constants)
            node = rand(engine.rng, constants)
            temperature = clamp(engine.current_temperature, 0.0, 1.0)
            bottom = 0.1
            max_change = engine.cfg.perturbation_factor * temperature + 1.0 + bottom
            factor = max_change ^ rand(engine.rng)
            rand(engine.rng) < 0.5 && (factor = 1.0 / factor)
            rand(engine.rng) > engine.cfg.probability_negate_constant && (factor *= -1.0)
            node.value = clamp(node.value * factor, -1e6, 1e6)
        end
        return tree

    elseif mutation === :mutate_feature
        vars_only = [n for n in leaves if n isa VarNode]
        if !isempty(vars_only)
            node = rand(engine.rng, vars_only)
            if engine.n_features > 1
                choices = [i for i in 1:engine.n_features if i != node.feature]
                node.feature = rand(engine.rng, choices)
            else
                node.feature = rand(engine.rng, 1:engine.n_features)
            end
        end
        return tree

    elseif mutation === :mutate_operator
        op_nodes = [n for (n, _, _) in nodes if n isa OpNode]
        if !isempty(op_nodes)
            node = rand(engine.rng, op_nodes)
            if isnothing(node.right) && !isempty(engine.unary_ops)
                node.op = rand(engine.rng, engine.unary_ops)
            elseif !isnothing(node.right) && !isempty(engine.binary_ops)
                node.op = rand(engine.rng, engine.binary_ops)
            end
        end
        return tree

    elseif mutation === :swap_operands
        binary_nodes = [n for (n, _, _) in nodes if n isa OpNode && !isnothing(n.right)]
        if !isempty(binary_nodes)
            node = rand(engine.rng, binary_nodes)
            node.left, node.right = node.right, node.left
        end
        return tree

    elseif mutation === :delete_node
        deletable = [(n, p, s) for (n, p, s) in nodes if n isa OpNode]
        if !isempty(deletable)
            node, parent, side = rand(engine.rng, deletable)
            repl = if isnothing(node.right)
                copy(node.left)
            else
                rand(engine.rng) < 0.5 ? copy(node.left) : copy(node.right)
            end
            tree = replace_subtree(tree, parent, side, repl)
        end
        return tree

    elseif mutation === :rotate_tree
        valid = Tuple{OpNode, Union{OpNode, Nothing}, Union{Symbol, Nothing}, Vector{Symbol}}[]
        for (node, parent, side) in nodes
            node isa OpNode || continue
            pivot_sides = Symbol[]
            node.left isa OpNode && push!(pivot_sides, :left)
            !isnothing(node.right) && node.right isa OpNode && push!(pivot_sides, :right)
            !isempty(pivot_sides) && push!(valid, (node, parent, side, pivot_sides))
        end
        if !isempty(valid)
            node, parent, side, pivot_sides = rand(engine.rng, valid)
            pivot_side = rand(engine.rng, pivot_sides)
            pivot = pivot_side === :left ? node.left : node.right
            pivot isa OpNode || return tree
            grand_sides = Symbol[:left]
            !isnothing(pivot.right) && push!(grand_sides, :right)
            grand_side = rand(engine.rng, grand_sides)
            grand_child = grand_side === :left ? pivot.left : pivot.right
            pivot_side === :left ? (node.left = grand_child) : (node.right = grand_child)
            grand_side === :left ? (pivot.left = node) : (pivot.right = node)
            tree = replace_subtree(tree, parent, side, pivot)
        end
        return tree

    elseif mutation === :add_node
        return rand(engine.rng) < 0.5 ? append_random_op(engine, tree) : prepend_random_op(engine, tree)

    elseif mutation === :insert_node
        return insert_random_op(engine, tree)

    elseif mutation === :simplify || mutation === :optimize
        return tree

    elseif mutation === :randomize
        target_size = rand(engine.rng, 1:engine.cfg.maxsize)
        return random_tree_fixed_size(engine, target_size)
    end
    return tree
end

function default_crossover(engine::RegularizedEvolutionEngine, parent1::Node, parent2::Node)
    t1 = copy(parent1)
    t2 = copy(parent2)
    n1 = nodes_with_parent(t1)
    n2 = nodes_with_parent(t2)
    node1, p1, s1 = rand(engine.rng, n1)
    node2, p2, s2 = rand(engine.rng, n2)
    t1 = replace_subtree(t1, p1, s1, copy(node2))
    t2 = replace_subtree(t2, p2, s2, copy(node1))
    return t1, t2
end

function default_migration(engine::RegularizedEvolutionEngine, populations::Vector{Vector{Individual}},
                           pop_idx::Int, dominating::Vector{Individual})
    cfg = engine.cfg
    target = populations[pop_idx]
    isempty(target) && return
    function replace_from!(candidates::Vector{Individual}, frac::Float64)
        (isempty(candidates) || frac <= 0) && return
        n = poisson_sample(engine.rng, max(0.0, length(target) * frac))
        n <= 0 && return
        n = min(n, length(target))
        for _ in 1:n
            dst = rand(engine.rng, 1:length(target))
            src = copy(rand(engine.rng, candidates))
            src.birth = next_birth!(engine)
            src.ref = next_ref!(engine)
            target[dst] = src
        end
    end
    if cfg.migration
        best_of_each = Individual[]
        for pop in populations
            isempty(pop) && continue
            topk = sort(pop, by=m -> m.cost)[1:max(1, min(cfg.topn, length(pop)))]
            append!(best_of_each, copy.(topk))
        end
        replace_from!(best_of_each, cfg.fraction_replaced)
    end
    cfg.hof_migration && replace_from!(dominating, cfg.fraction_replaced_hof)
end

function accept_candidate(engine::RegularizedEvolutionEngine, parent::Individual, child::Individual,
                          stats::RunningSearchStatistics, temperature::Float64)
    isfinite(child.cost) || return false
    prob = 1.0
    if engine.cfg.annealing
        delta = child.cost - parent.cost
        denom = max(1e-8, temperature * engine.cfg.alpha)
        prob *= exp(clamp(-delta / denom, -50.0, 50.0))
    end
    if engine.cfg.use_frequency
        old_size = min(max(parent.complexity, 1), engine.cfg.maxsize)
        new_size = min(max(child.complexity, 1), engine.cfg.maxsize)
        old_f = stats.normalized_frequencies[old_size]
        new_f = stats.normalized_frequencies[new_size]
        prob *= old_f / max(new_f, 1e-12)
    end
    prob = min(prob, 1e6)
    prob >= 1.0 && return true
    return rand(engine.rng) < prob
end

function calculate_pareto_frontier_from_dict(hof_by_complexity::Dict{Int, Individual})
    dominating = Individual[]
    best_so_far = Inf
    for c in sort(collect(keys(hof_by_complexity)))
        member = hof_by_complexity[c]
        if member.loss < best_so_far
            push!(dominating, copy(member))
            best_so_far = member.loss
        end
    end
    return dominating
end

function _json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"'
            write(buf, "\\\"")
        elseif c == '\\'
            write(buf, "\\\\")
        elseif c == '\n'
            write(buf, "\\n")
        elseif c == '\r'
            write(buf, "\\r")
        elseif c == '\t'
            write(buf, "\\t")
        elseif c < ' '
            write(buf, string("\\u", lpad(string(UInt16(c), base=16), 4, '0')))
        else
            write(buf, c)
        end
    end
    return String(take!(buf))
end

function _json_float(x::Float64)
    if isnan(x); return "NaN"; end
    if isinf(x); return x > 0 ? "Infinity" : "-Infinity"; end
    return string(x)
end

function write_hof_log(engine::RegularizedEvolutionEngine, cycle::Int,
                       dominating::Vector{Individual})
    path = engine.cfg.log_file
    isempty(path) && return
    budget = engine.eval_budget
    progress = (isnothing(budget) || budget == 0) ? 0.0 : engine.eval_count / budget
    try
        open(path, "a") do io
            write(io, "{\"cycle\":", string(cycle))
            write(io, ",\"eval_count\":", string(engine.eval_count))
            write(io, ",\"progress\":", _json_float(progress))
            write(io, ",\"frontier\":[")
            for (i, member) in enumerate(dominating)
                i > 1 && write(io, ",")
                eqn = node_string(member.tree)
                write(io, "{\"complexity\":", string(member.complexity))
                write(io, ",\"loss\":", _json_float(member.loss))
                write(io, ",\"equation\":\"", _json_escape(eqn), "\"}")
            end
            write(io, "]}\n")
        end
    catch err
        @warn "write_hof_log failed" path=path err=err
    end
end

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
