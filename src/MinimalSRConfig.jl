# Concrete MinimalSR configurations. This file is included inside module
# `MinimalSR`; the core data structures and primitive operators live in
# `MinimalSR.jl`.

# ─── Policy objects ─────────────────────────────────────────────────────────

"""Bare-bones genetic programming policy for MinimalSR."""
struct DefaultMinimalPolicy <: AbstractMinimalSRPolicy end

"""Compatibility policy that mirrors the original MiniSR.jl search loop."""
struct PySRCompatPolicy <: AbstractMinimalSRPolicy end

default_minimal_policy() = DefaultMinimalPolicy()
pysr_compat_policy() = PySRCompatPolicy()

resolve_policy(policy::Py) = resolve_policy(pyconvert(Any, policy))
resolve_policy(policy::AbstractString) = resolve_policy(Symbol(policy))

function resolve_policy(policy::Symbol)
    policy in (:default, :minimal, :barebones) && return default_minimal_policy()
    policy in (:pysr, :pysr_compat, :minisr, :compat) && return pysr_compat_policy()
    error("Unknown MinimalSR policy: $(policy)")
end

# ─── Default MinimalSR config ───────────────────────────────────────────────

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

function simple_tournament_select(population::Vector{Individual}, cfg::EngineConfig, rng)
    n = length(population)
    k = min(max(1, cfg.tournament_selection_n), n)
    candidate_idx = sample_indices(rng, n, k)
    costs = [population[i].cost for i in candidate_idx]
    return candidate_idx[argmin(costs)]
end

function topk_survive!(population::Vector{Individual}, offspring::Vector{Individual}, cfg::EngineConfig)
    isempty(offspring) && return
    combined = vcat(copy.(population), copy.(offspring))
    sort!(combined, by=m -> (m.cost, m.loss, m.complexity, m.birth))
    keep = min(length(population), cfg.population_size, length(combined))
    empty!(population)
    append!(population, combined[1:keep])
    return nothing
end

function update_topk_archive!(archive::Vector{Individual}, population::Vector{Individual}, cfg::EngineConfig)
    candidates = vcat(copy.(archive), copy.(population))
    sort!(candidates, by=m -> (m.loss, m.cost, m.complexity, m.birth))
    empty!(archive)
    seen = Set{String}()
    for member in candidates
        isfinite(member.loss) || continue
        key = node_string(member.tree)
        key in seen && continue
        push!(seen, key)
        push!(archive, copy(member))
        length(archive) >= max(1, cfg.topn) && break
    end
    return nothing
end

function default_minimal_generation!(engine::RegularizedEvolutionEngine, population::Vector{Individual})
    offspring = Individual[]
    target_offspring = max(1, length(population))
    for _ in 1:target_offspring
        has_budget(engine) || break
        if length(population) < 2 || rand(engine.rng) > engine.cfg.crossover_probability
            pidx = simple_tournament_select(population, engine.cfg, engine.rng)
            parent = population[pidx]
            child_tree = nothing
            for _attempt in 1:10
                proposal = minimal_replace_subtree_mutation(engine, parent.tree)
                if valid_tree(engine, proposal)
                    child_tree = proposal
                    break
                end
            end
            child_tree === nothing && continue
            child = create_individual(engine, child_tree; parent_ref=parent.ref)
            child === nothing && break
            push!(offspring, child)
        else
            p1 = simple_tournament_select(population, engine.cfg, engine.rng)
            p2 = simple_tournament_select(population, engine.cfg, engine.rng)
            if p1 == p2 && length(population) > 1
                p2 = (p2 % length(population)) + 1
            end
            pair = nothing
            for _attempt in 1:10
                t1, t2 = default_crossover(engine, population[p1].tree, population[p2].tree)
                if valid_tree(engine, t1) && valid_tree(engine, t2)
                    pair = (t1, t2)
                    break
                end
            end
            pair === nothing && continue
            t1, t2 = pair
            c1 = create_individual(engine, t1; parent_ref=population[p1].ref)
            c1 === nothing && break
            push!(offspring, c1)
            has_budget(engine) || break
            c2 = create_individual(engine, t2; parent_ref=population[p2].ref)
            c2 === nothing && break
            push!(offspring, c2)
        end
    end
    topk_survive!(population, offspring, engine.cfg)
    return nothing
end

function run_engine(engine::RegularizedEvolutionEngine, ::DefaultMinimalPolicy)
    populations = [initialize_population(engine) for _ in 1:max(1, engine.cfg.populations)]
    archive = Individual[]

    for pop in populations
        update_topk_archive!(archive, pop, engine.cfg)
    end
    isempty(engine.cfg.log_file) || write_hof_log(engine, -1, archive)

    total_cycles = max(0, Int(engine.cfg.niterations * length(populations)))
    for cycle in 0:(total_cycles - 1)
        has_budget(engine) || break
        j = (cycle % length(populations)) + 1
        pop = populations[j]
        for _ in 1:max(1, engine.cfg.ncycles_per_iteration)
            default_minimal_generation!(engine, pop)
            has_budget(engine) || break
        end
        update_topk_archive!(archive, pop, engine.cfg)
        isempty(engine.cfg.log_file) || write_hof_log(engine, cycle, archive)
    end

    if isempty(archive)
        best = populations[1][1]
        for pop in populations, member in pop
            member.cost < best.cost && (best = member)
        end
        archive = [copy(best)]
    end
    return archive, engine.eval_count
end

function default_minimal_config(; kwargs...)
    defaults = (
        policy=default_minimal_policy(),
        binary_operators=String["+", "-", "*", "/"],
        unary_operators=String[],
        constants=Float64[],
        mutation_weights=Dict{Symbol, Float64}(),
        mutation_weight_names=Symbol[],
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

function fit_default_sr(args...; kwargs...)
    return fit_minimal_sr(args...; default_minimal_config(; kwargs...)...)
end

# ─── PySR/MiniSR compatibility config ───────────────────────────────────────

const PYSR_COMPAT_MUTATION_WEIGHT_NAMES = [
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
    :custom_mutation_1,
    :custom_mutation_2,
    :custom_mutation_3,
    :custom_mutation_4,
    :custom_mutation_5,
    :custom_mutation_6,
]

function pysr_compat_mutation_weights()
    return Dict{Symbol, Float64}(
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
        :custom_mutation_1 => 0.0,
        :custom_mutation_2 => 0.0,
        :custom_mutation_3 => 0.0,
        :custom_mutation_4 => 0.0,
        :custom_mutation_5 => 0.0,
        :custom_mutation_6 => 0.0,
    )
end

function run_engine(engine::RegularizedEvolutionEngine, ::PySRCompatPolicy)
    populations = [initialize_population(engine) for _ in 1:max(1, engine.cfg.populations)]
    stats = [RunningSearchStatistics(engine.cfg.maxsize) for _ in populations]
    hof_by_complexity = Dict{Int, Individual}()

    function update_hof_from_population!(pop)
        for member in pop
            c = member.complexity
            1 <= c <= engine.cfg.maxsize || continue
            best = get(hof_by_complexity, c, nothing)
            if isnothing(best) || member.loss < best.loss
                hof_by_complexity[c] = copy(member)
            end
        end
    end

    for pop in populations
        update_hof_from_population!(pop)
    end
    if !isempty(engine.cfg.log_file)
        write_hof_log(engine, -1, calculate_pareto_frontier_from_dict(hof_by_complexity))
    end

    total_cycles = max(0, Int(engine.cfg.niterations * max(1, engine.cfg.populations)))
    for cycle in 0:(total_cycles - 1)
        has_budget(engine) || break
        j = (cycle % length(populations)) + 1
        pop = populations[j]
        s = stats[j]
        normalize!(s)
        temps = (engine.cfg.annealing && engine.cfg.ncycles_per_iteration > 1) ?
            collect(range(1.0, 0.0, length=engine.cfg.ncycles_per_iteration)) :
            fill(1.0, max(1, engine.cfg.ncycles_per_iteration))
        for temp in temps
            regularized_cycle!(engine, pop, s, temp)
            has_budget(engine) || break
        end
        has_budget(engine) && optimize_and_simplify!(engine, pop)
        for member in pop
            update_size!(s, member.complexity)
        end
        move_window!(s)
        update_hof_from_population!(pop)
        dominating = calculate_pareto_frontier_from_dict(hof_by_complexity)
        isempty(engine.cfg.log_file) || write_hof_log(engine, cycle, dominating)
        default_migration(engine, populations, j, dominating)
    end

    dominating = calculate_pareto_frontier_from_dict(hof_by_complexity)
    if isempty(dominating)
        best = populations[1][1]
        for pop in populations, member in pop
            member.cost < best.cost && (best = member)
        end
        dominating = [copy(best)]
    end
    return dominating, engine.eval_count
end

function pysr_compat_config(; kwargs...)
    defaults = (
        policy=pysr_compat_policy(),
        binary_operators=String["+", "-", "*", "/"],
        unary_operators=String["sin", "cos", "exp", "log", "sqrt", "square"],
        constants=Float64[],
        mutation_weights=pysr_compat_mutation_weights(),
        mutation_weight_names=PYSR_COMPAT_MUTATION_WEIGHT_NAMES,
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

function fit_pysr_compat_sr(args...; kwargs...)
    return fit_minimal_sr(args...; pysr_compat_config(; kwargs...)...)
end

run_engine(engine::RegularizedEvolutionEngine) = run_engine(engine, default_minimal_policy())
