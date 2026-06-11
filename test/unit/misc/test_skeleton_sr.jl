@testitem "SkeletonSR policies run and PySR policy honors eval budget" begin
    using SymbolicRegression

    Xj = hcat(
        collect(range(-1.0, 1.0; length=24)),
        collect(range(0.5, 2.5; length=24)),
    )
    yj = @. 1.5 * Xj[:, 1] - 0.75 * Xj[:, 2] + 0.2
    variable_names = ["x0", "x1"]

    basic_result = SymbolicRegression.BasicSRConfig.fit_basic_sr(
        Xj,
        yj,
        variable_names;
        binary_operators=["+", "-", "*", "/"],
        unary_operators=String[],
        population_size=8,
        populations=1,
        niterations=4,
        ncycles_per_iteration=2,
        maxsize=10,
        maxdepth=5,
        max_evals=120,
        tournament_selection_n=3,
        topn=5,
        random_state=11,
    )

    @test basic_result["n_evals"] <= 120
    @test !isempty(basic_result["rows"])
    @test length(basic_result["rows"]) <= 10

    constraints = Dict{String, Any}(
        "/" => [-1, 6],
        "sin" => 6,
        "cos" => 6,
    )
    nested_constraints = Dict{String, Any}(
        "sin" => Dict("sin" => 0, "cos" => 0),
        "cos" => Dict("sin" => 0, "cos" => 0),
    )
    pysr_kwargs = (
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        constants=Float64[],
        constraints=constraints,
        nested_constraints=nested_constraints,
        population_size=9,
        populations=2,
        niterations=8,
        ncycles_per_iteration=3,
        maxsize=12,
        maxdepth=6,
        max_evals=240,
        parsimony=0.0,
        tournament_selection_n=15,
        tournament_selection_p=0.982,
        crossover_probability=0.0259,
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
        random_state=17,
    )
    skeleton_result = SymbolicRegression.PySRConfig.fit_pysr_sr(
        Xj, yj, variable_names; pysr_kwargs...
    )

    @test skeleton_result["n_evals"] <= pysr_kwargs.max_evals
    @test !isempty(skeleton_result["rows"])
    @test issorted([row["complexity"] for row in skeleton_result["rows"]])
    @test all(row -> row["complexity"] >= 1 && isfinite(row["loss"]), skeleton_result["rows"])
end

@testitem "SkeletonSR applies acceptance to crossover children" begin
    using SymbolicRegression

    const S = SymbolicRegression.SkeletonSR

    mutable struct RejectCrossoverState <: S.AbstractPolicyState
        archive::Vector{S.Individual}
    end

    acceptance_calls = Ref(0)
    policy = S.SkeletonSRPolicy(;
        init_state=config -> RejectCrossoverState(S.Individual[]),
        loss_function=(tree, complexity, state, config) -> (
            Float64(complexity), Float64(complexity)
        ),
        survival=(population, candidates, state, config) -> error(
            "rejected crossover children should not reach survival"
        ),
        selection=(population, state, config) -> population[1],
        mutation=(parent, state, config) -> error("mutation should not run"),
        acceptance=(parent, child, state, config) -> begin
            acceptance_calls[] += 1
            false
        end,
        crossover=(parent_a, parent_b, state, config) -> (S.VarNode(1), S.VarNode(1)),
        cycles_per_population=(population, state, config) -> 1,
        should_crossover=(population, state, config) -> true,
        update_population=(policy_state, populations, state, config) -> populations,
        update_state! = (populations, state, config) -> nothing,
    )

    result = S.fit_skeleton_sr(
        reshape(collect(range(-1.0, 1.0; length=8)), :, 1),
        collect(range(-1.0, 1.0; length=8)),
        ["x0"];
        config=S.skeleton_sr_config(
            policy;
            binary_operators=["+"],
            unary_operators=String[],
            population_size=2,
            populations=1,
            niterations=1,
            ncycles_per_iteration=1,
            maxsize=8,
            maxdepth=4,
            max_evals=8,
            random_state=3,
        ),
    )

    @test acceptance_calls[] == 2
    @test !isempty(result["rows"])
end
