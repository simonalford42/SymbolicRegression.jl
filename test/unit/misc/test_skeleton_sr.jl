@testitem "SkeletonSR policies run and PySR policy matches MiniSR" begin
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
    @test length(basic_result["rows"]) <= 5

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
        tournament_selection_n=3,
        tournament_selection_p=0.982,
        crossover_probability=0.2,
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
        topn=4,
        should_optimize_constants=false,
        optimize_probability=0.0,
        optimizer_iterations=2,
        optimizer_nrestarts=1,
        optimizer_f_calls_limit=100,
        should_simplify=true,
        random_state=17,
        log_file="",
    )

    minisr_kwargs = merge(
        pysr_kwargs,
        (
            mutation_weights=copy(SymbolicRegression.PySRConfig.PYSR_MUTATION_WEIGHTS),
            mutation_weight_names=SymbolicRegression.PySRConfig.PYSR_MUTATION_NAMES,
        ),
    )
    minisr_result = SymbolicRegression.MiniSR.fit_mini_sr(
        Xj, yj, variable_names; minisr_kwargs...
    )
    skeleton_result = SymbolicRegression.PySRConfig.fit_pysr_sr(
        Xj, yj, variable_names; pysr_kwargs...
    )

    @test skeleton_result["n_evals"] == minisr_result["n_evals"]
    @test skeleton_result["rows"] == minisr_result["rows"]
end
