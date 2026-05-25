@testitem "SkeletonSR policies run and PySR policy matches MiniSR" begin
    using SymbolicRegression
    using Optim

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
        random_state=17,
        log_file="",
    )
    mutation_weight_names = [
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
    mutation_weights = Dict{Symbol, Float64}(
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

    minisr_kwargs = merge(
        pysr_kwargs,
        (
            mutation_weights=mutation_weights,
            mutation_weight_names=mutation_weight_names,
        ),
    )
    minisr_result = SymbolicRegression.MiniSR.fit_mini_sr(
        Xj, yj, variable_names; minisr_kwargs...
    )
    skeleton_result = SymbolicRegression.PySRConfig.fit_pysr_sr(
        Xj, yj, variable_names; pysr_kwargs..., optimizer_algorithm=Optim.NelderMead()
    )

    @test skeleton_result["n_evals"] == minisr_result["n_evals"]
    @test length(skeleton_result["rows"]) == length(minisr_result["rows"])
    @test [row["complexity"] for row in skeleton_result["rows"]] ==
        [row["complexity"] for row in minisr_result["rows"]]
    @test all(zip(skeleton_result["rows"], minisr_result["rows"])) do (left, right)
        isapprox(left["loss"], right["loss"]; rtol=1e-2, atol=1e-8)
    end
end
