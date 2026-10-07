@testitem "Integer expression DAG choices, definedness and decoded weights" default_imports=false begin
    using Test, TOML
    import CompositionalNetworks as CN
    function choose!(n,i,role,j)
        b=only(b for b in CN.view_weight_blocks(n) if b.node==i && b.role==role)
        n.weights[b.range].=false; n.weights[first(b.range)+j-1]=true
    end
    function operation!(n,i,op,operands)
        choose!(n,i,:operation,only(findall(==(op),n.nodes[i].operations)))
        for (j,k) in enumerate(operands); choose!(n,i,Symbol("operand",j),k); end
    end
    function penalty!(n)
        child=n.child; fill!(parent(child.weights),false); offset=0
        for (layer,name) in zip(child.layers,(:id,:sum,:sum,:condition_residual))
            j=only(findall(==(name),collect(keys(layer.fn))))
            parent(child.weights)[offset+j]=true; offset+=length(layer.fn)
        end
        n.weights[n.child_range].=child.weights
        @test CN.check_weights_validity(n,n.weights)
    end
    input=CN.ReshapedReference((CN.ViewReference(1),),(1,))
    build()=CN.learnable_views((;op=(==),val=CN.InputReference(3)),[1,1,1];input,
        views=(CN.ViewNode(2,(CN.InputReference(1),CN.InputReference(2))),))
    n=build(); penalty!(n)
    cases=((:add,+),(:sub,-),(:mul,*),(:div,div),(:rem,rem),(:min,min),(:max,max),
        (:eq,==),(:ne,!=),(:lt,<),(:le,<=),(:gt,>),(:ge,>=),
        (:and,(a,b)->Bool(a)&&Bool(b)),(:or,(a,b)->Bool(a)||Bool(b)),
        (:xor,(a,b)->xor(Bool(a),Bool(b))))
    for (op,fn) in cases
        operation!(n,1,op,(1,2))
        bits=TOML.parse(sprint(io->TOML.print(io,Dict("weights"=>collect(n.weights)))))["weights"]
        fresh=build(); @test CN.apply!(fresh,BitVector(bits)); decoded=CN.composition(fresh)
        for a in -2:2,b in -2:2,target in -3:3
            defined=!(op in (:div,:rem) && b==0) &&
                !(op in (:and,:or,:xor) && !(a in 0:1 && b in 0:1))
            expected=defined && fn(a,b)==target
            v=decoded([a,b,target])
            @test isfinite(v) && v>=0
            @test iszero(v)==expected
            @test v==CN.evaluate(fresh,CN.Solution([a,b,target]))
        end
    end
    operation!(n,1,:div,(1,2))
    @test CN.composition(n)([1,0,0])==1
    choose!(n,0,:definedness,2)
    @test CN.composition(n)([1,0,0])==0 # The guard is a real genotype choice.
    @test occursin("use_totalized_values",CN.code(n))
    @test !CN.incremental_supported(n)
    @test_throws ArgumentError CN.ViewNode(0,(1,))
    @test_throws ArgumentError CN.ViewNode(2,(identity,))
    @test_throws ArgumentError CN.learnable_views((;),[1];
        views=(CN.ViewNode(1,(CN.ViewReference(1),)),))
    @test_throws BoundsError CN.learnable_views((;),[1];
        input=CN.ReshapedReference((CN.ViewReference(1),),(1,)),
        views=(CN.ViewNode(1,(CN.InputReference(2),)),))
    @test_throws ArgumentError CN._integer_view_operation(:add,(1.0,2))
    @test CN._integer_view_operation(:neg,(typemin(Int),))==(big(typemax(Int))+1,true)
    @test CN._integer_view_operation(:pow,(0,0))==(1,true)
    @test CN._integer_view_operation(:pow,(-1,-3))==(-1,true)
    @test CN._integer_view_operation(:pow,(2,-1))==(0,false)

    # Enumerate operation weights: a solver sees the same selectable alternatives.
    unary=CN.learnable_views((;op=(==),val=1),[1];input,
        views=(CN.ViewNode(1,(CN.InputReference(1),)),))
    penalty!(unary)
    errors=Dict{Symbol,Int}()
    for op in unary.nodes[1].operations
        operation!(unary,1,op,(1,))
        errors[op]=count(-2:2) do x
            iszero(CN.composition(unary)([x])) != (abs(x)==1)
        end
    end
    @test errors[:abs]==0
    @test errors[:id]>0 && errors[:neg]>0 && errors[:not]>0
    @test CN.generate_new_valid_weights!(unary)
    @test CN.check_weights_validity(unary,unary.weights)
end

@testitem "LocalSearchSolvers consumes expression-view genotype" tags=[:extension] default_imports=false begin
    using Test, Random
    import CompositionalNetworks as CN
    import LocalSearchSolvers as LS
    n=CN.learnable_views((;op=(==),val=1),[1];
        input=CN.ReshapedReference((CN.ViewReference(1),),(1,)),
        views=(CN.ViewNode(1,(CN.InputReference(1),)),))
    configs=Set{CN.Configuration}([CN.Solution([1]),CN.Solution([-1]),CN.NonSolution([0])])
    seen=Int[]
    Random.seed!(0x71e7)
    optimizer=CN.LocalSearchOptimizer(;options=LS.Options(iteration=(false,2000),time_limit=30.0,
        print_level=:silent,log_to_file=false,use_progress_meter=false,process_threads_map=Dict(1=>1)))
    result=CN.optimize!(n,configs,CN.zero_set_loss,optimizer;
        candidate_observer=c->push!(seen,length(c.weights)))
    @test first(result)===n
    @test !isempty(seen)
    @test all(==(length(n.weights)),seen)
end
