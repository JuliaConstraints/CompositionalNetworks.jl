"""Reference a scalar integer expression node. Nodes may only reference earlier nodes."""
struct ViewReference
    node::Int
end

"""A scalar expression slot with learnable operation and operand edges.

`arity` and the types of `sources` determine the grammar, never a concept name.
`sqr`, `dist`, `iff` and other composite expressions are deliberately absent:
use multiplication, subtraction/absolute value, or equality of Boolean values.
"""
struct ViewNode
    arity::Int
    sources::Tuple
    operations::Tuple
end
function ViewNode(arity::Integer, sources::Tuple)
    1 <= arity <= 3 || throw(ArgumentError("scalar nodes have arity 1, 2 or 3; fold larger expressions"))
    isempty(sources) && throw(ArgumentError("a view needs operand sources"))
    all(s -> s isa Union{Integer,InputReference,ParameterReference,ViewReference}, sources) ||
        throw(ArgumentError("scalar sources must be literals or explicit references, not functions"))
    operations = arity == 1 ? (:id,:neg,:abs,:not) : arity == 2 ?
        (:add,:sub,:mul,:div,:rem,:pow,:min,:max,:eq,:ne,:lt,:le,:gt,:ge,:and,:or,:xor) : (:if,)
    ViewNode(Int(arity),sources,operations)
end

"""Integer expression DAG followed by a keyword-induced ICN, in one Boolean genotype.

Operations, operand edges, the definedness gate and the child weights are all
optimizer-visible. Input/keyword layout is modeling metadata (as in `learnable_binding`).
No expression evaluator, constraint identifier or concept callback is accepted.
Integer arithmetic uses BigInt to avoid overflow changing the zero set. This is
a feasibility path, not a compiled/performance-qualified kernel.
"""
struct ViewICN{C <: AbstractICN,I,P <: NamedTuple} <: AbstractICN
    nodes::Vector{ViewNode}
    child::C
    input::I
    bindings::P
    weights::BitVector
    layers::Vector{CompositionChoiceLayer}
    weightlen::Vector{Int}
    blocks::Vector{NamedTuple{(:node,:role,:range),Tuple{Int,Symbol,UnitRange{Int}}}}
    child_range::UnitRange{Int}
    constants::Dict{Symbol,Any}
    parameters::Set{Symbol}
end

function _check_view_reference(ref, count)
    if ref isa ViewReference
        1 <= ref.node <= count || throw(ArgumentError("view references must point to earlier nodes"))
    elseif ref isa ReshapedReference
        _check_view_reference(ref.reference,count)
    elseif ref isa Union{Tuple,NamedTuple}
        foreach(r->_check_view_reference(r,count),ref)
    elseif !(ref isa Union{InputReference,ParameterReference,Integer,Function,Symbol,Nothing,AbstractArray})
        throw(ArgumentError("unsupported view binding $(typeof(ref))"))
    end
end

"""Build scalar expression slots and a child ICN from resolved keyword types/shapes.

The signature sample must define the initial (identity/addition/if) choices.
An invalid sample is rejected, never silently replaced by a numerical penalty.
Select witness operations and edges by changing the returned Boolean weights.
"""
function learnable_views(bindings::NamedTuple, sample_input; views,
        input=InputReference(), runtime=(;), max_depth=2, nodes=0)
    slots=ViewNode[views...]
    for (i,slot) in enumerate(slots)
        foreach(r->_check_view_reference(r,i-1),slot.sources)
    end
    _check_view_reference(input,length(slots)); _check_view_reference(bindings,length(slots))
    weights=BitVector(); layers=CompositionChoiceLayer[]; lengths=Int[]
    blocks=NamedTuple{(:node,:role,:range),Tuple{Int,Symbol,UnitRange{Int}}}[]
    function add!(i,role,n; mutex=true,bits=BitVector(j==1 for j in 1:n))
        r=(length(weights)+1):(length(weights)+n)
        append!(weights,bits); push!(layers,CompositionChoiceLayer(role,mutex)); push!(lengths,n)
        push!(blocks,(;node=i,role,range=r))
    end
    for (i,slot) in enumerate(slots)
        add!(i,:operation,length(slot.operations))
        for j in 1:slot.arity
            add!(i,Symbol("operand",j),length(slot.sources))
        end
    end
    # Gate is itself a structural choice, not a hidden Boolean concept correction.
    add!(0,:definedness,2) # require_defined / use_totalized_values
    resolve=_view_resolver(slots,weights,blocks,sample_input,runtime)
    sx,okx=resolve(input); sp,okp=resolve(bindings)
    okx && okp || throw(ArgumentError("initial view signature sample is undefined"))
    sx isa AbstractVector || throw(ArgumentError("view input must resolve to a vector"))
    child=nodes>0 ? learnable_graph(sp;nodes) : learnable_composition(sp;max_depth)
    firstchild=length(weights)+1
    offset=1
    for (layer,n) in zip(child.layers,child.weightlen)
        add!(-1,:child,n;mutex=layer.mutex,bits=child.weights[offset:offset+n-1])
        offset+=n
    end
    params=Set{Symbol}()
    function visit(ref)
        if ref isa ParameterReference
            push!(params,ref.name)
        elseif ref isa ReshapedReference
            visit(ref.reference)
        elseif ref isa Union{Tuple,NamedTuple}
            foreach(visit,ref)
        end
    end
    visit(input); visit(bindings); foreach(s->visit(s.sources),slots)
    ViewICN(slots,child,input,bindings,weights,layers,lengths,blocks,
        firstchild:length(weights),Dict{Symbol,Any}(),params)
end
view_weight_blocks(n::ViewICN)=copy(n.blocks)
function check_weights_validity(n::ViewICN, weights::AbstractVector{Bool})
    length(weights)==length(n.weights) || throw(DimensionMismatch("view weights"))
    for (b,layer) in zip(n.blocks,n.layers)
        active=count(identity,view(weights,b.range))
        (layer.mutex ? active==1 : active>=1) || return false
    end
    check_weights_validity(n.child,view(weights,n.child_range))
end
function _view_choice(weights,blocks,node,role)
    b=only(b for b in blocks if b.node==node && b.role==role)
    something(findfirst(view(weights,b.range)))
end

function _integer_view_operation(op,args)
    all(a->a isa Integer,args) || throw(ArgumentError("expression views currently require integer operands"))
    a=map(BigInt,args)
    op===:id && return (a[1],true)
    op===:neg && return (-a[1],true)
    op===:abs && return (abs(a[1]),true)
    if op in (:not,:and,:or,:xor)
        all(v->v in (0,1),a) || return (big(0),false)
        b=map(!iszero,a)
        v=op===:not ? !b[1] : op===:and ? b[1] && b[2] :
          op===:or ? b[1] || b[2] : xor(b[1],b[2])
        return (BigInt(v),true)
    end
    op===:add && return (a[1]+a[2],true)
    op===:sub && return (a[1]-a[2],true)
    op===:mul && return (a[1]*a[2],true)
    op in (:div,:rem) && return iszero(a[2]) ? (big(0),false) :
        (op===:div ? div(a[1],a[2]) : rem(a[1],a[2]),true)
    if op===:pow
        a[2]<0 && abs(a[1])!=1 && return (big(0),false)
        # ±1 admit negative integer exponents without converting to Float64.
        abs(a[1])==1 && return (a[1]==1 || iseven(a[2]) ? big(1) : big(-1),true)
        return (a[1]^a[2],true)
    end
    op===:min && return (min(a...),true)
    op===:max && return (max(a...),true)
    relation=op===:eq ? (==) : op===:ne ? (!=) : op===:lt ? (<) :
        op===:le ? (<=) : op===:gt ? (>) : op===:ge ? (>=) : nothing
    isnothing(relation) && throw(ArgumentError("unknown scalar operation $op"))
    (BigInt(relation(a...)),true)
end

function _view_resolver(nodes,weights,blocks,x,parameters)
    cache=Vector{Any}(undef,length(nodes)); ready=falses(length(nodes))
    function resolve(ref)
        if ref isa ViewReference
            i=ref.node
            ready[i] && return cache[i]
            slot=nodes[i]
            op=slot.operations[_view_choice(weights,blocks,i,:operation)]
            operand(j)=resolve(slot.sources[_view_choice(weights,blocks,i,Symbol("operand",j))])
            result=if op===:if
                condition,defined=operand(1)
                !defined || !(condition in (0,1)) ? (big(0),false) : operand(iszero(condition) ? 3 : 2)
            else
                args=ntuple(operand,slot.arity)
                all(last,args) ? _integer_view_operation(op,map(first,args)) : (big(0),false)
            end
            cache[i]=result; ready[i]=true
            return result
        elseif ref isa ReshapedReference
            values,defined=resolve(ref.reference)
            return (_reshape_binding(values,ref.shape),defined)
        elseif ref isa Union{Tuple,NamedTuple}
            resolved=map(resolve,ref)
            return (map(first,resolved),all(last,resolved))
        end
        (_resolve_parameter(ref,x,parameters),true)
    end
    resolve
end
function evaluate(n::ViewICN, config::Configuration;
        weights_validity=check_weights_validity(n,n.weights),parameters...)
    weights_validity || return Inf
    resolve=_view_resolver(n.nodes,n.weights,n.blocks,config.x,(;n.constants...,parameters...))
    x,dx=resolve(n.input); p,dp=resolve(n.bindings)
    if !(dx && dp) && _view_choice(n.weights,n.blocks,0,:definedness)==1
        return 1.0
    end
    apply!(n.child,view(n.weights,n.child_range))
    evaluate(n.child,Solution(x);p...)
end
struct DecodedViewICN{N <: ViewICN}
    network::N
end
composition(n::ViewICN;name::Symbol=gensym(:view_icn)) =
    check_weights_validity(n,n.weights) ? DecodedViewICN(deepcopy(n)) : throw(ArgumentError("invalid view weights"))
(d::DecodedViewICN)(x;parameters...)=evaluate(d.network,Solution(x);parameters...)
function symbols(n::ViewICN;simplified=true)
    apply!(n.child,view(n.weights,n.child_range))
    (;views=[(;operation=s.operations[_view_choice(n.weights,n.blocks,i,:operation)],
        operands=ntuple(j->s.sources[_view_choice(n.weights,n.blocks,i,Symbol("operand",j))],s.arity))
        for (i,s) in enumerate(n.nodes)],
      gate=(:require_defined,:use_totalized_values)[_view_choice(n.weights,n.blocks,0,:definedness)],
      child=symbols(n.child;simplified))
end
function code(n::ViewICN,language::Symbol=:maths;name="composition",simplified=true)
    language===:maths || throw(ArgumentError("view ICNs currently export :maths"))
    s=symbols(n;simplified)
    "$name(x) = views{$(repr(s.views)); gate=$(s.gate); input=$(repr(n.input)); bindings=$(repr(n.bindings)); " *
        code(n.child,:maths;name="penalty",simplified) * "}"
end
code(d::DecodedViewICN,args...;kwargs...)=code(d.network,args...;kwargs...)
symbols(d::DecodedViewICN;kwargs...)=symbols(d.network;kwargs...)
canonical_key(n::ViewICN;simplified=true)=code(n,:maths;name="view_icn",simplified)
canonical_key(d::DecodedViewICN;kwargs...)=canonical_key(d.network;kwargs...)
compose(n::ViewICN;name::Symbol=gensym(:view_icn))=(composition(n;name),code(n,:maths;name=String(name)))
incremental_supported(::Union{ViewICN,DecodedViewICN})=false
