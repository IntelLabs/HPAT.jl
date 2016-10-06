#=
Copyright (c) 2016, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
=#


module DomainPass

import ParallelAccelerator
import ParallelAccelerator.DomainIR
import ParallelAccelerator.ParallelIR.computeLiveness

using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools.LambdaHandling
using CompilerTools.Helper
using CompilerTools.AstWalker
#using Debug

import HPAT
import HPAT.CaptureAPI
import HPAT.CaptureAPI.getColName
import HPAT.CaptureAPI.revColName

import CompilerTools.AstWalker
import CompilerTools.AliasAnalysis

mk_alloc(typ, s) = Expr(:alloc, typ, s)
mk_call(fun,args) = Expr(:call, fun, args...)

const generatedFuncs = [:__hpat_data_source_HDF5_open,
                        :__hpat_data_source_HDF5_size,
                        :__hpat_get_H5_dim_size,
                        :__hpat_data_source_HDF5_read,
                        :__hpat_data_source_HDF5_close,
                        :__hpat_data_source_TXT_open,
                        :__hpat_data_source_TXT_size,
                        :__hpat_get_TXT_dim_size,
                        :__hpat_data_source_TXT_read,
                        :__hpat_data_source_TXT_close,
                        :__hpat_filter,
                        :__hpat_join,
                        :__hpat_aggregate,
                        ]


const generatedExprHeads = [:alloc,
                            :filter,
                            :join,
                            :aggregate]

function remove_dead(node :: Expr, data :: ParallelAccelerator.ParallelIR.RemoveDeadState, top_level_number, is_top_level, read)
    # FIX FIX FIX  This is wrong...for now we just assume anything HPAT added to AST is not dead.
    if in(node.head, generatedExprHeads)
        return node
    end
    return ParallelAccelerator.ParallelIR.remove_dead(node, data, top_level_number, is_top_level, read)
end

function remove_dead(node :: ANY, data :: ParallelAccelerator.ParallelIR.RemoveDeadState, top_level_number, is_top_level, read)
    return ParallelAccelerator.ParallelIR.remove_dead(node, data, top_level_number, is_top_level, read)
end

# ENTRY to DomainPass
function from_root(function_name, ast)

    linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    @dprintln(1,"Starting main DomainPass.from_root.  function = ", function_name, " ast = ", linfo, body)

    lives = computeLiveness(body, linfo)

    tableCols, tableTypes, tableIds = get_table_meta(body)
    @dprintln(3,"HPAT tables: ", tableCols,tableTypes)
    state::DomainState = DomainState(linfo, tableCols, tableTypes, tableIds, 0, -1, lives)

    # transform body
    body.args = from_toplevel_body(body.args, state)
    @dprintln(1,"DomainPass.from_root returns function = ", function_name, " body = ", body)
    #println("DomainPass.from_root returns function = ", function_name, " body = ", body)

    lives = computeLiveness(body, linfo)
    body = ParallelAccelerator.ParallelIR.AstWalk(body, remove_dead, ParallelAccelerator.ParallelIR.RemoveDeadState(lives))
    @dprintln(1,"Body after dead code elimination.  function = ", function_name, " ast = ", linfo, body)

    # transform body
    return LambdaVarInfoToLambda(state.linfo, body.args, ParallelAccelerator.DomainIR.AstWalk)
end

# information about AST gathered and used in DomainPass
type DomainState
    linfo  :: LambdaVarInfo
    tableCols::Dict{Symbol,Vector{Symbol}}
    tableTypes::Dict{Symbol,Vector{Symbol}}
    tableIds::Dict{Int,Symbol}
    # a unique id for domain operations (data sources/sinks, table operations)
    unique_id::Int
    # first column src of each table stores id so others reads its size variable
    prev_table_first_src_num::Int
    lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
end

"""
    get a unique id for renaming
"""
function get_unique_id(state)
  state.unique_id += 1
  return state.unique_id
end

function get_table_meta(body)
    first_arg = body.args[1]
    if isa(first_arg, Expr) && first_arg.head==:meta
        for meta in first_arg.args
            if meta.head==:hpat_tables
                @dprintln(3, "hpat tables found: ", meta)
                return meta.args[1], meta.args[2], meta.args[3]
            end
        end
    end
    return Dict{Symbol,Vector{Symbol}}(),Dict{Symbol,Vector{Symbol}}()
end

# nodes are :body of AST
function from_toplevel_body(nodes::Array{Any,1}, state::DomainState)
    res::Array{Any,1} = []
    nodes = translate_table_oprs(nodes,state)
    @dprintln(3,"body after table translation: ", Expr(:body,nodes...))

    for node in nodes
        new_exprs = from_expr(node, state)
        append!(res, new_exprs)
    end
    return res
end

function from_expr(node::Expr, state::DomainState)
    head = node.head
    if head==:(=)
        return from_assignment(node, state)
    elseif getHPATcall(node)==:data_sink_HDF5
        return translate_data_sink_HDF5(node.args[2], node.args[3], node.args[4], state)
    elseif head==:return && getHPATcall(node.args[1])==:data_sink_HDF5
        snode = node.args[1]
        s_expr = translate_data_sink_HDF5(snode.args[2], snode.args[3], snode.args[4], state)
        return Any[s_expr;Expr(:return,nothing)]
    else
        return [node]
    end
end


function from_expr(node::Any, state::DomainState)
    return [node]
end

function translate_table_oprs(nodes::Array{Any,1}, state::DomainState)
    # array of new nodes after translation
    new_nodes = []
    # number of nodes to skip in the loop
    # translating operations requires skipping some nodes after current node
    skip = 0
    for i in 1:length(nodes)
        if skip!=0
            skip-=1
            continue
        end
        if isa(nodes[i],Expr) && nodes[i].head==:(=) && (isCall(nodes[i].args[2]) || isInvoke(nodes[i].args[2]))
            local func_call = getCallFunction(nodes[i].args[2])
            local args = getCallArguments(nodes[i].args[2])
            # convert :invoke to :call to be consistent
            if isInvoke(nodes[i].args[2])
                nodes[i].args[2].head = :call
                nodes[i].args[2].args = [func_call; args]
            end
            if func_call==GlobalRef(HPAT.API, :join)
                @dprintln(3,"new nodes before join: ", Expr(:body,new_nodes...))
                remove_before,remove_after,ast = translate_join(nodes[i], nodes, i,state)
                skip += remove_after
                s_start = (length(new_nodes)-remove_before)+1
                s_end = length(new_nodes)
                # replace ast nodes with new node
                splice!(new_nodes, s_start:s_end, ast)
                @dprintln(3,"new nodes after join: ", Expr(:body,new_nodes...))
                continue
            elseif func_call==GlobalRef(HPAT.API, :aggregate)
                @dprintln(3,"new nodes before aggregate: ", Expr(:body,new_nodes...))
                remove_before,remove_after,ast = translate_aggregate(nodes, i, nodes[i],state)
                skip += remove_after
                s_start = (length(new_nodes)-remove_before)+1
                s_end = length(new_nodes)
                # replace ast nodes with new node
                splice!(new_nodes, s_start:s_end, ast)
                @dprintln(3,"new nodes after aggregate: ", Expr(:body,new_nodes...))
                continue
            elseif func_call==GlobalRef(HPAT.API, :table_filter!)
                @dprintln(3,"new nodes before filter: ", Expr(:body,new_nodes...))
                # returns: new ast :filter node
                # number of junk nodes to remove AFTER the filter call
                # number of junk nodes to remove BEFORE the filter call
                remove_before,remove_after,ast = translate_filter(nodes,i,nodes[i],state)
                skip += remove_after
                s_start = (length(new_nodes)-remove_before)+1
                s_end = length(new_nodes)
                # replace ast nodes with new node
                splice!(new_nodes, s_start:s_end, ast)
                @dprintln(3,"new nodes after filter: ", Expr(:body,new_nodes...))
                continue
            end
        # TODO: any recursive case?
        # elseif isa(nodes[i],Expr) && nodes[i].head==:block
        elseif isCall(nodes[i]) || isInvoke(nodes[i])
            func_call = getCallFunction(nodes[i])
            args = getCallArguments(nodes[i])
            # convert :invoke to :call to be consistent
            if isInvoke(nodes[i])
                nodes[i].head = :call
                nodes[i].args = [func_call; args]
            end
        end
        push!(new_nodes, nodes[i])
    end
    return new_nodes
end

""" Translate table_filter to Expr(:filter, cond_arr, t1 ,col_arrs...) and remove array of array garbage

    returns: number of junk nodes to remove before the filter call
             number of junk nodes to remove after the filter call
             new ast :filter node

    example:
    _6 = (_4 .> 2)::BitArray{1}
    _7 = (Core.ccall)(:jl_alloc_array_1d,(Core.apply_type)(Core.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},(Core.svec)(Core.Any,Core.Int)::SimpleVector,(Core.apply_type)(Core.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},0,2,0)::Array{Array{T,1},1}
    _8 = _4
    (ParallelAccelerator.API.setindex!)(_7,_8,1)::Array{Array{T,1},1}
    _8
    _9 = _5
    (ParallelAccelerator.API.setindex!)(_7,_9,2)::Array{Array{T,1},1}
    _9
    _10 = (HPAT.API.table_filter!)(1,3,_6,_7)
    SSAValue(0) = (ParallelAccelerator.API.getindex)(_10,1)
    _11 = (Base.convert)(Array{Int64,1},SSAValue(0))::Array{Int64,1}
    SSAValue(1) = (ParallelAccelerator.API.getindex)(_10,2)
    _12 = (Core.typeassert)((Base.convert)(Array{Float64,1},SSAValue(1)),Array{Float64,1})::Array{Float64,1}
"""
function translate_filter(nodes::Array{Any,1},curr_pos,filter_node::Expr,state)
    @dprintln(3,"translating filter: ",filter_node)

    opr_num = get_unique_id(state)

    cond_arr = toLHSVar(filter_node.args[2].args[4])
    in_arr_of_arrs = toLHSVar(filter_node.args[2].args[5])
    out_arr_of_arrs = toLHSVar(filter_node.args[1])
    # TODO: remove arr_of_arrs from Lambda
    # convert _filter_t1 to t1
    in_table_name = state.tableIds[filter_node.args[2].args[2]]
    out_table_name = state.tableIds[filter_node.args[2].args[3]]

    in_cols = state.tableCols[in_table_name]
    in_col_arrs = []  #map(x->getColName(in_table_name, x), in_cols)
    in_num_cols = length(in_cols)

    i = curr_pos-1
    while !isAllocAssignment(nodes[i])
        if isArraySet(nodes[i])
            set_in = nodes[i].args[3]
            # previous node is array_col = set_in
            @assert nodes[i-1].head==:(=) && nodes[i-1].args[1]==set_in
            array_col = nodes[i-1].args[2]
            # remove variable?
            #CompilerTools.LambdaHandling.
            push!(in_col_arrs, array_col)
        end
        i -= 1
    end
    in_col_arrs = in_col_arrs[end:-1:1]
    remove_before = curr_pos-i

    out_cols = state.tableCols[out_table_name]
    out_col_arrs = []  #map(x->getColName(out_table_name, x), out_cols)
    out_num_cols = length(out_cols)
    for k in curr_pos+2:2:curr_pos+2*out_num_cols
        #@assert nodes[i].head==:(=) && nodes[i].args[2].args[1]==GlobalRef(Core,:typeassert)
        #println(nodes[k])
        push!(out_col_arrs, nodes[k].args[1])
    end

    # remove temp array assignment and setindex!() for each column, remove array of array allocation

    # remove type convert calls after filter()
    remove_after = 2*out_num_cols
    new_filter_node = Expr(:filter, cond_arr, out_table_name, in_table_name,
                                in_cols, out_col_arrs, in_col_arrs, opr_num)
    @dprintln(3,"filter remove_before: ",remove_before," remove_after: ",remove_after," filter_node: ",filter_node)
    return remove_before, remove_after, Any[new_filter_node]
end

""" Translate join to Expr(:join, t3,t1,t2,out_cols, in1_cols, in2_cols) and remove array of array garbage

    returns: number of junk nodes to remove before the join call
             number of junk nodes to remove after the join call
             new ast :join node
    example:
    _8 = (Core.ccall)(:jl_alloc_array_1d,(Core.apply_type)(Core.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},(Core.svec)(Core.Any,Core.Int)::SimpleVector,(Core.apply_type)(Core.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},0,2,0)::Array{Array{T,1},1}
    _9 = _4
    (ParallelAccelerator.API.setindex!)(_8,_9,1)::Array{Array{T,1},1}
    _9
    _10 = _5
    (ParallelAccelerator.API.setindex!)(_8,_10,2)::Array{Array{T,1},1}
    _10
    _11 = (Core.ccall)(:jl_alloc_array_1d,(Core.apply_type)(Core.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},(Core.svec)(Core.Any,Core.Int)::SimpleVector,(Core.apply_type)(Core.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},0,2,0)::Array{Array{T,1},1}
    _12 = _6
    (ParallelAccelerator.API.setindex!)(_11,_12,1)::Array{Array{T,1},1}
    _12
    _13 = _7
    (ParallelAccelerator.API.setindex!)(_11,_13,2)::Array{Array{T,1},1}
    _13
    _14 = (HPAT.API.join)(3,1,2,_8,_11)
    SSAValue(0) = (ParallelAccelerator.API.getindex)(_14,1)
    _15 = (Base.convert)(Array{Int64,1},SSAValue(0))::Array{Int64,1}
    SSAValue(1) = (ParallelAccelerator.API.getindex)(_14,2)
    _16 = (Core.typeassert)((Base.convert)(Array{Float64,1},SSAValue(1)),Array{Float64,1})::Array{Float64,1}
    SSAValue(2) = (ParallelAccelerator.API.getindex)(_14,3)
    _17 = (Core.typeassert)((Base.convert)(Array{Float64,1},SSAValue(2)),Array{Float64,1})::Array{Float64,1}
"""
function translate_join(join_node, nodes, curr_pos, state)
    @dprintln(3, "translating join: ", join_node)

    opr_num = get_unique_id(state)

    out_arr = toLHSVar(join_node.args[1])
    local t3_id::Int = join_node.args[2].args[2]
    local t1_id::Int = join_node.args[2].args[3]
    local t2_id::Int = join_node.args[2].args[4]
    in1_arr = toLHSVar(join_node.args[2].args[5])
    in2_arr = toLHSVar(join_node.args[2].args[6])

    # convert _join_out_t3 to t3
    t3 = state.tableIds[t3_id]
    # convert _join_t1 to t1
    t1 = state.tableIds[t1_id]
    t2 = state.tableIds[t2_id]

    t3_cols = state.tableCols[t3]
    t1_cols = state.tableCols[t1]
    t2_cols = state.tableCols[t2]
    t3_num_cols = length(t3_cols)
    t1_num_cols = length(t1_cols)
    t2_num_cols = length(t2_cols)

    t2_end = 2*t2_num_cols

    t1_start = t2_end + 3
    t1_end = 2*(t1_num_cols-1) + t1_start
    # Extract inputs columns from nodes above join node
    t2_cols_sorted = []
    t1_cols_sorted = []

    # assuming Julia doesn't reorder nodes, produces same AST!
    # read columns until array of array allocation in AST
    # read t2 columns
    i = curr_pos-1
    while !isAllocAssignment(nodes[i])
        if isArraySet(nodes[i])
            set_in = nodes[i].args[3]
            # previous node is array_col = set_in
            @assert nodes[i-1].head==:(=) && nodes[i-1].args[1]==set_in
            array_col = nodes[i-1].args[2]
            # remove variable?
            #CompilerTools.LambdaHandling.
            push!(t2_cols_sorted, array_col)
        end
        i -= 1
    end
    # read t1 columns
    i -= 1
    while !isAllocAssignment(nodes[i])
        if isArraySet(nodes[i])
            set_in = nodes[i].args[3]
            # previous node is array_col = set_in
            @assert nodes[i-1].head==:(=) && nodes[i-1].args[1]==set_in "array column set expected"
            array_col = nodes[i-1].args[2]
            # remove variable?
            #CompilerTools.LambdaHandling.
            push!(t1_cols_sorted, array_col)
        end
        i -= 1
    end

    # As we read columns from bottom up we need to reverse them
    t2_cols_sorted = t2_cols_sorted[end:-1:1]
    t1_cols_sorted = t1_cols_sorted[end:-1:1]
    t3_arrs = []
    for k in curr_pos+2:2:curr_pos+2*t3_num_cols
        #@assert nodes[i].head==:(=) && nodes[i].args[2].args[1]==GlobalRef(Core,:typeassert)
        #println(nodes[k])
        push!(t3_arrs, nodes[k].args[1])
    end
    remove_before = curr_pos-i
    remove_after =  2*t3_num_cols
    new_join_node = Expr(:join, t3, t1, t2, t3_cols,
        map(x->get_col_name_from_arr(x,state.linfo), t1_cols_sorted),
        map(x->get_col_name_from_arr(x,state.linfo), t2_cols_sorted),
        t3_arrs, t1_cols_sorted, t2_cols_sorted, opr_num)
    return remove_before, remove_after, Any[new_join_node]
end

function isAllocAssignment(node::Expr)
    if node.head==:(=) && isa(node.args[2],Expr) && node.args[2].head==:call &&
          node.args[2].args[1]==GlobalRef(Core,:ccall)
        return true
    end
    return false
end

isAllocAssignment(node::ANY) = false

function isArraySet(node::Expr)
    if node.head==:call && node.args[1].name==:setindex!
        return true
    end
    return false
end

isArraySet(node::ANY) = false

# Extract col name from array var (e.g. _16 -> :col)
function get_col_name_from_arr(arr, linfo)
    col = CompilerTools.LambdaHandling.lookupVariableName(arr, linfo)
    arr = split(string(col),"@")
    return Symbol(arr[3])
end

"""   Example:
        _5 = _4
        _6 = \$(Expr(:invoke, LambdaInfo for .>(::Array{Float64,1}, ::Float64), :(ParallelAccelerator.API..>), :(_4), 2.0))
        _7 = \$(Expr(:invoke, LambdaInfo for .==(::Array{Float64,1}, ::Float64), :(ParallelAccelerator.API..==), :(_4), 1.0))
        \$(Expr(:invoke, LambdaInfo for sum(::Array{Float64,1}), :(ParallelAccelerator.API.sum), :(_5)))
        _8 = Float64
        _9 = Int64
        _10 = Int64
        SSAValue(4) = (Core.tuple)((Core.tuple)(_5,Main.sum)::Tuple{Array{Float64,1},Base.#sum},(Core.tuple)(_6,Main.length)::Tuple{BitArray{1},Base.#length},(Core.tuple)(_7,Main.length)::Tuple{BitArray{1},Base.#length})::Tuple{Tuple{Array{Float64,1},Base.#sum},Tuple{BitArray{1},Base.#length},Tuple{BitArray{1},Base.#length}}
        # meta: location abstractarray.jl vect 24
        _16 = \$(QuoteNode(Tuple{DenseArray{T,1},Function})) # line 27:
        SSAValue(5) = (Base.nfields)(SSAValue(4))::Int64
        # meta: pop location
        _11 = (HPAT.API.aggregate)(1,2,_3,\$(Expr(:invoke, LambdaInfo for copy!(::Array{Tuple{DenseArray{T,1},Function},1}, ::Tuple{Tuple{Array{Float64,1},Base.#sum},Tuple{BitArray{1},Base.#length},Tuple{BitArray{1},Base.#length}}), :(Base.copy!), :((Core.ccall)(:jl_alloc_array_1d,(Core.apply_type)(Core.Array,Tuple{DenseArray{T,1},Function},1)::Type{Array{Tuple{DenseArray{T,1},Function},1}},(Core.svec)(Core.Any,Core.Int)::SimpleVector,(Core.apply_type)(Core.Array,Tuple{DenseArray{T,1},Function},1)::Type{Array{Tuple{DenseArray{T,1},Function},1}},0,SSAValue(5),0)::Array{Tuple{DenseArray{T,1},Function},1}), SSAValue(4))))
        SSAValue(0) = (ParallelAccelerator.API.getindex)(_11,1)
        _12 = (Base.convert)(Array{Int64,1},SSAValue(0))::Array{Int64,1}
        SSAValue(1) = (ParallelAccelerator.API.getindex)(_11,2)
        _13 = (Core.typeassert)((Base.convert)(Array{Float64,1},SSAValue(1)),Array{Float64,1})::Array{Float64,1}
        SSAValue(2) = (ParallelAccelerator.API.getindex)(_11,3)
        _14 = (Base.convert)(Array{Int64,1},SSAValue(2))::Array{Int64,1}
        SSAValue(3) = (ParallelAccelerator.API.getindex)(_11,4)
        _15 = (Base.convert)(Array{Int64,1},SSAValue(3))::Array{Int64,1} # line 15:
"""
function translate_aggregate(nodes, curr_pos, aggregate_node, state)
    @dprintln(3,"translating aggregate: ",aggregate_node)

    opr_num = get_unique_id(state)

    out_arr = toLHSVar(aggregate_node.args[1])

    t1 = state.tableIds[aggregate_node.args[2].args[2]]
    t2 = state.tableIds[aggregate_node.args[2].args[3]]

    t1_cols = state.tableCols[t1]
    t2_cols = state.tableCols[t2]
    t1_num_cols = length(t1_cols)
    t2_num_cols = length(t2_cols)

    # one typeof() call for each output column except key
    remove_before = t2_num_cols-1
    # extra assignments
    remove_after =  2*t2_num_cols

    # t1's key to aggregate on
    key_arr = toLHSVar(aggregate_node.args[2].args[4])

    agg_list = []

    #   Julia sometimes inlines the aggregate expresson
    aggregate_expr = aggregate_node.args[2].args[5]
    @assert aggregate_expr.head==:invoke || aggregate_expr.head==:call "invalid aggregate AST format"
    local aggregate_expr_call = getCallFunction(aggregate_expr)
    local aggregate_expr_args = getCallArguments(aggregate_expr)

    if aggregate_expr_call.name==:vect
        agg_list = aggregate_expr_args
    else
        @assert aggregate_expr_call.name==:copy! "invalid aggregate AST format"
        i = curr_pos-1
        while !isTupleAssignment(nodes[i])
            i -= 1
        end
        remove_before += curr_pos-i
        @assert nodes[i].args[2].args[1]==GlobalRef(Core,:tuple) "expected aggregate tuple assignment"
        # example: :((Core.tuple)((Core.tuple)(_5,Main.sum)::Tuple{Array{Float64,1},Base.#sum},
        #    (Core.tuple)(_6,Main.length)...
        agg_list = nodes[i].args[2].args[2:end]
    end
    in_e_arr_list = map(x->toLHSVar(x.args[2]), agg_list)
    in_func_list = map(x->x.args[3], agg_list)
    replace_length_unique(in_func_list)


    out_col_arrs = []
    for k in curr_pos+2:2:curr_pos+2*t2_num_cols
        #@assert nodes[i].head==:(=) && nodes[i].args[2].args[1]==GlobalRef(Core,:typeassert)
        #println(nodes[k])
        push!(out_col_arrs, nodes[k].args[1])
    end
    new_aggregate_node = Expr(:aggregate, t2, t1, key_arr, in_e_arr_list, in_func_list, out_col_arrs, opr_num)
    @dprintln(3,"translate_aggregate returning with remove_before = ", remove_before, " remove_after = ", remove_after, " node = ", new_aggregate_node)
    return 0, remove_after, Any[new_aggregate_node]
    #return remove_before, remove_after, Any[new_aggregate_node]
end

# dummy function so liveness in parallel-ir can know the output type (typeOfOpr)
function length_unique(a::Array)
    r::Int = length(a)
    return r
end

function replace_length_unique(func_list)
    for (i,f) in enumerate(func_list)
        if isa(f,GlobalRef) && f.name==:length_unique
            func_list[i] = GlobalRef(HPAT.DomainPass, :length_unique)
        end
    end
end

function isTupleAssignment(node::Expr)
    if node.head==:(=) && isa(node.args[2],Expr) &&
         node.args[2].args[1]==GlobalRef(Core,:tuple)
        return true
    end
    return false
end

isTupleAssignment(node::ANY) = false

# :(=) assignment (:(=), lhs, rhs)
function from_assignment(node::Expr, state)

    # pattern match distributed calls that need domain translation
    hpat_call::Symbol = getHPATcall(node.args[2])
    if hpat_call==:null
        return [node]
    end
    return translate_hpat_dist_calls(node.args[1], node.args[2], hpat_call, state)
end

function translate_hpat_dist_calls(lhs::LHSVar, rhs::Expr, hpat_call::Symbol, state)
    if hpat_call==:data_source_HDF5
        return translate_data_source_HDF5(lhs, rhs, state)
    elseif hpat_call==:data_source_TXT
        return translate_data_source_TXT(lhs, rhs, state)
    elseif hpat_call in [:Kmeans,:LinearRegression,:NaiveBayes]
        # enable OpenMP for DAAL calls
        #HPAT.enableOMP()
        # set type for output since type inference is broken!
        in_typ = CompilerTools.LambdaHandling.getType(rhs.args[2], state.linfo)
        CompilerTools.LambdaHandling.setType(lhs, in_typ, state.linfo)
        # no change
        #return [Expr(:(=),lhs,rhs)]
        # instead of assignment, return a call with lhs as 1st argument
        return [Expr(:call,GlobalRef(HPAT.API,hpat_call), lhs,rhs.args[2:end]...)]
    else # not handled yet
        return [Expr(:(=),lhs,rhs)]
    end
end

function translate_hpat_dist_calls(lhs::ANY, rhs::ANY, hpat_call::Symbol, state)
    return Any[]
end

function getHPATcall(call::Expr)
    if isCall(call) || isInvoke(call)
        local func = getCallFunction(call)
        local args = getCallArguments(call)
        # convert :invoke to :call to be consistent
        if isInvoke(call)
            call.head = :call
            call.args = [func; args]
        end
        # TODO: hack to get around Julia 0.4 function resolution issue (q26)
        # remove in 0.5
        if isa(call.args[1],GlobalRef) && call.args[1].name==:Kmeans
            call.args[1]=GlobalRef(HPAT.API,:Kmeans)
        end
        if isa(call.args[1],GlobalRef) && call.args[1].name==:LinearRegression
            call.args[1]=GlobalRef(HPAT.API,:LinearRegression)
        end
        if isa(call.args[1],GlobalRef) && call.args[1].name==:NaiveBayes
            call.args[1]=GlobalRef(HPAT.API,:NaiveBayes)
        end
        if isa(call.args[1],GlobalRef) && call.args[1].name==:runStencil
            call.args[1]=GlobalRef(ParallelAccelerator.API,:runStencil)
        end
        return getHPATcall_inner(call.args[1])
    end
    return :null
end

function getHPATcall(call::ANY)
    return :null
end

function getHPATcall_inner(func::GlobalRef)
    if func.mod==HPAT.API
        return func.name
    end
    return :null
end

function getHPATcall_inner(func::ANY)
    return :null
end

function translate_data_source_HDF5(lhs::LHSVar, rhs::Expr, state)
    res = Any[]
    dprintln(3,"HPAT data source found ", rhs)
    hdf5_var = rhs.args[3]
    hdf5_file = rhs.args[4]
    # update counter and get data source number
    dsrc_num = get_unique_id(state)
    # if array is part of table, first column stores the sizes and others read
    # allocs should have same sizes so fusion works
    is_table_column_not_first = false
    if table_column_first(hdf5_var, state)
        state.prev_table_first_src_num = dsrc_num
    elseif table_column_notfirst(hdf5_var, state)
        is_table_column_not_first = true
    end
    dsrc_id_var = addTempVariable(Int64, state.linfo)
    push!(res, TypedExpr(Int64, :(=), dsrc_id_var, dsrc_num))
    # get array type
    arr_typ = getType(lhs, state.linfo)
    dims = ndims(arr_typ)
    elem_typ = eltype(arr_typ)
    # generate open call
    # lhs is dummy argument so ParallelIR wouldn't reorder
    open_call = mk_call(GlobalRef(HPAT.API,:__hpat_data_source_HDF5_open), [dsrc_id_var, hdf5_var, hdf5_file, lhs])
    push!(res, open_call)
    # generate array size call
    # arr_size_var = addTempVariable(Tuple, state.linfo)
    # assume 1D for now
    arr_size_var = addTempVariable(ParallelAccelerator.H5SizeArr_t, state.linfo)
    size_call = mk_call(GlobalRef(HPAT.API,:__hpat_data_source_HDF5_size), [dsrc_id_var, lhs])
    push!(res, TypedExpr(arr_size_var, :(=), arr_size_var, size_call))
    # generate array allocation
    size_expr = Any[]
    for i in dims:-1:1
        size_i_name = Symbol("__hpat_h5_dim_size_"*string(dsrc_num)*"_"*string(i))
        size_i = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
            size_i_name, Int64, ISASSIGNEDONCE | ISASSIGNED, state.linfo))
        # size_i = addTempVariable(Int64, state.linfo)
        # table columns just read size from first column array for fusion
        if is_table_column_not_first
            prev_id = state.prev_table_first_src_num
            prev_size_name = Symbol("__hpat_h5_dim_size_"*string(prev_id)*"_"*string(i))
            prev_size_var = CompilerTools.LambdaHandling.lookupLHSVarByName(prev_size_name, state.linfo)
            push!(res, Expr(:(=), size_i, prev_size_var))
        else
            size_i_call = mk_call(GlobalRef(HPAT.API,:__hpat_get_H5_dim_size), [arr_size_var, i])
            push!(res, TypedExpr(Int64, :(=), size_i, size_i_call))
        end
        push!(size_expr, size_i)
    end
    arrdef = TypedExpr(arr_typ, :alloc, elem_typ, size_expr)
    push!(res, TypedExpr(arr_typ, :(=), lhs, arrdef))
    # generate read call
    read_call = mk_call(GlobalRef(HPAT.API,:__hpat_data_source_HDF5_read), [dsrc_id_var, lhs])
    push!(res, read_call)
    close_call = mk_call(GlobalRef(HPAT.API,:__hpat_data_source_HDF5_close), [dsrc_id_var])
    push!(res, close_call)
    return res
end

function table_column_first(hdf5_var, state)
    col_name = Symbol(hdf5_var[2:end])
    for t in values(state.tableCols)
        if t[1]==col_name
            dprintln(3, "first table column src found ", col_name, " ", t)
            return true
        end
    end
    return false
end

function table_column_notfirst(hdf5_var, state)
    col_name = Symbol(hdf5_var[2:end])
    for t in values(state.tableCols)
        if col_name in t[2:end]
            dprintln(3, "notfirst table column src found ", col_name, " ", t)
            return true
        end
    end
    return false
end

function translate_data_source_TXT(lhs::LHSVar, rhs::Expr, state)
    dprintln(3,"TXT data source found ", rhs)
    res = Any[]
    txt_file = rhs.args[3]
    # update counter and get data source number
    dsrc_num = get_unique_id(state)
    dsrc_id_var = addTempVariable(Int64, state.linfo)
    push!(res, TypedExpr(Int64, :(=), dsrc_id_var, dsrc_num))
    # get array type
    arr_typ = getType(lhs, state.linfo)
    dims = ndims(arr_typ)
    elem_typ = eltype(arr_typ)
    # generate open call
    # lhs is dummy argument so ParallelIR wouldn't reorder
    open_call = mk_call(:__hpat_data_source_TXT_open, [dsrc_id_var, txt_file, lhs])
    push!(res, open_call)
    # generate array size call
    # arr_size_var = addTempVariable(Tuple, state.linfo)
    arr_size_var = addTempVariable(ParallelAccelerator.SizeArr_t, state.linfo)
    size_call = mk_call(:__hpat_data_source_TXT_size, [dsrc_id_var, lhs])
    push!(res, TypedExpr(arr_size_var, :(=), arr_size_var, size_call))
    # generate array allocation
    size_expr = Any[]
    for i in dims:-1:1
        size_i_name = Symbol("__hpat_txt_dim_size_"*string(dsrc_num)*"_"*string(i))
        size_i = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
            size_i_name, Int64, ISASSIGNEDONCE | ISASSIGNED, state.linfo))
        #size_i = addTempVariable(Int64, state.linfo)
        size_i_call = mk_call(:__hpat_get_TXT_dim_size, [arr_size_var, i])
        push!(res, TypedExpr(Int64, :(=), size_i, size_i_call))
        push!(size_expr, size_i)
    end
    arrdef = TypedExpr(arr_typ, :alloc, elem_typ, size_expr)
    push!(res, TypedExpr(arr_typ, :(=), lhs, arrdef))
    # generate read call
    read_call = mk_call(:__hpat_data_source_TXT_read, [dsrc_id_var, lhs])
    push!(res, read_call)
    close_call = mk_call(:__hpat_data_source_TXT_close, [dsrc_id_var])
    push!(res, close_call)
    return res
end

            #=
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hpat_Kmeans
                dprintln(3,"kmeans found ", inner_call)
                HPAT.enableOMP()
                lib_call = mk_call(:__hpat_Kmeans, [lhs,inner_call.args[2], inner_call.args[3]])
                return [lib_call]
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hpat_LinearRegression
                dprintln(3,"LinearRegression found ", inner_call)
                HPAT.enableOMP()
                lib_call = mk_call(:__hpat_LinearRegression, [lhs,inner_call.args[2], inner_call.args[3]])
                return [lib_call]
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hpat_NaiveBayes
                dprintln(3,"NaiveBayes found ", inner_call)
                HPAT.enableOMP()
                lib_call = mk_call(:__hpat_NaiveBayes, [lhs,inner_call.args[2], inner_call.args[3], inner_call.args[4]])
                return [lib_call]
            end
        end
    end

    return Any[]
end

function pattern_match_hpat_dist_calls(lhs::Any, rhs::Any, state)
    return Any[]
end
=#
function translate_data_sink_HDF5(y, hdf5_var, hdf5_file, state)
    res = Any[]
    dprintln(3,"HPAT data sink found ", y)
    # update counter and get data source number
    dsrc_num = get_unique_id(state)
    dsrc_id_var = addTempVariable(Int64, state.linfo)
    push!(res, TypedExpr(Int64, :(=), dsrc_id_var, dsrc_num))
    # get array type
    arr_typ = getType(y, state.linfo)
    dims = ndims(arr_typ)
    elem_typ = eltype(arr_typ)
    # generate open call
    # y is dummy argument so ParallelIR wouldn't reorder
    open_call = mk_call(GlobalRef(HPAT.API,:__hpat_data_sink_HDF5_open), [dsrc_id_var, hdf5_var, hdf5_file, y])
    push!(res, open_call)
    #=
    # generate array size call
    # arr_size_var = addTempVariable(Tuple, state.linfo)
    # assume 1D for now
    arr_size_var = addTempVariable(ParallelAccelerator.H5SizeArr_t, state.linfo)
    size_call = mk_call(GlobalRef(HPAT.API,:__hpat_data_source_HDF5_size), [dsrc_id_var, y])
    push!(res, TypedExpr(arr_size_var, :(=), arr_size_var, size_call))
    # generate array allocation
    size_expr = Any[]
    for i in dims:-1:1
        size_i = symbol("__hpat_h5_dim_size_"*string(dsrc_num)*"_"*string(i))
        CompilerTools.LambdaHandling.addLocalVariable(size_i, Int64, ISASSIGNEDONCE | ISASSIGNED, state.linfo)
        # size_i = addTempVariable(Int64, state.linfo)
        size_i_call = mk_call(GlobalRef(HPAT.API,:__hpat_get_H5_dim_size), [arr_size_var, i])
        push!(res, TypedExpr(Int64, :(=), size_i, size_i_call))
        push!(size_expr, size_i)
    end
    arrdef = TypedExpr(arr_typ, :alloc, elem_typ, size_expr)
    push!(res, TypedExpr(arr_typ, :(=), y, arrdef))
    =#
    # generate read call
    write_call = mk_call(GlobalRef(HPAT.API,:__hpat_data_sink_HDF5_write), [dsrc_id_var, hdf5_var, y])
    push!(res, write_call)
    close_call = mk_call(GlobalRef(HPAT.API,:__hpat_data_sink_HDF5_close), [dsrc_id_var])
    push!(res, close_call)
    return res
end

function AstWalkCallback(node::Expr,dw)

    if node.head==:filter
        # Structure: cond, output table, input table, cols, output table cols, input table cols, id
        cond_arr = node.args[1]
        node.args[1] = AstWalker.AstWalk(cond_arr, ParallelAccelerator.DomainIR.AstWalkCallback, dw)
        #out_t = node.args[2]
        in_t = node.args[3]
        node.args[3] = AstWalker.AstWalk(node.args[3], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
        node.args[3]!=in_t && @dprintln(3,"Mapping from :",in_col," to ",in_col_arrs[i])
        col_arrs = node.args[4]
        #out_col_arrs = node.args[5]
        in_col_arrs = node.args[6]
        for i in 1:length(col_arrs)
            #out_col = out_col_arrs[i]
            in_col = in_col_arrs[i]
            #out_col_arrs[i] = AstWalker.AstWalk(out_col_arrs[i], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
            in_col_arrs[i] = AstWalker.AstWalk(in_col_arrs[i], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
            #out_col_arrs!=out_col && @dprintln(3,"Mapping from :",out_col," to ",out_col_arrs[i])
            in_col_arrs!=in_col && @dprintln(3,"Mapping from :",in_col," to ",in_col_arrs[i])
            #@assert col_arrs[i]==old_arr "Table column name shouldn't change for now"
        end
        return node
    elseif node.head==:join
        # Structure: output table, input table 1, input table 2, out cols, in 1 cols, in 2 cols , output table cols, input table 1 cols , input table 2 cols, id
        in_t1 = node.args[2]
        in_t2 = node.args[3]
        node.args[2] = AstWalker.AstWalk(node.args[2], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
        node.args[2]!=in_t1 && @dprintln(3,"Mapping from :",in_t1," to ",node.args[2])
        node.args[3] = AstWalker.AstWalk(node.args[3], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
        node.args[3]!=in_t2 && @dprintln(3,"Mapping from :",in_t2," to ",node.args[3])

        t3_arrs = node.args[7]
        t1_arrs = node.args[8]
        t2_arrs = node.args[9]
        for i in 1:length(t3_arrs)
            old_arr = t3_arrs[i]
            t3_arrs[i] = AstWalker.AstWalk(t3_arrs[i], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
            t3_arrs[i]!=old_arr && @dprintln(3,"Mapping from :",old_arr," to ",t3_arrs[i])
        end
        for i in 1:length(t1_arrs)
            old_arr = t1_arrs[i]
            t1_arrs[i] = AstWalker.AstWalk(t1_arrs[i], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
            t1_arrs[i]!=old_arr && @dprintln(3,"Mapping from :",old_arr," to ",t1_arrs[i])
        end
        for i in 1:length(t2_arrs)
            old_arr = t2_arrs[i]
            t2_arrs[i] = AstWalker.AstWalk(t2_arrs[i], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
            t2_arrs[i]!=old_arr && @dprintln(3,"Mapping from :",old_arr," to ",t2_arrs[i])
        end
        return node
    elseif node.head==:aggregate
        # Structure: output table, input table, groupby key, output expr list, expr rhs list, func list, output list, id
        in_t = node.args[2]
        key_arr = node.args[3]
        out_e_arrs = node.args[4]
        func_arrs = node.args[5]
        out_col_arrs = node.args[6]

        node.args[2] = AstWalker.AstWalk(in_t, ParallelAccelerator.DomainIR.AstWalkCallback, dw)
        node.args[3] = AstWalker.AstWalk(key_arr, ParallelAccelerator.DomainIR.AstWalkCallback, dw)
        for i in 1:length(out_col_arrs)
            old_arr = out_col_arrs[i]
            out_col_arrs[i] = AstWalker.AstWalk(out_col_arrs[i], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
        end
        return node
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function live_cb(node::Expr)

    if node.head==:filter
        @dprintln(3,"DomainPass filter live CB on: ",node)
        # read condition array and column arrays
        # write all column arrays
        cond_arr = node.args[1]
        t = node.args[2]
        cols = node.args[3]
        col_arrs = node.args[4]
        # artificial assignments to signify write to LivenessAnalysis
        assign_exprs = map(x->Expr(:(=),x,1), col_arrs)
        exprs_to_process = Any[cond_arr;col_arrs;assign_exprs]
        @dprintln(3,"DomainPass filter live CB returns: ", exprs_to_process)
        return exprs_to_process
    elseif node.head==:join
        @dprintln(3,"DomainPass join live CB on: ",node)
        t3_arrs = node.args[7]
        t1_arrs = node.args[8]
        t2_arrs = node.args[9]
        # only t3 is written
        assign_exprs = map(x->Expr(:(=),x,1), t3_arrs)
        exprs_to_process = Any[t1_arrs;t2_arrs;assign_exprs]
        @dprintln(3,"DomainPass join live CB returns: ", exprs_to_process)
        return exprs_to_process
    elseif node.head==:aggregate
        @dprintln(3,"DomainPass aggregate live CB on: ",node)
        key_arr = node.args[3]
        in_e_arrs = node.args[4]
        out_col_arrs = node.args[5]
        assign_exprs = map(x->Expr(:(=),x,1), out_col_arrs)
        exprs_to_process = [key_arr;in_e_arrs;assign_exprs]
        @dprintln(3,"DomainPass aggregate live CB returns: ", exprs_to_process)
        return exprs_to_process
    elseif node.head==:call && isa(node.args[1],GlobalRef) && node.args[1]==GlobalRef(Core,:tuple) &&
          length(node.args)==3 && isa(node.args[3],GlobalRef) && node.args[3].name==:length_unique
        @dprintln(3,"DomainPass length_unique live CB returns: ", Any[node.args[2]])
        return Any[node.args[2]]
    end
    return nothing
end

function alias_cb(node::Expr)
    if node.head==:filter || node.head==:join || node.head==:aggregate
        return CompilerTools.AliasAnalysis.NotArray
    end
    return nothing
end

end # module
