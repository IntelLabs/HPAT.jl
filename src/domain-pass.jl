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

using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools.LambdaHandling
using CompilerTools.Helper

import HPAT
import HPAT.CaptureAPI
import HPAT.CaptureAPI.getColName
import HPAT.CaptureAPI.revColName

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
                        :__hpat_data_source_TXT_close]


const generatedExprHeads = [:alloc,
                            :filter,
                            :join,
                            :aggregate]

# ENTRY to DomainPass
function from_root(function_name, ast)
    @dprintln(1,"Starting main DomainPass.from_root.  function = ", function_name, " ast = ", ast)

    linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    tableCols, tableTypes = get_table_meta(body)
    @dprintln(3,"HPAT tables: ", tableCols,tableTypes)
    state::DomainState = DomainState(linfo, 0, tableCols, tableTypes)
    
    # transform body
    body.args = from_toplevel_body(body.args, state)
    @dprintln(1,"DomainPass.from_root returns function = ", function_name, " body = ", body)
    #println("DomainPass.from_root returns function = ", function_name, " body = ", body)
    return LambdaVarInfoToLambda(state.linfo, body.args)
end

# information about AST gathered and used in DomainPass
type DomainState
    linfo  :: LambdaVarInfo
    data_source_counter::Int64 # a unique counter for data sources in program
    tableCols::Dict{Symbol,Vector{Symbol}}
    tableTypes::Dict{Symbol,Vector{Symbol}}
end

function get_table_meta(body)
    first_arg = body.args[1]
    if isa(first_arg, Expr) && first_arg.head==:meta
        for meta in first_arg.args
            if meta.head==:hpat_tables
                @dprintln(3, "hpat tables found: ", meta)
                return meta.args[1],meta.args[2]
            end
        end
    end
    return Dict{Symbol,Vector{Symbol}}(),Dict{Symbol,Vector{Symbol}}()
end

# nodes are :body of AST
function from_toplevel_body(nodes::Array{Any,1}, state::DomainState)
    res::Array{Any,1} = []
    nodes = translate_table_oprs(nodes,state)
    @dprintln(3,"body after table translation: ", nodes)
    
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
        out = []
        if skip!=0 
            skip-=1
            continue
        end
        if isa(nodes[i],Expr) && nodes[i].head==:(=) && isCall(nodes[i].args[2])
            func_call = nodes[i].args[2].args[1]
            if func_call==GlobalRef(HPAT.API, :join)
                remove_before,remove_after,ast = translate_join(nodes[i],state)
                skip += remove_after
                s_start = (length(new_nodes)-remove_before)+1
                s_end = length(new_nodes)
                # replace ast nodes with new node
                splice!(new_nodes, s_start:s_end, ast)
                continue
            elseif func_call==GlobalRef(HPAT.API, :aggregate)
                remove_before,remove_after,ast = translate_aggregate(nodes[i],state)
                skip += remove_after
                s_start = (length(new_nodes)-remove_before)+1
                s_end = length(new_nodes)
                # replace ast nodes with new node
                splice!(new_nodes, s_start:s_end, ast)
                continue
            end
        # TODO: any recursive case?
        # elseif isa(nodes[i],Expr) && nodes[i].head==:block
        elseif isCall(nodes[i])
            func_call = nodes[i].args[1]
            if func_call==GlobalRef(HPAT.API, :table_filter!)
                # returns: new ast :filter node
                # number of junk nodes to remove AFTER the filter call
                # number of junk nodes to remove BEFORE the filter call
                remove_before,remove_after,ast = translate_filter(nodes[i],state)
                skip += remove_after
                s_start = (length(new_nodes)-remove_before)+1
                s_end = length(new_nodes)
                # replace ast nodes with new node
                splice!(new_nodes, s_start:s_end, ast)
                continue
            end
        end
        if length(out)==0
            push!(new_nodes, nodes[i])
        else
            append!(new_nodes,out)
        end
    end
    return new_nodes
end

"""
Translate table_filter to Expr(:filter, cond_arr, t1 ,col_arrs...) and remove array of array garbage

    returns: number of junk nodes to remove before the filter call 
             number of junk nodes to remove after the filter call
             new ast :filter node

    example:
        _sale_items_cond_e = _sale_items_i_category::Array{Int64,1} .== category::Int64::BitArray{1}
        _filter_sale_items = (top(ccall))(:jl_alloc_array_1d,(top(apply_type))(Base.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},(top(svec))(Base.Any,Base.Int)::SimpleVector,Array{Array{T,1},1},0,4,0)::Array{Array{T,1},1}
        ##7580 = _sale_items_ss_item_sk::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_filter_sale_items::Array{Array{T,1},1},##7580::Array{Int64,1},1)::Array{Array{T,1},1}
        ##7581 = _sale_items_ss_customer_sk::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_filter_sale_items::Array{Array{T,1},1},##7581::Array{Int64,1},2)::Array{Array{T,1},1}
        ##7582 = _sale_items_i_category::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_filter_sale_items::Array{Array{T,1},1},##7582::Array{Int64,1},3)::Array{Array{T,1},1}
        ##7583 = _sale_items_i_class_id::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_filter_sale_items::Array{Array{T,1},1},##7583::Array{Int64,1},4)::Array{Array{T,1},1}
        (HPAT.API.table_filter!)(_sale_items_cond_e::BitArray{1},_filter_sale_items::Array{Array{T,1},1})::Void
        _sale_items_ss_item_sk = (top(convert))(Array{Int64,1},(ParallelAccelerator.API.getindex)(_filter_sale_items::Array{Array{T,1},1},1)::Array{T,1})::Array{Int64,1}
        _sale_items_ss_customer_sk = (top(convert))(Array{Int64,1},(ParallelAccelerator.API.getindex)(_filter_sale_items::Array{Array{T,1},1},2)::Array{T,1})::Array{Int64,1}
        _sale_items_i_category = (top(convert))(Array{Int64,1},(ParallelAccelerator.API.getindex)(_filter_sale_items::Array{Array{T,1},1},3)::Array{T,1})::Array{Int64,1}
        _sale_items_i_class_id = (top(convert))(Array{Int64,1},(ParallelAccelerator.API.getindex)(_filter_sale_items::Array{Array{T,1},1},4)::Array{T,1})::Array{Int64,1} # /Users/etotoni/.julia/v0.4/HPAT/examples/queries_devel/tests/test_q26.jl, line 15:
"""
function translate_filter(filter_node::Expr,state)
    @dprintln(3,"translating filter: ",filter_node)
    cond_arr = toLHSVar(filter_node.args[2])
    arr_of_arrs = toLHSVar(filter_node.args[3])
    # TODO: remove arr_of_arrs from Lambda
    # convert _filter_t1 to t1
    table_name = Symbol(string(arr_of_arrs)[9:end])
    cols = state.tableCols[table_name]
    col_arrs = map(x->getColName(table_name, x), cols)
    num_cols = length(cols)
    
    # remove temp array assignment and setindex!() for each column, remove array of array allocation
    remove_before = 2*num_cols+1;
    # remove type convert calls after filter() 
    remove_after = num_cols
    new_filter_node = Expr(:filter, cond_arr, table_name, cols, col_arrs)
    @dprintln(3,"filter remove_before: ",remove_before," remove_after: ",remove_after," filter_node: ",filter_node)
    return remove_before, remove_after, [new_filter_node]
end

"""
Translate join to Expr(:join, t3,t1,t2,out_cols, in1_cols, in2_cols) and remove array of array garbage

    returns: number of junk nodes to remove before the filter call 
             number of junk nodes to remove after the filter call
             new ast :filter node
             
    example:
        _join_store_sales = (top(ccall))(:jl_alloc_array_1d,(top(apply_type))(Base.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},(top(svec))(Base.Any,Base.Int)::SimpleVector,Array{Array{T,1},1},0,2,0)::Array{Array{T,1},1}
        ##7583 = _store_sales_ss_item_sk::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_join_store_sales::Array{Array{T,1},1},##7583::Array{Int64,1},1)::Array{Array{T,1},1}
        ##7584 = _store_sales_ss_customer_sk::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_join_store_sales::Array{Array{T,1},1},##7584::Array{Int64,1},2)::Array{Array{T,1},1}
        _join_item = (top(ccall))(:jl_alloc_array_1d,(top(apply_type))(Base.Array,Array{T,1},1)::Type{Array{Array{T,1},1}},(top(svec))(Base.Any,Base.Int)::SimpleVector,Array{Array{T,1},1},0,3,0)::Array{Array{T,1},1}
        ##7585 = _item_i_item_sk::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_join_item::Array{Array{T,1},1},##7585::Array{Int64,1},1)::Array{Array{T,1},1}
        ##7586 = _item_i_category::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_join_item::Array{Array{T,1},1},##7586::Array{Int64,1},2)::Array{Array{T,1},1}
        ##7587 = _item_i_class_id::Array{Int64,1}
        (ParallelAccelerator.API.setindex!)(_join_item::Array{Array{T,1},1},##7587::Array{Int64,1},3)::Array{Array{T,1},1}
        _join_out_sale_items = (HPAT.API.join)(_join_store_sales::Array{Array{T,1},1},_join_item::Array{Array{T,1},1})::Array{Array{Any,1},1}
        GenSym(2) = (ParallelAccelerator.API.getindex)(_join_out_sale_items::Array{Array{Any,1},1},1)::Array{Any,1}
        GenSym(3) = (Base.arraysize)(GenSym(2),1)::Int64
        GenSym(5) = (top(ccall))(:jl_alloc_array_1d,(top(apply_type))(Base.Array,Int64,1)::Type{Array{Int64,1}},(top(svec))(Base.Any,Base.Int)::SimpleVector,Array{Int64,1},0,GenSym(3),0)::Array{Int64,1}
        _sale_items_ss_item_sk = (Base.copy!)($(Expr(:new, :((top(getfield))(Base,:LinearFast)::Type{Base.LinearFast}))),GenSym(5),$(Expr(:new, :((top(getfield))(Base,:LinearFast)::Type{Base.LinearFast}))),GenSym(2))::Array{Int64,1}
        GenSym(6) = (ParallelAccelerator.API.getindex)(_join_out_sale_items::Array{Array{Any,1},1},2)::Array{Any,1}
        GenSym(7) = (Base.arraysize)(GenSym(6),1)::Int64
        GenSym(9) = (top(ccall))(:jl_alloc_array_1d,(top(apply_type))(Base.Array,Int64,1)::Type{Array{Int64,1}},(top(svec))(Base.Any,Base.Int)::SimpleVector,Array{Int64,1},0,GenSym(7),0)::Array{Int64,1}
        _sale_items_ss_customer_sk = (Base.copy!)($(Expr(:new, :((top(getfield))(Base,:LinearFast)::Type{Base.LinearFast}))),GenSym(9),$(Expr(:new, :((top(getfield))(Base,:LinearFast)::Type{Base.LinearFast}))),GenSym(6))::Array{Int64,1}
        GenSym(10) = (ParallelAccelerator.API.getindex)(_join_out_sale_items::Array{Array{Any,1},1},3)::Array{Any,1}
        GenSym(11) = (Base.arraysize)(GenSym(10),1)::Int64
        GenSym(13) = (top(ccall))(:jl_alloc_array_1d,(top(apply_type))(Base.Array,Int64,1)::Type{Array{Int64,1}},(top(svec))(Base.Any,Base.Int)::SimpleVector,Array{Int64,1},0,GenSym(11),0)::Array{Int64,1}
        _sale_items_i_category = (Base.copy!)($(Expr(:new, :((top(getfield))(Base,:LinearFast)::Type{Base.LinearFast}))),GenSym(13),$(Expr(:new, :((top(getfield))(Base,:LinearFast)::Type{Base.LinearFast}))),GenSym(10))::Array{Int64,1}
        GenSym(14) = (ParallelAccelerator.API.getindex)(_join_out_sale_items::Array{Array{Any,1},1},4)::Array{Any,1}
        GenSym(15) = (Base.arraysize)(GenSym(14),1)::Int64
        GenSym(17) = (top(ccall))(:jl_alloc_array_1d,(top(apply_type))(Base.Array,Int64,1)::Type{Array{Int64,1}},(top(svec))(Base.Any,Base.Int)::SimpleVector,Array{Int64,1},0,GenSym(15),0)::Array{Int64,1}
        _sale_items_i_class_id = (Base.copy!)($(Expr(:new, :((top(getfield))(Base,:LinearFast)::Type{Base.LinearFast}))),GenSym(17),$(Expr(:new, :((top(getfield))(Base,:LinearFast)::Type{Base.LinearFast}))),GenSym(14))::Array{Int64,1} # /home/etotoni/.julia/v0.4/HPAT/examples/queries_devel/tests/test_q26.jl, line 14:

"""
function translate_join(join_node,state)
    @dprintln(3,"translating join: ",join_node)
    out_arr = toLHSVar(join_node.args[1])
    in1_arr = toLHSVar(join_node.args[2].args[2])
    in2_arr = toLHSVar(join_node.args[2].args[3])
    
    # convert _join_out_t3 to t3
    t3 = Symbol(string(out_arr)[11:end])
    # convert _join_t1 to t1
    t1 = Symbol(string(in1_arr)[7:end])
    t2 = Symbol(string(in2_arr)[7:end])
    
    t3_cols = state.tableCols[t3]
    t1_cols = state.tableCols[t1]
    t2_cols = state.tableCols[t2]
    t3_num_cols = length(t3_cols)
    t1_num_cols = length(t1_cols)
    t2_num_cols = length(t2_cols)
    
    remove_before = 2*t1_num_cols+1+2*t2_num_cols+1
    remove_after =  4*t3_num_cols
    new_join_node = Expr(:join, t3, t1, t2, t3_cols, t1_cols, t2_cols)
    return remove_before, remove_after, [new_join_node]
end

"""
    Example:
        _T_ss_item_count = (Main.typeof)((Base.arraylen)(_customer_i_class_ss_item_count_e::Array{Int64,1})::Int64)::Type{Int64}
        _T_id1 = (Main.typeof)((ParallelAccelerator.API.sum)(1 .* _customer_i_class_id1_e::BitArray{1}::Array{Int64,1})::Int64)::Type{Int64}
        ...
        _T_id15 = (Main.typeof)((ParallelAccelerator.API.sum)(1 .* _customer_i_class_id15_e::BitArray{1}::Array{Int64,1})::Int64)::Type{Int64}
        _agg_out_customer_i_class = (HPAT.API.aggregate)(_sale_items_ss_customer_sk::Array{Int64,1},(top(vect))((top(tuple))(_customer_i_class_ss_item_count_e::Array{Int64,1},Main.length)::Tuple{Array{Int64,1},Function},(top(tuple))(_customer_i_class_id1_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id2_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id3_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id4_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id5_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id6_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id7_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id8_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id9_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id10_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id11_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id12_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id13_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id14_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function},(top(tuple))(_customer_i_class_id15_e::BitArray{1},Main.sum)::Tuple{BitArray{1},Function}))::Array{Array{T,1},1}
        _customer_i_class_ss_customer_sk = (top(convert))(Array{Int64,1},(ParallelAccelerator.API.getindex)(_agg_out_customer_i_class::Array{Array{T,1},1},1)::Array{T,1})::Array{Int64,1}
        _customer_i_class_ss_item_count = (top(convert))(Array{Int64,1},(ParallelAccelerator.API.getindex)(_agg_out_customer_i_class::Array{Array{T,1},1},2)::Array{T,1})::Array{Int64,1}
        _customer_i_class_id1 = (top(convert))(Array{Int64,1},(ParallelAccelerator.API.getindex)(_agg_out_customer_i_class::Array{Array{T,1},1},3)::Array{T,1})::Array{Int64,1}
        ...
        _customer_i_class_id15 = (top(convert))(Array{Int64,1},(ParallelAccelerator.API.getindex)(_agg_out_customer_i_class::Array{Array{T,1},1},17)::Array{T,1})::Array{Int64,1} # /home/etotoni/.julia/v0.4/HPAT/examples/queries_devel/tests/test_q26.jl, line 33:
"""
function translate_aggregate(aggregate_node,state)
    @dprintln(3,"translating aggregate: ",aggregate_node)
    out_arr = toLHSVar(aggregate_node.args[1])
    # convert _agg_out_t2_in_t1 to t2, t1
    out_names = string(out_arr)[10:end]
    in_c = search(out_names,"in").start
    t1 = Symbol(out_names[in_c+3:end])
    t2 = Symbol(out_names[1:in_c-2])
    
    t1_cols = state.tableCols[t1]
    t2_cols = state.tableCols[t2]
    t1_num_cols = length(t1_cols)
    t2_num_cols = length(t2_cols)
    
    # one typeof() call for each output column except key
    remove_before = t2_num_cols-1
    # extra assignments
    remove_after =  t2_num_cols
    # TODO: simplify args
    new_aggregate_node = Expr(:aggregate, t2, t1, aggregate_node.args[2].args[2:3])
    return remove_before, remove_after, [new_aggregate_node]
end

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
        HPAT.enableOMP()
        # no change
        return [Expr(:(=),lhs,rhs)]
    else # not handled yet
        return [Expr(:(=),lhs,rhs)]
    end
end

function translate_hpat_dist_calls(lhs::ANY, rhs::ANY, hpat_call::Symbol, state)
    return Any[]
end

function getHPATcall(call::Expr)
    if call.head==:call
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
    state.data_source_counter += 1
    dsrc_num = state.data_source_counter
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
        size_i = symbol("__hpat_h5_dim_size_"*string(dsrc_num)*"_"*string(i))
        CompilerTools.LambdaHandling.addLocalVariable(size_i, Int64, ISASSIGNEDONCE | ISASSIGNED, state.linfo)
        # size_i = addTempVariable(Int64, state.linfo)
        size_i_call = mk_call(GlobalRef(HPAT.API,:__hpat_get_H5_dim_size), [arr_size_var, i])
        push!(res, TypedExpr(Int64, :(=), size_i, size_i_call))
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

function translate_data_source_TXT(lhs::LHSVar, rhs::Expr, state)
    dprintln(3,"TXT data source found ", rhs)
    res = Any[]
    txt_file = rhs.args[3]
    # update counter and get data source number
    state.data_source_counter += 1
    dsrc_num = state.data_source_counter
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
        size_i = symbol("__hpat_txt_dim_size_"*string(dsrc_num)*"_"*string(i))
        CompilerTools.LambdaHandling.addLocalVariable(size_i, Int64, ISASSIGNEDONCE | ISASSIGNED, state.linfo)
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

function AstWalkCallback(node::Expr,dw)

    if node.head==:filter
        cond_arr = node.args[1]
        node.args[1] = AstWalker.AstWalk(cond_arr, ParallelAccelerator.DomainIR.AstWalkCallback, dw)
        t = node.args[2]
        cols = node.args[3]
        col_arrs = node.args[4]
        for i in 1:length(col_arrs[i])
            old_arr = col_arrs[i]
            col_arrs[i] = AstWalker.AstWalk(col_arrs[i], ParallelAccelerator.DomainIR.AstWalkCallback, dw)
            @assert col_arrs[i]==old_arr "Table column name shouldn't change for now"
        end
        return node
    elseif node.head==:join
    elseif node.head==:aggregate
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function live_cb(node::Expr)

    if node.head==:filter
        cond_arr = node.args[1]
        t = node.args[2]
        cols = node.args[3]
        col_arrs = node.args[4]
        @println(3,"DomainPass filter CB ",node)
        return [cond_arr;col_arrs]
    elseif node.head==:join
    elseif node.head==:aggregate
    end
    return nothing
end


end # module

