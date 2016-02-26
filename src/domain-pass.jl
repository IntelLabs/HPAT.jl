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

using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools.LambdaHandling
using CompilerTools.Helper

mk_alloc(typ, s) = Expr(:alloc, typ, s)
mk_call(fun,args) = Expr(:call, fun, args...)

# ENTRY to distributedIR
function from_root(function_name, ast :: Expr)
    @assert ast.head == :lambda "Input to DomainPass should be :lambda Expr"
    @dprintln(1,"Starting main DomainPass.from_root.  function = ", function_name, " ast = ", ast)

    linfo = CompilerTools.LambdaHandling.lambdaExprToLambdaVarInfo(ast)
    state::DomainState = DomainState(linfo, 0)
    
    # transform body
    @assert ast.args[3].head==:body "DomainPass: invalid lambda input"
    body = TypedExpr(ast.args[3].typ, :body, from_toplevel_body(ast.args[3].args, state)...)
    new_ast = CompilerTools.LambdaHandling.LambdaVarInfoToLambdaExpr(state.linfo, body)
    @dprintln(1,"DomainPass.from_root returns function = ", function_name, " ast = ", new_ast)
    # ast = from_expr(ast)
    return new_ast
end

# information about AST gathered and used in DomainPass
type DomainState
    linfo  :: LambdaVarInfo
    data_source_counter::Int64 # a unique counter for data sources in program 
end


# nodes are :body of AST
function from_toplevel_body(nodes::Array{Any,1}, state::DomainState)
    res::Array{Any,1} = []
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

# :(=) assignment (:(=), lhs, rhs)
function from_assignment(node::Expr, state)
    
    # pattern match distributed calls that need domain translation
    matched::Array{Any,1} = pattern_match_hps_dist_calls(node.args[1], node.args[2], state)
    # matched is an expression, :not_matched head is used if not matched 
    if length(matched)!=0
        return matched
    else
        return [node]
    end
end

function pattern_match_hps_dist_calls(lhs::SymGen, rhs::Expr, state)
    # example of data source call: 
    # :((top(typeassert))((top(convert))(Array{Float64,1},(ParallelAccelerator.API.__hps_data_source_HDF5)("/labels","./test.hdf5")),Array{Float64,1})::Array{Float64,1})
    if rhs.head==:call && length(rhs.args)>=2 && isCall(rhs.args[2])
        in_call = rhs.args[2]
        if length(in_call.args)>=3 && isCall(in_call.args[3]) 
            inner_call = in_call.args[3]
            if isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_data_source_HDF5
                res = Any[]
                dprintln(3,"data source found ", inner_call)
                hdf5_var = inner_call.args[2]
                hdf5_file = inner_call.args[3]
                # update counter and get data source number
                state.data_source_counter += 1
                dsrc_num = state.data_source_counter
                dsrc_id_var = addGenSym(Int64, state.linfo)
                push!(res, TypedExpr(Int64, :(=), dsrc_id_var, dsrc_num))
                # get array type
                arr_typ = getType(lhs, state.linfo)
                dims = ndims(arr_typ)
                elem_typ = eltype(arr_typ)
                # generate open call
                # lhs is dummy argument so ParallelIR wouldn't reorder
                open_call = mk_call(:__hps_data_source_HDF5_open, [dsrc_id_var, hdf5_var, hdf5_file, lhs])
                push!(res, open_call)
                # generate array size call
                # arr_size_var = addGenSym(Tuple, state.linfo)
                # assume 1D for now
                arr_size_var = addGenSym(ParallelAccelerator.H5SizeArr_t, state.linfo)
                size_call = mk_call(:__hps_data_source_HDF5_size, [dsrc_id_var, lhs])
                push!(res, TypedExpr(arr_size_var, :(=), arr_size_var, size_call))
                # generate array allocation
                size_expr = Any[]
                for i in dims:-1:1
                    size_i = addGenSym(Int64, state.linfo)
                    size_i_call = mk_call(:__hps_get_H5_dim_size, [arr_size_var, i])
                    push!(res, TypedExpr(Int64, :(=), size_i, size_i_call))
                    push!(size_expr, size_i)
                end
                arrdef = TypedExpr(arr_typ, :alloc, elem_typ, size_expr)
                push!(res, TypedExpr(arr_typ, :(=), lhs, arrdef))
                # generate read call
                read_call = mk_call(:__hps_data_source_HDF5_read, [dsrc_id_var, lhs])
                push!(res, read_call)
                return res
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_data_source_TXT
                dprintln(3,"data source found ", inner_call)
                res = Any[]
                txt_file = inner_call.args[2]
                # update counter and get data source number
                state.data_source_counter += 1
                dsrc_num = state.data_source_counter
                dsrc_id_var = addGenSym(Int64, state.linfo)
                push!(res, TypedExpr(Int64, :(=), dsrc_id_var, dsrc_num))
                # get array type
                arr_typ = getType(lhs, state.linfo)
                dims = ndims(arr_typ)
                elem_typ = eltype(arr_typ)
                # generate open call
                # lhs is dummy argument so ParallelIR wouldn't reorder
                open_call = mk_call(:__hps_data_source_TXT_open, [dsrc_id_var, txt_file, lhs])
                push!(res, open_call)
                # generate array size call
                # arr_size_var = addGenSym(Tuple, state.linfo)
                arr_size_var = addGenSym(ParallelAccelerator.SizeArr_t, state.linfo)
                size_call = mk_call(:__hps_data_source_TXT_size, [dsrc_id_var, lhs])
                push!(res, TypedExpr(arr_size_var, :(=), arr_size_var, size_call))
                # generate array allocation
                size_expr = Any[]
                for i in dims:-1:1
                    size_i = addGenSym(Int64, state.linfo)
                    size_i_call = mk_call(:__hps_get_TXT_dim_size, [arr_size_var, i])
                    push!(res, TypedExpr(Int64, :(=), size_i, size_i_call))
                    push!(size_expr, size_i)
                end
                arrdef = TypedExpr(arr_typ, :alloc, elem_typ, size_expr)
                push!(res, TypedExpr(arr_typ, :(=), lhs, arrdef))
                # generate read call
                read_call = mk_call(:__hps_data_source_TXT_read, [dsrc_id_var, lhs])
                push!(res, read_call)
                return res
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_kmeans
                dprintln(3,"kmeans found ", inner_call)
                lib_call = mk_call(:__hps_kmeans, [lhs,inner_call.args[2], inner_call.args[3]])
                return [lib_call]
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_LinearRegression
                dprintln(3,"LinearRegression found ", inner_call)
                lib_call = mk_call(:__hps_LinearRegression, [lhs,inner_call.args[2], inner_call.args[3]])
                return [lib_call] 
            elseif isa(inner_call.args[1],GlobalRef) && inner_call.args[1].name==:__hps_NaiveBayes
                dprintln(3,"NaiveBayes found ", inner_call)
                lib_call = mk_call(:__hps_NaiveBayes, [lhs,inner_call.args[2], inner_call.args[3], inner_call.args[4]])
                return [lib_call]
            end
        end
    end
    
    return Any[]
end

function pattern_match_hps_dist_calls(lhs::Any, rhs::Any, state)
    return Any[]
end


end # module

