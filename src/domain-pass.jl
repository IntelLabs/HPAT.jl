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

import HPAT

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
                        :__hpat_Kmeans,
                        :__hpat_LinearRegression,
                        :__hpat_NaiveBayes]

# ENTRY to DomainPass
function from_root(function_name, ast)
    @dprintln(1,"Starting main DomainPass.from_root.  function = ", function_name, " ast = ", ast)

    linfo, body = CompilerTools.LambdaHandling.lambdaToLambdaVarInfo(ast)
    state::DomainState = DomainState(linfo, 0)
    
    # transform body
    body.args = from_toplevel_body(body.args, state)
    @dprintln(1,"DomainPass.from_root returns function = ", function_name, " body = ", body)
    println("DomainPass.from_root returns function = ", function_name, " body = ", body)
    return LambdaVarInfoToLambda(state.linfo, body.args)
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
        return Expr(:(=),lhs,rhs)
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

end # module

