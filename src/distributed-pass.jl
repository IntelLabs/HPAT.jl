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

module DistributedPass

#using Debug

import Base.show

using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools.AstWalker
import CompilerTools.ReadWriteSet
using CompilerTools.LambdaHandling
using CompilerTools.Helper

import HPAT
using HPAT.Partitioning
using HPAT.SEQ
using HPAT.TWO_D
using HPAT.ONE_D
using HPAT.ONE_D_VAR

using ParallelAccelerator
import ParallelAccelerator.ParallelIR
import ParallelAccelerator.ParallelIR.isArrayType
import ParallelAccelerator.ParallelIR.getParforNode
import ParallelAccelerator.ParallelIR.isBareParfor
import ParallelAccelerator.ParallelIR.isAllocation
import ParallelAccelerator.ParallelIR.TypedExpr
import ParallelAccelerator.ParallelIR.get_alloc_shape
import ParallelAccelerator.ParallelIR.computeLiveness

import ParallelAccelerator.ParallelIR.ISCAPTURED
import ParallelAccelerator.ParallelIR.ISASSIGNED
import ParallelAccelerator.ParallelIR.ISASSIGNEDBYINNERFUNCTION
import ParallelAccelerator.ParallelIR.ISCONST
import ParallelAccelerator.ParallelIR.ISASSIGNEDONCE
import ParallelAccelerator.ParallelIR.ISPRIVATEPARFORLOOP
import ParallelAccelerator.ParallelIR.PIRReduction

mk_call(fun,args) = Expr(:call, fun, args...)

function mk_mult_int_expr(args::Array)
    if length(args)==0
        return 1
    elseif length(args)==1
        return args[1]
    end
    next = 2
    prev_expr = args[1]

    while next<=length(args)
        m_call = mk_call(GlobalRef(Base,:mul_int),[prev_expr,args[next]])
        prev_expr  = mk_call(GlobalRef(Base,:box),[Int64,m_call])
        next += 1
    end
    return prev_expr
end

mk_add_int_expr(a,b) = mk_call(GlobalRef(Base,:box),[Int64, mk_call(GlobalRef(Base,:add_int),[a,b])])
mk_sub_int_expr(a,b) = mk_call(GlobalRef(Base,:box),[Int64, mk_call(GlobalRef(Base,:sub_int),[a,b])])
mk_div_int_expr(a,b) = mk_call(GlobalRef(Base,:box),[Int64, mk_call(GlobalRef(Base,:sdiv_int),[a,b])])

dist_ir_funcs = Set([   :unsafe_arrayref,
                        :unsafe_arrayset,
                        :__hpat_data_source_HDF5_open,
                        :__hpat_data_sink_HDF5_open,
                        :__hpat_data_source_HDF5_size,
                        :__hpat_get_H5_dim_size,
                        :__hpat_data_source_HDF5_read,
                        :__hpat_data_sink_HDF5_write,
                        :__hpat_data_source_TXT_open,
                        :__hpat_data_source_TXT_size,
                        :__hpat_get_TXT_dim_size,
                        :__hpat_data_source_TXT_read,
                        :__hpat_filter,
                        :__hpat_join,
                        :__hpat_aggregate,
                        :Kmeans,
                        :LinearRegression,
                        :NaiveBayes,
                        :arraylen, :arraysize, :reshape, :tuple, :hcat, :transpose!,
                        :gemm_wrapper!,
                        :gemv!])

# ENTRY to distributedIR
function from_root(function_name, ast::Tuple)
    @dprintln(1,"Starting main DistributedPass.from_root.  function = ", function_name, " ast = ", ast)

    (linfo, body) = ast
    lives = computeLiveness(body, linfo)
    user_partitionings = get_user_partitionings(body)
    state::DistPassState = initDistState(linfo,lives,user_partitionings)

    # find if an array should be partitioned, sequential, or shared
    getArrayDistributionInfo(body, state)

    # transform body
    body.args = from_toplevel_body(body.args, state)
    @dprintln(1,"DistributedPass.from_root returns function = ", function_name, " ast = ", body)
    return state.LambdaVarInfo, body
end

type ArrDistInfo
    partitioning::Partitioning      # partitioning of array (SEQ, ONE_D_VAR, TWO_D, ONE_D)
    dim_sizes::Array{Union{RHSVar,Int,Expr},1}      # sizes of array dimensions
    # assuming only last dimension is partitioned
    arr_id::Int # assign ID to distributed array to access partitioning info later

    # array partitioning information similar to HDF5 hyperslab interface
    # for 1D, only start and count of last dimension are set
    # for 2D, these are set to mirror ScaLAPACK's block cyclic distribution
    starts::Array{LHSVar}
    counts::Array{LHSVar}
    strides::Array{LHSVar}
    blocks::Array{LHSVar}
    local_sizes::Array{LHSVar}
    leftovers::Array{LHSVar}

    function ArrDistInfo(num_dims::Int)
        # one dimensional partitioning is default
        new(ONE_D, zeros(Int64,num_dims),0,
        Array(LHSVar,num_dims), Array(LHSVar,num_dims),
        Array(LHSVar,num_dims), Array(LHSVar,num_dims),
        Array(LHSVar,num_dims),Array(LHSVar,num_dims))
    end
end

function show(io::IO, pnode::HPAT.DistributedPass.ArrDistInfo)
    print(io,"partitioning: ",pnode.partitioning," sizes: ", pnode.dim_sizes)
end

# information about AST gathered and used in DistributedPass
type DistPassState
    # information about all arrays
    arrs_dist_info::Dict{LHSVar,ArrDistInfo}
    parfor_partitioning::Dict{Int,Partitioning}
    parfor_arrays::Dict{Int,Vector{LHSVar}}
    LambdaVarInfo::LambdaVarInfo
    uniqueId::Int
    lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
    # keep values for constant tuples. They are often used for allocating and reshaping arrays.
    tuple_table              :: Dict{LHSVar,Array{Union{LHSVar,Int},1}}
    max_label :: Int # holds the max number of all LabelNodes
    user_partitionings::Dict{Symbol,Partitioning}

    function DistPassState(linfo, lives, user_partitionings)
        new(Dict{LHSVar, Array{ArrDistInfo,1}}(), Dict{Int,Partitioning}(), Dict{Int,Vector{LHSVar}}(), linfo,0, lives,
             Dict{LHSVar,Array{Union{LHSVar,Int},1}}(),0,user_partitionings)
    end
end

isSEQ(arr,state) = (state.arrs_dist_info[arr].partitioning==SEQ)
isONE_D_VAR(arr,state) = (state.arrs_dist_info[arr].partitioning==ONE_D_VAR)
isONE_D(arr,state) = (state.arrs_dist_info[arr].partitioning==ONE_D)
isTWO_D(arr,state) = (state.arrs_dist_info[arr].partitioning==TWO_D)

function setSEQ(arr,state)
  state.arrs_dist_info[arr].partitioning=SEQ
end

function setTWO_D(arr,state)
  state.arrs_dist_info[arr].partitioning=TWO_D
end

getArrayPartitioning(arr,state) = state.arrs_dist_info[arr].partitioning

function setArrayPartitioning(arr,part,state)
  state.arrs_dist_info[arr].partitioning=part
end

include("distributed-pass-ast-walk.jl")

function show(io::IO, pnode::HPAT.DistributedPass.DistPassState)
    println(io,"DistPassState arrs_dist_info:")
    for i in pnode.arrs_dist_info
        println(io,"  ", i)
    end
    println(io,"DistPassState parfor_partitioning: ",pnode.parfor_partitioning)
    #= println(io,"DistPassState parfor_info:")
    for i in pnode.parfor_info
        println(io,"  ", i)
    end
    println(io,"DistPassState seq_parfors:")
    for i in pnode.seq_parfors
        print(io," ", i)
    end=#
    println(io,"")
end

function get_user_partitionings(body)
    first_arg = body.args[1]
    if isa(first_arg, Expr) && first_arg.head==:meta
        for meta in first_arg.args
            if meta.head==:hpat_partitioning
                @dprintln(3, "hpat partitionings found: ", meta)
                return meta.args[1]
            end
        end
    end
    return Dict{Symbol,Symbol}()
end

function initDistState(linfo::LambdaVarInfo, lives, user_partitionings)
    state = DistPassState(linfo, lives, user_partitionings)

    vars = getLocalVariables(linfo)
    # Populate the symbol table
    for var in vars
        typ = CompilerTools.LambdaHandling.getType(var, linfo)
        if isArrayType(typ)
            arrInfo = ArrDistInfo(ndims(typ))
            state.arrs_dist_info[var] = arrInfo
        end
    end

    return state
end


# nodes are :body of AST
function from_toplevel_body(nodes::Array{Any,1}, state::DistPassState)
    state.max_label = ParallelIR.getMaxLabel(state.max_label, nodes)
    res::Array{Any,1} = genDistributedInit(state)
    for node in nodes
        new_exprs = from_expr(node, state)
        append!(res, new_exprs)
    end
    return res
end

# nodes are :body of Parfor
function from_nested_body(nodes::Array{Any,1}, state::DistPassState)
    res = Any[]
    for node in nodes
        new_exprs = from_expr(node, state)
        append!(res, new_exprs)
    end
    return res
end

function from_expr(node::Expr, state::DistPassState)
    head = node.head
    if head==:(=)
        return from_assignment(node, state, toLHSVar(node.args[1]), node.args[2])
    elseif head==:parfor
        return from_parfor(node, state)
    #elseif head==:block
    elseif head==:call
        return from_call(node, state)
    else
        return [node]
    end
end


function from_expr(node::Any, state::DistPassState)
    return [node]
end

# generates initialization code for distributed execution
function genDistributedInit(state::DistPassState)
    initCall = Expr(:call, GlobalRef(HPAT.API,:hpat_dist_init))
    numPesCall = Expr(:call, GlobalRef(HPAT.API,:hpat_dist_num_pes))
    nodeIdCall = Expr(:call, GlobalRef(HPAT.API,:hpat_dist_node_id))

    CompilerTools.LambdaHandling.addLocalVariable(symbol("__hpat_num_pes"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    CompilerTools.LambdaHandling.addLocalVariable(symbol("__hpat_node_id"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

    num_pes_assign = Expr(:(=), :__hpat_num_pes, numPesCall)
    node_id_assign = Expr(:(=), :__hpat_node_id, nodeIdCall)
    res = Any[initCall; num_pes_assign; node_id_assign]
    if haskey(ENV, "ENABLE_GAAS")
        # Just to make things working for now
        # Add variables like above so that it does not break in future
        initCallGAAS = Expr(:call, GlobalRef(HPAT.API,:hpat_dist_init_gaas))
        append!(res,[initCallGAAS])
    end

    # generate 2D init if there is any 2D array or parfor
    if any([state.arrs_dist_info[arr].partitioning==TWO_D for arr in keys(state.arrs_dist_info)]) ||
        any([state.parfor_partitioning[parfor_id]==TWO_D for parfor_id in keys(state.parfor_partitioning)])
      @dprintln(3,"DistPass generating 2D init")
      extra_2D_includes = """#include <mkl_blacs.h>
                            #include <mkl_scalapack.h>
                            #include <mkl_pblas.h>
                            extern "C" {
                            void descinit_(MKL_INT*,MKL_INT*,MKL_INT*,MKL_INT*,MKL_INT*,MKL_INT*,MKL_INT*,MKL_INT*,MKL_INT*,MKL_INT*);
                            MKL_INT numroc_(const MKL_INT*,const MKL_INT*,const MKL_INT*,const MKL_INT*,const MKL_INT*);
                            }
                            """
      HPAT.addHpatInclude(extra_2D_includes,"","")
      initCall2d = Expr(:call, GlobalRef(HPAT.API,:hpat_dist_2d_init))
      CompilerTools.LambdaHandling.addLocalVariable(symbol("__hpat_num_pes_x"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
      CompilerTools.LambdaHandling.addLocalVariable(symbol("__hpat_num_pes_y"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
      CompilerTools.LambdaHandling.addLocalVariable(symbol("__hpat_node_id_x"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
      CompilerTools.LambdaHandling.addLocalVariable(symbol("__hpat_node_id_y"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

      res2 = Any[initCall2d]
      res = [res;res2]
    end

    return res
end

function from_assignment(node::Expr, state::DistPassState, lhs::LHSVar, rhs::Expr)
    @assert node.head==:(=) "DistributedPass invalid assignment head"
    if isAllocation(rhs)
      return from_assignment_alloc(node,state,lhs,rhs)
    elseif rhs.head==:call && isBaseFunc(rhs.args[1],:reshape)
      return from_assignment_reshape(node,state,lhs,rhs)
    elseif rhs.head==:call && isBaseFunc(rhs.args[1],:gemm_wrapper!)
        return from_assignment_gemm(node,state,lhs,rhs)
    elseif rhs.head==:call && isBaseFunc(rhs.args[1],:gemv!)
        return from_assignment_gemv(node,state,lhs,rhs)
    else
        node.args[2] = from_expr(rhs,state)[1]
    end
    return [node]
end

function from_assignment(node::Expr, state::DistPassState, lhs::LHSVar, rhs::ANY)
  node.args[2] = from_expr(rhs,state)[1]
  return [node]
end

function from_assignment_alloc(node::Expr, state::DistPassState, arr::LHSVar, rhs::Expr)
  @dprintln(3,"from assingment alloc: ", node)
  if isONE_D(arr,state)
      @dprintln(3,"DistPass allocation array: ", arr)
      #shape = get_alloc_shape(node.args[2].args[2:end])
      #old_size = shape[end]
      dim_sizes = state.arrs_dist_info[arr].dim_sizes
      # generate array division
      # simple 1D partitioning of last dimension, more general partitioning needed
      # match common big data matrix reperesentation
      arr_tot_size = dim_sizes[end]

      # Not sure if need this in future
      # if arr_tot_size == -1
      #     return [node]
      # end

      arr_id = getDistNewID(state)
      state.arrs_dist_info[arr].arr_id = arr_id
      darr_start_var = symbol("__hpat_dist_arr_start_"*string(arr_id))
      darr_div_var = symbol("__hpat_dist_arr_div_"*string(arr_id))
      darr_count_var = symbol("__hpat_dist_arr_count_"*string(arr_id))
      state.arrs_dist_info[arr].starts[end] = darr_start_var
      state.arrs_dist_info[arr].counts[end] = darr_count_var

      CompilerTools.LambdaHandling.addLocalVariable(darr_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
      CompilerTools.LambdaHandling.addLocalVariable(darr_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
      CompilerTools.LambdaHandling.addLocalVariable(darr_count_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)


      darr_div_expr = Expr(:(=),darr_div_var, mk_div_int_expr(arr_tot_size,:__hpat_num_pes))
      # zero-based index to match C interface of HDF5
      darr_start_expr = Expr(:(=), darr_start_var, mk_mult_int_expr([:__hpat_node_id,darr_div_var]))
      # darr_count_expr = :($darr_count_var = __hpat_node_id==__hpat_num_pes-1 ? $arr_tot_size-__hpat_node_id*$darr_div_var : $darr_div_var)
      darr_count_expr = Expr(:(=), darr_count_var, mk_call(GlobalRef(HPAT.API,:__hpat_get_node_portion),[arr_tot_size, darr_div_var, :__hpat_num_pes, :__hpat_node_id]))

      # set new divided allocation size
      rhs.args[end-1] = darr_count_var

      res = [darr_div_expr; darr_start_expr; darr_count_expr; node]
      #debug_size_print = :(println("size ",$darr_count_var))
      #push!(res,debug_size_print)
      return res
  elseif isTWO_D(arr,state)
    return from_assignment_alloc_2d(node, state, arr, rhs)
  end
  return [node]
end

""" 2D block cyclic distribution to match ScaLAPACK's interface.
Keep start, stride, and count to match HDF5's interface.
Assign extra blocks and leftover rows as well.
"""
function from_assignment_alloc_2d(node::Expr, state::DistPassState, arr::LHSVar, rhs::Expr)
  arr_id = getDistNewID(state)
  state.arrs_dist_info[arr].arr_id = arr_id
  dim_sizes = state.arrs_dist_info[arr].dim_sizes
  arr_tot_size_x = dim_sizes[end-1]
  arr_tot_size_y = dim_sizes[end]

  # constant block size for block-cyclic partitioning
  BLOCK_SIZE = HPAT.BLOCK_SIZE
  block_size_var = symbol("__hpat_dist_arr_2d_block_size_"*string(arr_id))
  CompilerTools.LambdaHandling.addLocalVariable(block_size_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  # same block size for both dimensions
  # block size should be less than smaller dimension
  # block_size_expr = Expr(:(=), block_size_var, mk_call(GlobalRef(HPAT.API,:__hpat_min),[:($BLOCK_SIZE), dim_sizes[end], dim_sizes[end-1]])
  block_size_expr = Expr(:(=), block_size_var, :($BLOCK_SIZE))
  state.arrs_dist_info[arr].blocks[end-1] = state.arrs_dist_info[arr].blocks[end] = block_size_var

  # start = id*block_size
  start_x_var = symbol("__hpat_dist_arr_2d_start_x_"*string(arr_id))
  start_y_var = symbol("__hpat_dist_arr_2d_start_y_"*string(arr_id))
  CompilerTools.LambdaHandling.addLocalVariable(start_x_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(start_y_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  start_x_expr = Expr(:(=), start_x_var, mk_mult_int_expr([:__hpat_node_id_x,block_size_var]))
  start_y_expr = Expr(:(=), start_y_var, mk_mult_int_expr([:__hpat_node_id_y,block_size_var]))
  state.arrs_dist_info[arr].starts[end-1] = start_x_var
  state.arrs_dist_info[arr].starts[end] = start_y_var

  # stride = num_pes*block_size
  stride_x_var = symbol("__hpat_dist_arr_2d_stride_x_"*string(arr_id))
  stride_y_var = symbol("__hpat_dist_arr_2d_stride_y_"*string(arr_id))
  CompilerTools.LambdaHandling.addLocalVariable(stride_x_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(stride_y_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  stride_x_expr = Expr(:(=), stride_x_var, mk_mult_int_expr([:__hpat_num_pes_x,block_size_var]))
  stride_y_expr = Expr(:(=), stride_y_var, mk_mult_int_expr([:__hpat_num_pes_y,block_size_var]))
  state.arrs_dist_info[arr].strides[end-1] = stride_x_var
  state.arrs_dist_info[arr].strides[end] = stride_y_var

  # calculate number of blocks in each dimension (excluding leftover rows/columns)
  # total_size/block_size
  num_blocks_x_var = symbol("__hpat_dist_arr_2d_num_blocks_x_"*string(arr_id))
  num_blocks_y_var = symbol("__hpat_dist_arr_2d_num_blocks_y_"*string(arr_id))
  CompilerTools.LambdaHandling.addLocalVariable(num_blocks_x_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(num_blocks_y_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  nb_x_div_expr = Expr(:(=),num_blocks_x_var, mk_div_int_expr(arr_tot_size_x, block_size_var))
  nb_y_div_expr = Expr(:(=),num_blocks_y_var, mk_div_int_expr(arr_tot_size_y, block_size_var))

  # number of blocks per local PE in each dimension
  # num_blocks/num_pes + possible extra block
  blocks_per_pe_x_var = symbol("__hpat_dist_arr_2d_blocks_per_pe_x_"*string(arr_id))
  blocks_per_pe_y_var = symbol("__hpat_dist_arr_2d_blocks_per_pe_y_"*string(arr_id))
  CompilerTools.LambdaHandling.addLocalVariable(blocks_per_pe_x_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(blocks_per_pe_y_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  bppx_div_expr = Expr(:(=),blocks_per_pe_x_var, mk_div_int_expr(num_blocks_x_var,:__hpat_num_pes_x))
  bppy_div_expr = Expr(:(=),blocks_per_pe_y_var, mk_div_int_expr(num_blocks_y_var,:__hpat_num_pes_y))
  extra_block_call_x = mk_call(GlobalRef(HPAT.API,:__hpat_add_extra_block),[blocks_per_pe_x_var, num_blocks_x_var,:__hpat_node_id_x,:__hpat_num_pes_x])
  extra_block_call_y = mk_call(GlobalRef(HPAT.API,:__hpat_add_extra_block),[blocks_per_pe_y_var, num_blocks_y_var,:__hpat_node_id_y,:__hpat_num_pes_y])
  state.arrs_dist_info[arr].counts[end-1] = blocks_per_pe_x_var
  state.arrs_dist_info[arr].counts[end] = blocks_per_pe_y_var

  leftovers_x_var = symbol("__hpat_dist_arr_2d_leftovers_x_"*string(arr_id))
  leftovers_y_var = symbol("__hpat_dist_arr_2d_leftovers_y_"*string(arr_id))
  CompilerTools.LambdaHandling.addLocalVariable(leftovers_x_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(leftovers_y_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  leftovers_x_expr = Expr(:(=), leftovers_x_var, mk_call(GlobalRef(HPAT.API,:__hpat_get_leftovers),
                                  [num_blocks_x_var,:__hpat_node_id_x,:__hpat_num_pes_x,arr_tot_size_x,block_size_var]))
  leftovers_y_expr = Expr(:(=), leftovers_y_var, mk_call(GlobalRef(HPAT.API,:__hpat_get_leftovers),
                                  [num_blocks_y_var,:__hpat_node_id_y,:__hpat_num_pes_y,arr_tot_size_y,block_size_var]))
  state.arrs_dist_info[arr].leftovers[end-1] = leftovers_x_var
  state.arrs_dist_info[arr].leftovers[end] = leftovers_y_var


  # local sizes
  loc_size_x_var = symbol("__hpat_dist_arr_2d_loc_size_x_"*string(arr_id))
  loc_size_y_var = symbol("__hpat_dist_arr_2d_loc_size_y_"*string(arr_id))
  CompilerTools.LambdaHandling.addLocalVariable(loc_size_x_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(loc_size_y_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  loc_size_x_expr = Expr(:(=), loc_size_x_var, mk_add_int_expr(mk_mult_int_expr([blocks_per_pe_x_var,block_size_var]),leftovers_x_var))
  loc_size_y_expr = Expr(:(=), loc_size_y_var, mk_add_int_expr(mk_mult_int_expr([blocks_per_pe_y_var,block_size_var]),leftovers_y_var))
  state.arrs_dist_info[arr].local_sizes[end-1] = loc_size_x_var
  state.arrs_dist_info[arr].local_sizes[end] = loc_size_y_var

  # set new divided allocation size
  rhs.args[end-1] = loc_size_y_var
  rhs.args[end-3] = loc_size_x_var

  # TODO: handle extra blocks and partial block
  res = [block_size_expr; start_x_expr; start_y_expr; stride_x_expr; stride_y_expr;
            nb_x_div_expr; nb_y_div_expr; bppx_div_expr; bppy_div_expr;
            extra_block_call_x; extra_block_call_y;
            leftovers_x_expr; leftovers_y_expr;
            loc_size_x_expr; loc_size_y_expr; node]
  #debug_size_print = :(println("size ",$darr_count_var))
  #push!(res,debug_size_print)
  return res
end


function from_assignment_reshape(node::Expr, state::DistPassState, arr::LHSVar, rhs::Expr)
  if isONE_D(arr,state)
      @dprintln(3,"DistPass reshape array: ", arr)
      dim_sizes = state.arrs_dist_info[arr].dim_sizes
      # generate array division
      # simple 1D partitioning of last dimension, more general partitioning needed
      # match common big data matrix reperesentation
      arr_tot_size = dim_sizes[end]

      arr_id = getDistNewID(state)
      state.arrs_dist_info[arr].arr_id = arr_id
      darr_start_var = symbol("__hpat_dist_arr_start_"*string(arr_id))
      darr_div_var = symbol("__hpat_dist_arr_div_"*string(arr_id))
      darr_count_var = symbol("__hpat_dist_arr_count_"*string(arr_id))
      state.arrs_dist_info[arr].starts[end] = darr_start_var
      state.arrs_dist_info[arr].counts[end] = darr_count_var

      CompilerTools.LambdaHandling.addLocalVariable(darr_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
      CompilerTools.LambdaHandling.addLocalVariable(darr_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
      CompilerTools.LambdaHandling.addLocalVariable(darr_count_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)


      darr_div_expr = Expr(:(=), darr_div_var, mk_div_int_expr(arr_tot_size,:__hpat_num_pes))
      # zero-based index to match C interface of HDF5
      darr_start_expr = Expr(:(=),darr_start_var, mk_mult_int_expr([:__hpat_node_id, darr_div_var]))
      #darr_count_expr = :($darr_count_var = __hpat_node_id==__hpat_num_pes-1 ? $arr_tot_size-__hpat_node_id*$darr_div_var : $darr_div_var)
      darr_count_expr = Expr(:(=), darr_count_var, mk_call(GlobalRef(HPAT.API,:__hpat_get_node_portion),[arr_tot_size, darr_div_var, :__hpat_num_pes, :__hpat_node_id]))

      # create a new tuple for reshape
      tup_call = Expr(:call, TopNode(:tuple), dim_sizes[1:end-1]... , darr_count_var)
      reshape_tup_var = symbol("__hpat_dist_tup_var_"*string(arr_id))
      tup_typ = CompilerTools.LambdaHandling.getType(rhs.args[3], state.LambdaVarInfo)
      CompilerTools.LambdaHandling.addLocalVariable(reshape_tup_var, tup_typ, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
      tup_expr = Expr(:(=),reshape_tup_var,tup_call)
      rhs.args[3] = reshape_tup_var
      res = [darr_div_expr; darr_start_expr; darr_count_expr; tup_expr; node]
      #debug_size_print = :(println("size ",$darr_count_var))
      #push!(res,debug_size_print)
      return res
  end
  return [node]
end

function from_assignment_gemm(node::Expr, state::DistPassState, lhs::LHSVar, rhs::Expr)
  arr1 = toLHSVar(rhs.args[5])
  t1 = (rhs.args[3]=='T')
  arr2 = toLHSVar(rhs.args[6])
  t2 = (rhs.args[4]=='T')

  # result is sequential but with reduction if both inputs are partitioned and second one is transposed
  # e.g. labels*points'
  if !isSEQ(arr1,state) && !isSEQ(arr2,state) && t2 && !t1 && isSEQ(lhs,state)
    @dprintln(3,"DistPass translating gemm reduce: ", node)
    # rhs.args[1] = :__hpat_gemm_reduce
    # allocate temporary array for local gemm values
    alloc_args = Array(Any,2)
    out_typ = CompilerTools.LambdaHandling.getType(lhs, state.LambdaVarInfo)
    alloc_args[1] = eltype(out_typ)
    out_dim_sizes = state.arrs_dist_info[lhs].dim_sizes
    alloc_args[2] = out_dim_sizes
    alloc_call = ParallelIR.from_alloc(alloc_args)
    reduce_num = getDistNewID(state)
    reduce_var = symbol("__hpat_gemm_reduce_"*string(reduce_num))
    CompilerTools.LambdaHandling.addLocalVariable(reduce_var, out_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    reduce_var_init = Expr(:(=), reduce_var, Expr(:call,alloc_call...))
    # TODO: deallocate temporary array
    # reduce_var_dealloc = Expr(:call, TopNode(:ccall), QuoteNode(:jl_dealloc_array), reduce_var)

    # get reduction size
    reduce_size_var = symbol("__hpat_gemm_reduce_size_"*string(reduce_num))
    CompilerTools.LambdaHandling.addLocalVariable(reduce_size_var, Int, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    size_expr = Expr(:(=), reduce_size_var, mk_mult_int_expr(out_dim_sizes))

    # add allreduce call
    allreduceCall = Expr(:call, GlobalRef(HPAT.API,:hpat_dist_allreduce), reduce_var, GlobalRef(Base,:add_float), rhs.args[2], reduce_size_var)
    res_copy = Expr(:(=), lhs, rhs.args[2])
    # replace gemm output with local var
    #node.args[1] = reduce_var
    rhs.args[2] = reduce_var

    return [reduce_var_init; node; size_expr; allreduceCall; res_copy]
    # first input is sequential but output is parallel if the second input is partitioned but not transposed
    # e.g. w*points
  elseif isSEQ(arr1,state) && !isSEQ(arr2,state) && !t2 && !isSEQ(lhs,state)
    @dprintln(3,"DistPass arr info gemm first input is sequential: ", arr1)
    #rhs.args[1] = :__hpat_gemm_broadcast
    @dprintln(3,"DistPass translating gemm broadcast: ", node)
    # otherwise, no known pattern found
  # if any array is 2D, then all arrays should be 2d
  elseif any([isTWO_D(arr1,state), isTWO_D(arr2,state), isTWO_D(lhs,state)])
    @assert all([isTWO_D(arr1,state), isTWO_D(arr2,state), isTWO_D(lhs,state)]) "invalid 2d gemm"
    @dprintln(3,"DistPass translating gemm 2d: ", node)
    rhs.args[1] = GlobalRef(HPAT.API,:__hpat_gemm_2d)

    # result in rhs has partitioning set in alloc, assign to lhs
    state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].dim_sizes
    state.arrs_dist_info[lhs].starts = state.arrs_dist_info[toLHSVar(rhs.args[2])].starts
    state.arrs_dist_info[lhs].counts = state.arrs_dist_info[toLHSVar(rhs.args[2])].counts
    state.arrs_dist_info[lhs].strides = state.arrs_dist_info[toLHSVar(rhs.args[2])].strides
    state.arrs_dist_info[lhs].blocks = state.arrs_dist_info[toLHSVar(rhs.args[2])].blocks
    state.arrs_dist_info[lhs].local_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].local_sizes
    state.arrs_dist_info[lhs].leftovers = state.arrs_dist_info[toLHSVar(rhs.args[2])].leftovers

    push!(rhs.args, state.arrs_dist_info[lhs].dim_sizes[end-1], state.arrs_dist_info[lhs].dim_sizes[end],
    state.arrs_dist_info[lhs].blocks[end-1], state.arrs_dist_info[lhs].blocks[end],
    state.arrs_dist_info[lhs].local_sizes[end-1], state.arrs_dist_info[lhs].local_sizes[end])

    push!(rhs.args, state.arrs_dist_info[arr1].dim_sizes[end-1], state.arrs_dist_info[arr1].dim_sizes[end],
    state.arrs_dist_info[arr1].blocks[end-1], state.arrs_dist_info[arr1].blocks[end],
    state.arrs_dist_info[arr1].local_sizes[end-1], state.arrs_dist_info[arr1].local_sizes[end])

    push!(rhs.args, state.arrs_dist_info[arr2].dim_sizes[end-1], state.arrs_dist_info[arr2].dim_sizes[end],
    state.arrs_dist_info[arr2].blocks[end-1], state.arrs_dist_info[arr2].blocks[end],
    state.arrs_dist_info[arr2].local_sizes[end-1], state.arrs_dist_info[arr2].local_sizes[end])
  end

  return [node]
end

function from_assignment_gemv(node::Expr, state::DistPassState, lhs::LHSVar, rhs::Expr)

  arr1 = toLHSVar(rhs.args[4])
  t1 = (rhs.args[3]=='T')
  arr2 = toLHSVar(rhs.args[5])

  # result is sequential but with reduction if both inputs are partitioned and matrix is not transposed (X*y)
  # result is sequential but with reduction if matrix partitioned (X'*y)
  if !isSEQ(arr1,state) && isSEQ(lhs,state) && !isSEQ(arr2,state) && !t1
    @dprintln(3,"DistPass translating gemv reduce: ", node)
    # rhs.args[1] = :__hpat_gemm_reduce
    # allocate temporary array for local gemm values
    alloc_args = Array(Any,2)
    out_typ = CompilerTools.LambdaHandling.getType(lhs, state.LambdaVarInfo)
    alloc_args[1] = eltype(out_typ)
    out_dim_sizes = state.arrs_dist_info[lhs].dim_sizes
    alloc_args[2] = out_dim_sizes
    alloc_call = ParallelIR.from_alloc(alloc_args)
    reduce_num = getDistNewID(state)
    reduce_var = symbol("__hpat_gemv_reduce_"*string(reduce_num))
    CompilerTools.LambdaHandling.addLocalVariable(reduce_var, out_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    reduce_var_init = Expr(:(=), reduce_var, Expr(:call,alloc_call...))
    # TODO: deallocate temporary array
    # reduce_var_dealloc = Expr(:call, TopNode(:ccall), QuoteNode(:jl_dealloc_array), reduce_var)

    # get reduction size
    reduce_size_var = symbol("__hpat_gemv_reduce_size_"*string(reduce_num))
    CompilerTools.LambdaHandling.addLocalVariable(reduce_size_var, Int, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    size_expr = Expr(:(=), reduce_size_var, mk_mult_int_expr(out_dim_sizes))

    # add allreduce call
    allreduceCall = Expr(:call, GlobalRef(HPAT.API,:hpat_dist_allreduce), reduce_var, GlobalRef(Base,:add_float), rhs.args[2], reduce_size_var)
    res_copy = Expr(:(=), lhs, rhs.args[2])
    # replace gemm output with local var
    #node.args[1] = reduce_var
    rhs.args[2] = reduce_var

    return [reduce_var_init; node; size_expr; allreduceCall; res_copy]
    # result and vector are sequential if matrix is parallel and transposed
  end
  return [node]
end

function from_parfor(node::Expr, state)
    @assert node.head==:parfor "DistributedPass invalid parfor head"

    parfor = node.args[1]
    parfor.body = from_nested_body(parfor.body, state)

    if state.parfor_partitioning[parfor.unique_id]==ONE_D
      return from_parfor_1d(node, state, parfor)
    elseif state.parfor_partitioning[parfor.unique_id]==TWO_D
      return from_parfor_2d(node, state, parfor)
    else
        # broadcast results of sequential parfors if rand() is used
        has_rand = false
        for stmt in parfor.body
            if isa(stmt,Expr) && stmt.head==:(=) && isa(stmt.args[2],Expr) && stmt.args[2].head==:call &&
                 (isBaseFunc(stmt.args[2].args[1],:rand!) || isBaseFunc(stmt.args[2].args[1],:rand) ||
                   isBaseFunc(stmt.args[2].args[1],:randn!) || isBaseFunc(stmt.args[2].args[1],:randn) )
                has_rand = true
                break
            end
        end
        if has_rand
            # only rank 0 executes rand(), then broadcasts results
            writeArrs = collect(keys(parfor.rws.writeSet.arrays))
            @assert length(writeArrs)==1 "Only one parfor output supported now"
            write_arr = toLHSVar(writeArrs[1])
            # generate new label
            label = next_label(state)
            label_node = LabelNode(label)
            goto_node = Expr(:gotoifnot, :(__hpat_node_id==0),label)
            # get broadcast size
            bcast_size_var = symbol("__hpat_bcast_size_"*string(label))
            CompilerTools.LambdaHandling.addLocalVariable(bcast_size_var, Int, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            size_expr = Expr(:(=), bcast_size_var, mk_mult_int_expr(state.arrs_dist_info[write_arr].dim_sizes))
            bcast_expr = Expr(:call,GlobalRef(HPAT.API,:__hpat_dist_broadcast), write_arr, bcast_size_var)

            @dprintln(3,"DistPass rand() in sequential parfor ", parfor)
            return [goto_node; node; label_node; size_expr; bcast_expr]
        end
    end
    return [node]
end

function from_parfor_1d(node::Expr, state, parfor)
  @dprintln(3,"DistPass translating 1d parfor: ", parfor.unique_id)

  # TODO: assuming 1st loop nest is the last dimension
  loopnest = parfor.loopNests[1]
  # TODO: build a constant table and check the loop variables at this stage
  # @assert loopnest.lower==1 && loopnest.step==1 "DistPass only simple PIR loops supported now"

  loop_start_var = symbol("__hpat_loop_start_"*string(parfor.unique_id))
  loop_end_var = symbol("__hpat_loop_end_"*string(parfor.unique_id))
  loop_div_var = symbol("__hpat_loop_div_"*string(parfor.unique_id))

  CompilerTools.LambdaHandling.addLocalVariable(loop_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(loop_end_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(loop_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

  #first_arr = state.parfor_info[parfor.unique_id][1];
  #@dprintln(3,"DistPass parfor first array ", first_arr)
  #global_size = state.arrs_dist_info[first_arr].dim_sizes[1]

  # some parfors have no arrays
  global_size = loopnest.upper

  loop_div_expr = Expr(:(=),loop_div_var, mk_div_int_expr(global_size,:__hpat_num_pes))
  loop_start_expr = Expr(:(=), loop_start_var, mk_add_int_expr(mk_mult_int_expr([:__hpat_node_id,loop_div_var]),1))
  #loop_end_expr = :($loop_end_var = __hpat_node_id==__hpat_num_pes-1 ?$(global_size):(__hpat_node_id+1)*$loop_div_var)
  loop_end_expr = Expr(:(=), loop_end_var, mk_call(GlobalRef(HPAT.API,:__hpat_get_node_end),[global_size, loop_div_var, :__hpat_num_pes, :__hpat_node_id]))

  loopnest.lower = loop_start_var
  loopnest.upper = loop_end_var

  for stmt in parfor.body
      #adjust_arrayrefs(stmt, loop_start_var)
      ParallelIR.AstWalk(stmt, adjust_arrayrefs, loopnest)
  end
  res = [loop_div_expr; loop_start_expr; loop_end_expr; node]

  dist_reductions = gen_dist_reductions(parfor.reductions, state)
  append!(res, dist_reductions)

  #debug_start_print = :(println("parfor start", $loop_start_var))
  #debug_end_print = :(println("parfor end", $loop_end_var))
  #push!(res,debug_start_print)
  #push!(res,debug_end_print)

  #debug_div_print = :(println("parfor div ", $loop_div_var))
  #push!(res,debug_div_print)
  #debug_pes_print = :(println("parfor pes ", __hpat_num_pes))
  #push!(res,debug_pes_print)
  #debug_rank_print = :(println("parfor rank ", __hpat_node_id))
  #push!(res,debug_rank_print)
  return res
end

function from_parfor_2d(node::Expr, state, parfor)
  @dprintln(3,"DistPass translating 2d parfor: ", parfor.unique_id)

  # TODO: assuming parfor has an array
  # TODO: using parfor index (e.g. comprehension) is not supported yet
  first_arr = state.parfor_arrays[parfor.unique_id][1]
  # assuming 1st loop nest is the last dimension
  parfor.loopNests[1].upper = state.arrs_dist_info[first_arr].local_sizes[end]
  parfor.loopNests[2].upper = state.arrs_dist_info[first_arr].local_sizes[end-1]

  res = [node]

  dist_reductions = gen_dist_reductions(parfor.reductions, state)
  append!(res, dist_reductions)

  return res
end

function next_label(state)
    state.max_label = state.max_label + 1
    return state.max_label
end

function from_call(node::Expr, state)
    @assert node.head==:call "Invalid call node"
    @dprintln(2,"DistPass from_call ", node)

    func = node.args[1]
    if (func==GlobalRef(HPAT.API,:__hpat_data_source_HDF5_read) || func==GlobalRef(HPAT.API,:__hpat_data_source_TXT_read))
        arr = toLHSVar(node.args[3])
        @dprintln(3,"DistPass data source for array: ", arr)
        if isONE_D(arr,state)
          # 1D read, add start and count indices of last dimension
          push!(node.args, state.arrs_dist_info[arr].starts[end], state.arrs_dist_info[arr].counts[end])
        elseif isTWO_D(arr,state)
          push!(node.args, state.arrs_dist_info[arr].starts[end-1], state.arrs_dist_info[arr].starts[end],
           state.arrs_dist_info[arr].strides[end-1], state.arrs_dist_info[arr].strides[end],
           state.arrs_dist_info[arr].counts[end-1], state.arrs_dist_info[arr].counts[end],
           state.arrs_dist_info[arr].blocks[end-1], state.arrs_dist_info[arr].blocks[end],
           state.arrs_dist_info[arr].local_sizes[end-1], state.arrs_dist_info[arr].local_sizes[end],
           state.arrs_dist_info[arr].leftovers[end-1], state.arrs_dist_info[arr].leftovers[end])
        end
        return [node]
    elseif func==GlobalRef(HPAT.API,:__hpat_data_sink_HDF5_write)
        arr = toLHSVar(node.args[4])
        @dprintln(3,"DistPass data source for array: ", arr)
        if isONE_D(arr,state)
          # 1D write, add start and count indices of last dimension, total sizes
          push!(node.args, state.arrs_dist_info[arr].starts[end], state.arrs_dist_info[arr].counts[end],
                    state.arrs_dist_info[arr].dim_sizes)
        elseif isTWO_D(arr,state)
          push!(node.args, state.arrs_dist_info[arr].starts[end-1], state.arrs_dist_info[arr].starts[end],
           state.arrs_dist_info[arr].strides[end-1], state.arrs_dist_info[arr].strides[end],
           state.arrs_dist_info[arr].counts[end-1], state.arrs_dist_info[arr].counts[end],
           state.arrs_dist_info[arr].blocks[end-1], state.arrs_dist_info[arr].blocks[end],
           state.arrs_dist_info[arr].local_sizes[end-1], state.arrs_dist_info[arr].local_sizes[end],
           state.arrs_dist_info[arr].leftovers[end-1], state.arrs_dist_info[arr].leftovers[end],
           state.arrs_dist_info[arr].dim_sizes)
        end

        return [node]
    elseif func==GlobalRef(HPAT.API,:Kmeans) && (isONE_D(toLHSVar(node.args[3]), state) || isONE_D_VAR(toLHSVar(node.args[3]), state))
        # 1st call argument (node.args[2]) is coeffs (lhs of original source code)
        arr = toLHSVar(node.args[3])
        @dprintln(3,"DistPass kmeans call for array: ", arr)
        node.args[1] = GlobalRef(HPAT.API,:Kmeans_dist)

        extra_daal_includes = """ #include "daal.h"
        using namespace daal;
        using namespace daal::data_management;
        using namespace daal::algorithms;
        """
        HPAT.addHpatInclude(extra_daal_includes,"-daal","-daal")
        # rebalance array if necessary
        # table operations like filter produce irregular chunk sizes on different processors
        rebalance_out = gen_rebalance_array(arr, state)

        push!(node.args, state.arrs_dist_info[arr].starts[end], state.arrs_dist_info[arr].counts[end],
                state.arrs_dist_info[arr].dim_sizes[1], state.arrs_dist_info[arr].dim_sizes[end])
        return [rebalance_out; node]
    elseif func==GlobalRef(HPAT.API,:LinearRegression) || func==GlobalRef(HPAT.API,:NaiveBayes)
        # 1st call argument (node.args[2]) is coeffs (lhs of original source code)
        arr1 = toLHSVar(node.args[3])
        arr2 = toLHSVar(node.args[4])
        @dprintln(3,"DistPass LinearRegression/NaiveBayes call for arrays: ", arr1," ", arr2)
        # both arrays can be 1D or variable length 1D
        if (isONE_D(arr1,state) || isONE_D_VAR(arr1,state)) &&
            (isONE_D(arr2,state) || isONE_D_VAR(arr2,state))

            node.args[1] = GlobalRef(HPAT.API, symbol("$(func.name)_dist"))

            extra_daal_includes = """ #include "daal.h"
            using namespace daal;
            using namespace daal::data_management;
            using namespace daal::algorithms;
            """
            HPAT.addHpatInclude(extra_daal_includes,"-daal", "-daal")
            # rebalance array if necessary
            # table operations like filter produce irregular chunk sizes on different processors
            rebalance_out1 = gen_rebalance_array(arr1, state)
            rebalance_out2 = gen_rebalance_array(arr2, state)

            push!(node.args, state.arrs_dist_info[arr1].starts[end], state.arrs_dist_info[arr1].counts[end],
                    state.arrs_dist_info[arr1].dim_sizes[1], state.arrs_dist_info[arr1].dim_sizes[end])
            push!(node.args, state.arrs_dist_info[arr2].starts[end], state.arrs_dist_info[arr2].counts[end],
                    state.arrs_dist_info[arr2].dim_sizes[1], state.arrs_dist_info[arr2].dim_sizes[end])
            return [rebalance_out1; rebalance_out2; node]
        end
    elseif isBaseFunc(func, :arraysize) && (isONE_D(toLHSVar(node.args[2]),state) || isTWO_D(toLHSVar(node.args[2]),state))
        arr = toLHSVar(node.args[2])
        # don't replace if it is variable length (ONE_D_VAR)
        # can be 2D, like hcat-transpose of variable length arrays
        @dprintln(3,"found arraysize on dist array: ",node," ",arr)

        # replace last dimension size queries since it is partitioned
        #if node.args[3]==length(state.arrs_dist_info[arr].dim_sizes)
        #    return [state.arrs_dist_info[arr].dim_sizes[end]]
        #end
        # Parfor array length does not get zero value
        @assert state.arrs_dist_info[arr].dim_sizes[node.args[3]]!=0 "$arr array size could not be zero"
        return [state.arrs_dist_info[arr].dim_sizes[node.args[3]]]
    elseif isBaseFunc(func,:arraylen) && (isONE_D(toLHSVar(node.args[2]), state) || isTWO_D(toLHSVar(node.args[2]),state))
        arr = toLHSVar(node.args[2])
        # don't replace if it is variable length (ONE_D_VAR)
        # can be 2D, like hcat-transpose of variable length arrays

        #len = parse(foldl((a,b)->"$a*$b", "1",state.arrs_dist_info[arr].dim_sizes))
        len = mk_mult_int_expr(state.arrs_dist_info[arr].dim_sizes)
        @dprintln(3,"found arraylen on dist array: ",node," ",arr," len: ",len)
        @dprintln(3,"found arraylen on dist array: ",node," ",arr)
        return [len]
    end
    return [node]
end

function getDistNewID(state)
    state.uniqueId+=1
    return state.uniqueId
end

function adjust_arrayrefs(stmt::Expr, loopnest, top_level_number, is_top_level, read)

    if isCall(stmt)
        topCall = stmt.args[1]
        #ref_args = stmt.args[2:end]
        if isBaseFunc(topCall,:unsafe_arrayref) || isBaseFunc(topCall,:unsafe_arrayset)
            # TODO: simply divide the last dimension, more general partitioning needed
            index_arg = toLHSVar(stmt.args[end])
            if isa(index_arg,Symbol) && index_arg==toLHSVar(loopnest.indexVariable)
                stmt.args[end] = mk_add_int_expr(mk_sub_int_expr(toLHSVar(index_arg),loopnest.lower),1)
                return stmt
            end
        end
    end
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function adjust_arrayrefs(stmt::Any, loopnest,top_level_number, is_top_level, read)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end


function gen_dist_reductions(reductions::Array{PIRReduction,1}, state)
    res = Any[]
    for reduce in reductions
        reduce_var = symbol("__hpat_reduce_"*string(getDistNewID(state)))
        CompilerTools.LambdaHandling.addLocalVariable(reduce_var, reduce.reductionVar.typ, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

        reduce_var_init = Expr(:(=), reduce_var, 0)
        reduceCall = Expr(:call,GlobalRef(HPAT.API,:hpat_dist_allreduce),reduce.reductionVar,reduce.reductionFunc, reduce_var, 1)
        rootCopy = Expr(:(=), reduce.reductionVar, reduce_var)
        append!(res,[reduce_var_init; reduceCall; rootCopy])
    end
    return res
end

"""
    generate code to rebalance 1D or 2D arrays resulting from table operations
    like filter and join.
"""
function gen_rebalance_array(arr::LHSVar, state)
  # no need to rebalance if not variable chunk length
  if !isONE_D_VAR(arr, state)
    return Any[]
  end

  num_dims = length(state.arrs_dist_info[arr].dim_sizes)
  out = []
  arr_id = getDistNewID(state)
  state.arrs_dist_info[arr].arr_id = arr_id

  # get total array size with allreduce
  darr_size_var = Symbol("_glob_arr_size_"*string(arr_id))
  darr_loc_size_var = Symbol("_loc_arr_size_"*string(arr_id))
  CompilerTools.LambdaHandling.addLocalVariable(darr_size_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(darr_loc_size_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  size_var_init = Expr(:(=), darr_size_var, -1)
  # get size of last dimension, assume 1D partitioning
  loc_size_var_init = Expr(:(=), darr_loc_size_var, Expr(:call, GlobalRef(Base, :arraysize), arr, num_dims))
  push!(out, size_var_init)
  push!(out, loc_size_var_init)
  reduceCall = Expr(:call, GlobalRef(HPAT.API,:hpat_dist_allreduce),
    darr_loc_size_var, GlobalRef(Base, :add_int), darr_size_var, 1)
  push!(out, reduceCall)
  state.arrs_dist_info[arr].dim_sizes[end] = darr_size_var

  darr_start_var = symbol("__hpat_dist_arr_start_"*string(arr_id))
  darr_div_var = symbol("__hpat_dist_arr_div_"*string(arr_id))
  darr_count_var = symbol("__hpat_dist_arr_count_"*string(arr_id))
  state.arrs_dist_info[arr].starts[end] = darr_start_var
  state.arrs_dist_info[arr].counts[end] = darr_count_var

  CompilerTools.LambdaHandling.addLocalVariable(darr_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(darr_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
  CompilerTools.LambdaHandling.addLocalVariable(darr_count_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

  darr_div_expr = Expr(:(=),darr_div_var, mk_div_int_expr(darr_size_var,:__hpat_num_pes))
  # zero-based index to match C interface of HDF5
  darr_start_expr = Expr(:(=), darr_start_var, mk_mult_int_expr([:__hpat_node_id,darr_div_var]))
  darr_count_expr = Expr(:(=), darr_count_var, mk_call(GlobalRef(HPAT.API,:__hpat_get_node_portion),[darr_size_var, darr_div_var, :__hpat_num_pes, :__hpat_node_id]))
  push!(out, darr_div_expr, darr_start_expr, darr_count_expr)

  rebalance_call = mk_call(GlobalRef(HPAT.API,:__hpat_arr_rebalance),[arr, darr_count_var])
  push!(out, rebalance_call)

  return out
end

end # DistributedPass
