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

using ParallelAccelerator
import ParallelAccelerator.ParallelIR
import ParallelAccelerator.ParallelIR.toSynGemOrInt
import ParallelAccelerator.ParallelIR.isArrayType
import ParallelAccelerator.ParallelIR.getParforNode
import ParallelAccelerator.ParallelIR.isAllocation
import ParallelAccelerator.ParallelIR.TypedExpr
import ParallelAccelerator.ParallelIR.get_alloc_shape

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

dist_ir_funcs = Set([   TopNode(:unsafe_arrayref),
                        TopNode(:unsafe_arrayset),
                        :__hpat_data_source_HDF5_open, 
                        :__hpat_data_source_HDF5_size, 
                        :__hpat_get_H5_dim_size, 
                        :__hpat_data_source_HDF5_read, 
                        :__hpat_data_source_TXT_open,
                        :__hpat_data_source_TXT_size,
                        :__hpat_get_TXT_dim_size,
                        :__hpat_data_source_TXT_read,
                        :__hpat_Kmeans,
                        :__hpat_LinearRegression,
                        :__hpat_NaiveBayes,
                        GlobalRef(Base,:arraylen), TopNode(:arraysize), GlobalRef(Base,:reshape), TopNode(:tuple), 
                        GlobalRef(Base.LinAlg,:gemm_wrapper!)])

# ENTRY to distributedIR
function from_root(function_name, ast :: Expr)
    @assert ast.head == :lambda "Input to DistributedPass should be :lambda Expr"
    @dprintln(1,"Starting main DistributedPass.from_root.  function = ", function_name, " ast = ", ast)

    linfo = CompilerTools.LambdaHandling.lambdaExprToLambdaVarInfo(ast)
    lives = CompilerTools.LivenessAnalysis.from_expr(ast, ParallelIR.pir_live_cb, linfo)
    state::DistPassState = initDistState(linfo,lives)
    
    # find if an array should be partitioned, sequential, or shared
    getArrayDistributionInfo(ast, state)
    
    # transform body
    @assert ast.args[3].head==:body "DistributedPass: invalid lambda input"
    body = TypedExpr(ast.args[3].typ, :body, from_toplevel_body(ast.args[3].args, state)...)
    new_ast = CompilerTools.LambdaHandling.LambdaVarInfoToLambdaExpr(state.LambdaVarInfo, body)
    @dprintln(1,"DistributedPass.from_root returns function = ", function_name, " ast = ", new_ast)
    # ast = from_expr(ast)
    return new_ast
end

type ArrDistInfo
    isSequential::Bool      # can't be distributed; e.g. it is used in sequential code
    dim_sizes::Array{Union{SymAllGen,Int,Expr},1}      # sizes of array dimensions
    # assuming only last dimension is partitioned
    arr_id::Int # assign ID to distributed array to access partitioning info later
    
    function ArrDistInfo(num_dims::Int)
        new(false, zeros(Int64,num_dims))
    end
end

function show(io::IO, pnode::HPAT.DistributedPass.ArrDistInfo)
    print(io,"seq:",pnode.isSequential," sizes:", pnode.dim_sizes)
end

# information about AST gathered and used in DistributedPass
type DistPassState
    # information about all arrays
    arrs_dist_info::Dict{SymGen, ArrDistInfo}
    parfor_info::Dict{Int, Array{SymGen,1}}
    LambdaVarInfo::LambdaVarInfo
    seq_parfors::Array{Int,1}
    dist_arrays::Array{SymGen,1}
    uniqueId::Int
    lives  :: CompilerTools.LivenessAnalysis.BlockLiveness
    # keep values for constant tuples. They are often used for allocating and reshaping arrays.
    tuple_table              :: Dict{SymGen,Array{Union{SymGen,Int},1}}
    max_label :: Int # holds the max number of all LabelNodes

    function DistPassState(linfo, lives)
        new(Dict{SymGen, Array{ArrDistInfo,1}}(), Dict{Int, Array{SymGen,1}}(), linfo, Int[], SymGen[],0, lives, 
             Dict{SymGen,Array{Union{SymGen,Int},1}}(),0)
    end
end

include("distributed-pass-ast-walk.jl")

function show(io::IO, pnode::HPAT.DistributedPass.DistPassState)
    println(io,"DistPassState arrs_dist_info:")
    for i in pnode.arrs_dist_info
        println(io,"  ", i)
    end
    println(io,"DistPassState parfor_info:")
    for i in pnode.parfor_info
        println(io,"  ", i)
    end
    println(io,"DistPassState seq_parfors:")
    for i in pnode.seq_parfors
        print(io," ", i)
    end
    println(io,"")
    println(io,"DistPassState dist_arrays:")
    for i in pnode.dist_arrays
        print(io," ", i)
    end
    println(io,"")
end

function initDistState(linfo::LambdaVarInfo, lives)
    state = DistPassState(linfo, lives)
    
    #params = linfo.input_params
    vars = linfo.var_defs
    gensyms = linfo.gen_sym_typs

    # Populate the symbol table
    for sym in keys(vars)
        v = vars[sym] # v is a VarDef
        if isArrayType(v.typ)
            arrInfo = ArrDistInfo(ndims(v.typ))
            state.arrs_dist_info[sym] = arrInfo
        end 
    end

    for k in 1:length(gensyms)
        typ = gensyms[k]
        if isArrayType(typ)
            arrInfo = ArrDistInfo(ndims(typ))
            state.arrs_dist_info[GenSym(k-1)] = arrInfo
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


function from_expr(node::Expr, state::DistPassState)
    head = node.head
    if head==:(=)
        return from_assignment(node, state)
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
    initCall = Expr(:call,TopNode(:hps_dist_init))
    numPesCall = Expr(:call,TopNode(:hps_dist_num_pes))
    nodeIdCall = Expr(:call,TopNode(:hps_dist_node_id))
    
    CompilerTools.LambdaHandling.addLocalVar(symbol("__hpat_num_pes"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    CompilerTools.LambdaHandling.addLocalVar(symbol("__hpat_node_id"), Int32, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

    num_pes_assign = Expr(:(=), :__hpat_num_pes, numPesCall)
    node_id_assign = Expr(:(=), :__hpat_node_id, nodeIdCall)

    return Any[initCall; num_pes_assign; node_id_assign]
end

function from_assignment(node::Expr, state::DistPassState)
    @assert node.head==:(=) "DistributedPass invalid assignment head"
    lhs = node.args[1]
    rhs = node.args[2]
    
    if isAllocation(rhs)
        arr = toSymGen(lhs)
        if in(arr, state.dist_arrays)
            @dprintln(3,"DistPass allocation array: ", arr)
            #shape = get_alloc_shape(node.args[2].args[2:end])
            #old_size = shape[end]
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

            CompilerTools.LambdaHandling.addLocalVar(darr_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(darr_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(darr_count_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)


            darr_div_expr = Expr(:(=),darr_div_var, mk_div_int_expr(arr_tot_size,:__hpat_num_pes))
            # zero-based index to match C interface of HDF5
            darr_start_expr = Expr(:(=), darr_start_var, mk_mult_int_expr([:__hpat_node_id,darr_div_var])) 
            # darr_count_expr = :($darr_count_var = __hpat_node_id==__hpat_num_pes-1 ? $arr_tot_size-__hpat_node_id*$darr_div_var : $darr_div_var)
            darr_count_expr = Expr(:(=), darr_count_var, mk_call(:__hpat_get_node_portion,[arr_tot_size, darr_div_var, :__hpat_num_pes, :__hpat_node_id])) 

            rhs.args[end-1] = darr_count_var

            res = [darr_div_expr; darr_start_expr; darr_count_expr; node]
            #debug_size_print = :(println("size ",$darr_count_var))
            #push!(res,debug_size_print)
            return res
        end
    elseif isa(rhs,Expr) && rhs.head==:call && rhs.args[1]==GlobalRef(Base,:reshape)
        arr = toSymGen(lhs)
        if in(arr, state.dist_arrays)
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
    
            CompilerTools.LambdaHandling.addLocalVar(darr_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(darr_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(darr_count_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
    
    
            darr_div_expr = Expr(:(=), darr_div_var, mk_div_int_expr(arr_tot_size,:__hpat_num_pes))
            # zero-based index to match C interface of HDF5
            darr_start_expr = Expr(:(=),darr_start_var, mk_mult_int_expr([:__hpat_node_id, darr_div_var]))
            #darr_count_expr = :($darr_count_var = __hpat_node_id==__hpat_num_pes-1 ? $arr_tot_size-__hpat_node_id*$darr_div_var : $darr_div_var)
            darr_count_expr = Expr(:(=), darr_count_var, mk_call(:__hpat_get_node_portion,[arr_tot_size, darr_div_var, :__hpat_num_pes, :__hpat_node_id])) 
    
            # create a new tuple for reshape
            tup_call = Expr(:call, TopNode(:tuple), dim_sizes[1:end-1]... , darr_count_var)
            reshape_tup_var = symbol("__hpat_dist_tup_var_"*string(arr_id))
            tup_typ = CompilerTools.LambdaHandling.getType(rhs.args[3], state.LambdaVarInfo)
            CompilerTools.LambdaHandling.addLocalVar(reshape_tup_var, tup_typ, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            tup_expr = Expr(:(=),reshape_tup_var,tup_call)
            rhs.args[3] = reshape_tup_var
            res = [darr_div_expr; darr_start_expr; darr_count_expr; tup_expr; node]
            #debug_size_print = :(println("size ",$darr_count_var))
            #push!(res,debug_size_print)
            return res
        end
    elseif isa(rhs,Expr) && rhs.head==:call && rhs.args[1]==GlobalRef(Base.LinAlg,:gemm_wrapper!)

                arr1 = toSymGen(rhs.args[5])
                t1 = (rhs.args[3]=='T')
                arr2 = toSymGen(rhs.args[6])
                t2 = (rhs.args[4]=='T')
                
                # result is sequential but with reduction if both inputs are partitioned and second one is transposed
                # e.g. labels*points'
                if !state.arrs_dist_info[arr1].isSequential && !state.arrs_dist_info[arr2].isSequential && t2 && !t1 &&
                            state.arrs_dist_info[lhs].isSequential
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
                    CompilerTools.LambdaHandling.addLocalVar(reduce_var, out_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
                    reduce_var_init = Expr(:(=), reduce_var, Expr(:call,alloc_call...))
                    # TODO: deallocate temporary array
                    # reduce_var_dealloc = Expr(:call, TopNode(:ccall), QuoteNode(:jl_dealloc_array), reduce_var)

                    # get reduction size
                    reduce_size_var = symbol("__hpat_gemm_reduce_size_"*string(reduce_num))
                    CompilerTools.LambdaHandling.addLocalVar(reduce_size_var, Int, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
                    size_expr = Expr(:(=), reduce_size_var, mk_mult_int_expr(out_dim_sizes))

                    # add allreduce call
                    allreduceCall = Expr(:call,TopNode(:hps_dist_allreduce), reduce_var, TopNode(:add_float), rhs.args[2], reduce_size_var)
                    res_copy = Expr(:(=), lhs, rhs.args[2])
                    # replace gemm output with local var
                    #node.args[1] = reduce_var
                    rhs.args[2] = reduce_var

                    return [reduce_var_init; node; size_expr; allreduceCall; res_copy]
                # first input is sequential but output is parallel if the second input is partitioned but not transposed
                # e.g. w*points
                elseif state.arrs_dist_info[arr1].isSequential && !state.arrs_dist_info[arr2].isSequential && !t2 && !state.arrs_dist_info[lhs].isSequential
                    @dprintln(3,"DistPass arr info gemm first input is sequential: ", arr1)
                    #rhs.args[1] = :__hpat_gemm_broadcast
                    @dprintln(3,"DistPass translating gemm broadcast: ", node)
                # otherwise, no known pattern found
                end
    else
        node.args[2] = from_expr(rhs,state)[1]
    end
    return [node]
end

function from_parfor(node::Expr, state)
    @assert node.head==:parfor "DistributedPass invalid parfor head"

    parfor = node.args[1]

    if !in(parfor.unique_id, state.seq_parfors)
        @dprintln(3,"DistPass translating parfor: ", parfor.unique_id)
        # TODO: assuming 1st loop nest is the last dimension
        loopnest = parfor.loopNests[1]
        # TODO: build a constant table and check the loop variables at this stage
        # @assert loopnest.lower==1 && loopnest.step==1 "DistPass only simple PIR loops supported now"

        loop_start_var = symbol("__hpat_loop_start_"*string(parfor.unique_id))
        loop_end_var = symbol("__hpat_loop_end_"*string(parfor.unique_id))
        loop_div_var = symbol("__hpat_loop_div_"*string(parfor.unique_id))

        CompilerTools.LambdaHandling.addLocalVar(loop_start_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
        CompilerTools.LambdaHandling.addLocalVar(loop_end_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
        CompilerTools.LambdaHandling.addLocalVar(loop_div_var, Int, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

        #first_arr = state.parfor_info[parfor.unique_id][1]; 
        #@dprintln(3,"DistPass parfor first array ", first_arr)
        #global_size = state.arrs_dist_info[first_arr].dim_sizes[1]

        # some parfors have no arrays
        global_size = loopnest.upper

        loop_div_expr = Expr(:(=),loop_div_var, mk_div_int_expr(global_size,:__hpat_num_pes))
        loop_start_expr = Expr(:(=), loop_start_var, mk_add_int_expr(mk_mult_int_expr([:__hpat_node_id,loop_div_var]),1))
        #loop_end_expr = :($loop_end_var = __hpat_node_id==__hpat_num_pes-1 ?$(global_size):(__hpat_node_id+1)*$loop_div_var)
        loop_end_expr = Expr(:(=), loop_end_var, mk_call(:__hpat_get_node_end,[global_size, loop_div_var, :__hpat_num_pes, :__hpat_node_id]))

        loopnest.lower = loop_start_var
        loopnest.upper = loop_end_var

        for stmt in parfor.body
            adjust_arrayrefs(stmt, loop_start_var)
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
    else
        # broadcast results of sequential parfors if rand() is used
        has_rand = false
        for stmt in parfor.body
            if isa(stmt,Expr) && stmt.head==:(=) && isa(stmt.args[2],Expr) && stmt.args[2].head==:call && stmt.args[2].args[1]==TopNode(:rand!)
                has_rand = true
                break
            end
        end
        if has_rand
            # only rank 0 executes rand(), then broadcasts results
            writeArrs = collect(keys(parfor.rws.writeSet.arrays))
            @assert length(writeArrs)==1 "Only one parfor output supported now"
            write_arr = toSymGen(writeArrs[1])
            # generate new label
            label = next_label(state)
            label_node = LabelNode(label)
            goto_node = Expr(:gotoifnot, :(__hpat_node_id==0),label)
            # get broadcast size
            bcast_size_var = symbol("__hpat_bcast_size_"*string(label))
            CompilerTools.LambdaHandling.addLocalVar(bcast_size_var, Int, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)
            size_expr = Expr(:(=), bcast_size_var, mk_mult_int_expr(state.arrs_dist_info[write_arr].dim_sizes))
            bcast_expr = Expr(:call,:__hpat_dist_broadcast, write_arr, bcast_size_var)

            @dprintln(3,"DistPass rand() in sequential parfor ", parfor)
            return [goto_node; node; label_node; size_expr; bcast_expr]
        end
    end
    return [node]
end

function next_label(state)
    state.max_label = state.max_label + 1
    return state.max_label
end

function from_call(node::Expr, state)
    @assert node.head==:call "Invalid call node"
    @dprintln(2,"DistPass from_call ", node)

    func = node.args[1]
    if (func==:__hpat_data_source_HDF5_read || func==:__hpat_data_source_TXT_read) && in(toSymGen(node.args[3]), state.dist_arrays)
        arr = toSymGen(node.args[3])
        @dprintln(3,"DistPass data source for array: ", arr)
        
        arr_id = state.arrs_dist_info[arr].arr_id 
        
        dsrc_start_var = symbol("__hpat_dist_arr_start_"*string(arr_id)) 
        dsrc_count_var = symbol("__hpat_dist_arr_count_"*string(arr_id)) 

        push!(node.args, dsrc_start_var, dsrc_count_var)
        return [node]
    elseif func==:__hpat_Kmeans && in(toSymGen(node.args[3]), state.dist_arrays)
        arr = toSymGen(node.args[3])
        @dprintln(3,"DistPass kmeans call for array: ", arr)
        
        arr_id = state.arrs_dist_info[arr].arr_id 
        
        dsrc_start_var = symbol("__hpat_dist_arr_start_"*string(arr_id))
        dsrc_count_var = symbol("__hpat_dist_arr_count_"*string(arr_id)) 

        push!(node.args, dsrc_start_var, dsrc_count_var, 
                state.arrs_dist_info[arr].dim_sizes[1], state.arrs_dist_info[arr].dim_sizes[end])
        return [node]
    elseif (func==:__hpat_LinearRegression || func==:__hpat_NaiveBayes) && in(toSymGen(node.args[3]), state.dist_arrays) && in(toSymGen(node.args[4]), state.dist_arrays)
        arr1 = toSymGen(node.args[3])
        arr2 = toSymGen(node.args[4])
        @dprintln(3,"DistPass LinearRegression/NaiveBayes call for arrays: ", arr1," ", arr2)
        
        arr1_id = state.arrs_dist_info[arr1].arr_id 
        arr2_id = state.arrs_dist_info[arr2].arr_id 
        
        dsrc_start_var1 = symbol("__hpat_dist_arr_start_"*string(arr1_id))
        dsrc_count_var1 = symbol("__hpat_dist_arr_count_"*string(arr1_id)) 
        
        dsrc_start_var2 = symbol("__hpat_dist_arr_start_"*string(arr2_id))
        dsrc_count_var2 = symbol("__hpat_dist_arr_count_"*string(arr2_id)) 

        push!(node.args, dsrc_start_var1, dsrc_count_var1,
                state.arrs_dist_info[arr1].dim_sizes[1], state.arrs_dist_info[arr1].dim_sizes[end])
        push!(node.args, dsrc_start_var2, dsrc_count_var2,
                state.arrs_dist_info[arr2].dim_sizes[1], state.arrs_dist_info[arr2].dim_sizes[end])
        return [node]
    elseif isTopNode(func) && func.name==:arraysize && in(toSymGen(node.args[2]), state.dist_arrays)
        arr = toSymGen(node.args[2])
        @dprintln(3,"found arraysize on dist array: ",node," ",arr)
        # replace last dimension size queries since it is partitioned
        if node.args[3]==length(state.arrs_dist_info[arr].dim_sizes)
            return [state.arrs_dist_info[arr].dim_sizes[end]]
        end
    elseif func==GlobalRef(Base,:arraylen) && in(toSymGen(node.args[2]), state.dist_arrays)
        arr = toSymGen(node.args[2])
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

function adjust_arrayrefs(stmt::Expr, loop_start_var::Symbol)
    
    if stmt.head==:(=)
        stmt = stmt.args[2]
    end
    
    if isCall(stmt) && isTopNode(stmt.args[1])
        topCall = stmt.args[1]
        #ref_args = stmt.args[2:end]
        if topCall.name==:unsafe_arrayref || topCall.name==:unsafe_arrayset
            # TODO: simply divide the last dimension, more general partitioning needed
            index_arg = stmt.args[end]
            stmt.args[end] = mk_add_int_expr(mk_sub_int_expr(toSymGen(index_arg),loop_start_var),1)
        end
    end
end

function adjust_arrayrefs(stmt::Any, loop_start_var::Symbol)
end


function gen_dist_reductions(reductions::Array{PIRReduction,1}, state)
    res = Any[]
    for reduce in reductions
        reduce_var = symbol("__hpat_reduce_"*string(getDistNewID(state)))
        CompilerTools.LambdaHandling.addLocalVar(reduce_var, reduce.reductionVar.typ, ISASSIGNEDONCE | ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo)

        reduce_var_init = Expr(:(=), reduce_var, 0)
        reduceCall = Expr(:call,TopNode(:hps_dist_reduce),reduce.reductionVar,reduce.reductionFunc, reduce_var)
        rootCopy = Expr(:(=), reduce.reductionVar, reduce_var)
        append!(res,[reduce_var_init; reduceCall; rootCopy])
    end
    return res
end

end # DistributedPass
