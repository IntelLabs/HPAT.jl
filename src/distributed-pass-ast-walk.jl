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

#using Debug

using CompilerTools.LivenessAnalysis
using CompilerTools.TransitiveDependence

import ParallelAccelerator.ParallelIR.PIRLoopNest
import ParallelAccelerator.ParallelIR.InputInfo

function getArrayDistributionInfo(ast, state)
    set_user_partitionings(state)
    before_arr_partitionings = [state.arrs_dist_info[arr].partitioning for arr in keys(state.arrs_dist_info)]

    while true
        @dprintln(3,"DistPass state before array info walk: ",state)
        ParallelIR.AstWalk(ast, get_arr_dist_info, state)
        @dprintln(3,"DistPass state after array info walk: ",state)
        set_user_partitionings(state)
        new_arr_partitionings = [state.arrs_dist_info[arr].partitioning for arr in keys(state.arrs_dist_info)]
        # break if no new sequential array discovered
        if new_arr_partitionings==before_arr_partitionings
            break
        end
        before_arr_partitionings = new_arr_partitionings
    end

    # if all parfors are sequential
    if HPAT.force_parallel && all([state.parfor_partitioning[parfor_id]==SEQ for parfor_id in keys(state.parfor_partitioning)]) &&
          all([state.arrs_dist_info[arr].partitioning==SEQ for arr in keys(state.arrs_dist_info)])
        error("HPAT failed to parallelize! Fix the parallelism problem or use \"HPAT.setForceParallel(false)\" to run sequentially.")
    end
    # for debugging purposes
    save_array_partitionings(state)
end

function set_user_partitionings(state)
    for (var,part) in state.user_partitionings
        arr = CompilerTools.LambdaHandling.lookupLHSVarByName(var, state.LambdaVarInfo)
      state.arrs_dist_info[arr].partitioning = part
    end
end

function save_array_partitionings(state)
    d = Dict{LHSVar,Partitioning}()
    for (k,v) in state.arrs_dist_info
        d[k] = state.arrs_dist_info[k].partitioning
    end
    HPAT.set_saved_array_partitionings(d, state.LambdaVarInfo)
end

"""
mark sequential arrays
"""
function get_arr_dist_info(node::Expr, state::DistPassState, top_level_number, is_top_level, read)
    head = node.head
    # arrays written in parfors are ok for now

    @dprintln(3,"DistPass arr info walk Expr head: ", head)
    if head==:(=)
        @dprintln(3,"DistPass arr info walk assignment: ", node)
        lhs = toLHSVar(node.args[1])
        rhs = node.args[2]
        return get_arr_dist_info_assignment(node, state, top_level_number, lhs, rhs)
    elseif head==:parfor
        @dprintln(3,"DistPass arr info walk parfor: ", node)
        parfor = getParforNode(node)
        return get_arr_dist_info_parfor(node, state, top_level_number, parfor)
        # functions dist_ir_funcs are either handled here or do not make arrays sequential
    elseif head==:call && (isa(node.args[1],GlobalRef) || isa(node.args[1],TopNode)) && in(node.args[1].name, dist_ir_funcs)
        func = node.args[1].name
        if func==:__hpat_data_source_HDF5_read || func==:__hpat_data_source_TXT_read
            @dprintln(2,"DistPass arr info walk data source read ", node)
            # will be parallel IO, intentionally do nothing
        elseif func==:Kmeans
            @dprintln(2,"DistPass arr info walk kmeans ", node)
            # first array is cluster output and is sequential
            # second array is input matrix and is parallel
            setSEQ(toLHSVar(node.args[2]),state)
        elseif func==:LinearRegression || func==:NaiveBayes
            @dprintln(2,"DistPass arr info walk LinearRegression/NaiveBayes ", node)
            # first array is cluster output and is sequential
            # second array is input matrix and is parallel
            # third array is responses and is parallel
            setSEQ(toLHSVar(node.args[2]),state)
        elseif func==:__hpat_join
            # output columns of join can have variable length on different processors
            # set partitioning to ONE_D_VAR
            # node.args[3] gives number of output columns
            start_col_ind = 6
            end_col_ind = start_col_ind + node.args[3] - 1
            for col_ind in start_col_ind:end_col_ind
                col = node.args[col_ind]
                # table columns are 1D
                @assert length(state.arrs_dist_info[col].dim_sizes)==1
                state.arrs_dist_info[col].partitioning = ONE_D_VAR
            end
        elseif func==:__hpat_filter
            # output columns of filter can have variable length on different processors
            # set partitioning to ONE_D_VAR
            # node.args[4] gives number of output columns
            start_col_ind = 5
            end_col_ind = start_col_ind + node.args[4] - 1
            for col_ind in start_col_ind:end_col_ind
                col = node.args[col_ind]
                # table columns are 1D
                @assert length(state.arrs_dist_info[col].dim_sizes)==1
                state.arrs_dist_info[col].partitioning = ONE_D_VAR
            end
        elseif func==:__hpat_aggregate
            # output columns of aggregate can have variable length on different processors
            # set partitioning to ONE_D_VAR
            # node.args[4] gives number of output columns
            # start index is 2*node.args[4] because we need to skips expression columns and functions
            start_col_ind = (2*node.args[4]) + 5
            end_col_ind = start_col_ind + node.args[4] - 1
            # one extra output to account for output key column
            end_col_ind += 1
            for col_ind in start_col_ind:end_col_ind
                col = node.args[col_ind]
                # table columns are 1D
                @assert length(state.arrs_dist_info[col].dim_sizes)==1
                state.arrs_dist_info[col].partitioning = ONE_D_VAR
            end
        end
        return node
    elseif head==:gotoifnot
        @dprintln(3,"DistPass arr info gotoifnot: ", node)
        return CompilerTools.AstWalker.ASTWALK_RECURSE
        # arrays written in serial code are not distributed
    elseif head!=:body && head!=:block && head!=:lambda
        @dprintln(3,"DistPass arr info walk serial code: ", node)
        return get_arr_dist_info_serial_code(node, state, top_level_number)
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function get_arr_dist_info_serial_code(node, state, top_level_number)
    live_info = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo,
                                                           TypedExpr(nothing, :body, node), ParallelIR.pir_live_cb, state.LambdaVarInfo)
    # @dprintln(3, "body_lives = ", body_lives)
    # live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, state.lives)
    # all_vars = union(live_info.def, live_info.use)
    all_vars = []

    for bb in collect(values(live_info.basic_blocks))
        for stmt in bb.statements
            append!(all_vars, collect(union(stmt.def, stmt.use)))
        end
    end

    @dprintln(3,"DistPass arr info walk serial code vars: ", all_vars)
    # ReadWriteSet is not robust enough now
    #rws = CompilerTools.ReadWriteSet.from_exprs([node], ParallelIR.pir_live_cb, state.LambdaVarInfo)
    #readArrs = collect(keys(rws.readSet.arrays))
    #writeArrs = collect(keys(rws.writeSet.arrays))
    #allArrs = [readArrs;writeArrs]

    for var in all_vars
        if haskey(state.arrs_dist_info, toLHSVar(var))
            @dprintln(2,"DistPass arr info walk array sequential since in serial code: ", var, "  ",lookupVariableName(var, state.LambdaVarInfo), " ", node)
            setSEQ(toLHSVar(var),state)
        end
    end
    return node
end

function get_arr_dist_info_parfor(node, state, top_level_number, parfor)
    rws = CompilerTools.ReadWriteSet.from_exprs(parfor.body, ParallelAccelerator.ParallelIR.pir_rws_cb, state.LambdaVarInfo)
    partitioning = ONE_D

    indexVariable = toLHSVar(parfor.loopNests[1].indexVariable)

    allArrAccesses = merge(rws.readSet.arrays,rws.writeSet.arrays)
    myArrs = LHSVar[]
    stencil_inds = Int[]

    body_lives = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo,
                                                            parfor.body, ParallelIR.pir_live_cb, state.LambdaVarInfo)
    # @dprintln(3, "body_lives = ", body_lives)
    @dprintln(2,"DistPass arr info walk parfor indexVariable: ", indexVariable, " accesses: ", allArrAccesses)
    @dprintln(2,"parfor = ", parfor)
    # If an array is accessed with a Parfor's index variable, the parfor and array should have same partitioning
    for arr in keys(allArrAccesses)
        # an array can be accessed multiple times in Pafor
        # for each access:
        for access_indices in allArrAccesses[arr]
            @dprintln(3, "access_indices = ", access_indices)
            indices = map(toLHSVar,access_indices)
            # if array would be accessed in parallel in this Parfor
            if indices[end]==indexVariable
                # put read arrays first
                if arr in keys(rws.readSet.arrays)
                    myArrs = LHSVar[arr; myArrs]
                else
                    push!(myArrs, arr)
                end
            end
            # only 1D stencil is supported for now
            if length(indices)==1 && isStencilAccess(indices[1], indexVariable)
                # put input array of stencil first
                myArrs = LHSVar[arr; myArrs]
                stencil_rel_ind = getStencilAccessInd(indices[1])
                push!(stencil_inds, stencil_rel_ind)
                continue
            end
            # An array access index can be dependent on parfor's
            # index variable as in nested comprehension case of K-Means.
            # Parfor can't be parallelized in general cases since array can't be partitioned properly.
            # ParallelIR should optimize out the trivial cases where indices are essentially equal (i=1+1*index-1 in k-means)
            if isAccessIndexDependent(indices[1:end-1], indexVariable, body_lives, state)
                @dprintln(2,"DistPass arr info walk arr index dependent: ",arr," ", indices, " ", indexVariable)
                partitioning = SEQ
            end
            if isAccessIndexDependent(indices[end], indexVariable, body_lives, state)
                push!(myArrs, arr)
                @dprintln(2,"DistPass arr info walk arr index dependent: ",arr," ", indices, " ", indexVariable)
                partitioning = SEQ
            end
            # sequential if not accessed column major (last dimension)
            # TODO: generalize?
            if in(indexVariable, indices[1:end-1])
                @dprintln(2,"DistPass arr info walk arr index sequential: ",arr," ", indices, " ", indexVariable)
                partitioning = SEQ
            end
        end
    end
    # keep mapping from parfors to arrays
    # state.parfor_info[parfor.unique_id] = myArrs
    @dprintln(3,"DistPass arr info walk parfor arrays: ", myArrs)
    @dprintln(3,"DistPass arr info walk parfor partitioning: ", partitioning)

    # stencils have arrays read/written past index
    if length(stencil_inds)==0 && (length(parfor.arrays_read_past_index)!=0 || length(parfor.arrays_written_past_index)!=0)
        @dprintln(2,"DistPass arr info walk parfor sequential: ", node)
        partitioning = SEQ
    end

    for arr in myArrs
        partitioning = min(partitioning,getArrayPartitioning(arr,state))
        if isSEQ(arr,state)
            # no need to check size for parallel arrays since ParallelIR already used equivalence class info
            # || !eqSize(state.arrs_dist_info[arr].dim_sizes[end], state.arrs_dist_info[myArrs[1]].dim_sizes[end])
            # last dimension of all parfor arrays should be equal since they are partitioned
            @dprintln(2,"DistPass parfor check array: ", arr," sequential: ", isSEQ(arr,state))
        end
    end
    # parfor and all its arrays have same partitioning
    state.parfor_partitioning[parfor.unique_id] = partitioning
    state.parfor_arrays[parfor.unique_id] = myArrs
    for arr in myArrs
        setArrayPartitioning(arr,partitioning,state)
    end
    if length(stencil_inds)!=0
        state.parfor_stencils[parfor.unique_id] = stencil_inds
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

#function isAccessIndexDependent(indices::Vector{Any}, indexVariable::LHSVar, body_lives::BlockLiveness, state)
#    deps = CompilerTools.TransitiveDependence.computeDependencies(body_lives)
function isAccessIndexDependent(indices::Vector{Any}, indexVariable::LHSVar, deps, state)
    # if any index is dependent
    return reduce(|, [ isAccessIndexDependent(indices[i], indexVariable, deps, state) for i in 1:length(indices)] )
end

function isAccessIndexDependent(index::LHSVar, indexVariable::LHSVar, deps::Dict{LHSVar, Set{LHSVar}}, state)
    @dprintln(3,"isAccessIndexDependent for index ", index, " and variable ", indexVariable, " with deps ", deps)
    if index == indexVariable
        return false
    end

    if haskey(deps, index)
        ret = in(indexVariable, deps[index])
        @dprintln(3,"isAccessIndexDependent returned ", ret)
        return ret
    else
        @dprintln(2, "isAccessIndexDependent could not find index ", index, " in deps.")
        return false
    end
end

function isAccessIndexDependent(index::Int, indexVariable::LHSVar, deps::Dict{LHSVar, Set{LHSVar}}, state)
    return false
end

"""
For each statement of parfor's body, see if array access index is in Def set and parfor's indexVariable is in Use set.
This is a hack to work around not having dependence analysis in CompilerTools.
"""
function isAccessIndexDependent(index::LHSVar, indexVariable::LHSVar, body_lives::BlockLiveness, state)
    for bb in collect(values(body_lives.basic_blocks))
        for stmt in bb.statements
            if isBareParfor(stmt.tls.expr)
                return isAccessIndexDependent(index, indexVariable, state.deps, state)
            elseif in(index,stmt.def) && in(indexVariable, stmt.use)
                @dprintln(2,"DistPass arr info walk dependent index found: ", index," ",indexVariable,"  ",stmt)
                return true
            end
        end
    end
    return false
end

function isAccessIndexDependent(index::Expr, indexVariable::LHSVar, body_lives::BlockLiveness, state)
    return reduce(|, [ isAccessIndexDependent(index.args[i], indexVariable, body_lives, state) for i in 1:length(index.args)] )
end

function isAccessIndexDependent(index::Union{GlobalRef,Type}, indexVariable::LHSVar, body_lives::BlockLiveness, state)
    return false
end

# TODO: is this general?
function isAccessIndexDependent(index::Int, indexVariable::LHSVar, body_lives::BlockLiveness, state)
    return false
end

function get_arr_dist_info(ast::Any, state::DistPassState, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function isStencilAccess(index::Expr, indexVariable::LHSVar)
    # example :((Base.box)(Int64,(Core.Intrinsics.add_int)(_9,-1)))
    if index.head==:call && index.args[1]==GlobalRef(Base,:box) && isa(index.args[3],Expr) &&
        index.args[3].head==:call && index.args[3].args[1]==GlobalRef(Core.Intrinsics,:add_int) &&
        index.args[3].args[2]==indexVariable && isa(index.args[3].args[3],Int)
        @dprintln(3, "stencil found ", index.args[3].args[3])
        return true
    end
    return false
end

isStencilAccess(index::LHSVar, indexVariable::LHSVar) = false
isStencilAccess(index::Int, indexVariable::LHSVar) = false

function getStencilAccessInd(index::Expr)
    return index.args[3].args[3]
end


""" return LHSVar if arg is RHSVar, otherwise no change
    used for allocation sizes which could be LHSVar or Int or TypedVar or Expr
"""
replaceAllocTypedVar(a::TypedVar) = toLHSVar(a)
replaceAllocTypedVar(a::Union{Int,LHSVar,Expr}) = a

function get_arr_dist_info_assignment(node::Expr, state::DistPassState, top_level_number, lhs::LHSVar, rhs::RHSVar)
    rhs = toLHSVar(rhs)
    if haskey(state.arrs_dist_info, rhs)
        state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[rhs].dim_sizes
        state.arrs_dist_info[lhs].starts = state.arrs_dist_info[rhs].starts
        state.arrs_dist_info[lhs].counts = state.arrs_dist_info[rhs].counts
        state.arrs_dist_info[lhs].strides = state.arrs_dist_info[rhs].strides
        state.arrs_dist_info[lhs].blocks = state.arrs_dist_info[rhs].blocks
        state.arrs_dist_info[lhs].local_sizes = state.arrs_dist_info[rhs].local_sizes
        state.arrs_dist_info[lhs].leftovers = state.arrs_dist_info[rhs].leftovers

        # lhs and rhs are sequential if either is sequential
        # partitioning based on precedence, SEQ has highest precedence
        partitioning = min(state.arrs_dist_info[lhs].partitioning, state.arrs_dist_info[rhs].partitioning)
        state.arrs_dist_info[lhs].partitioning = state.arrs_dist_info[rhs].partitioning = partitioning
        @dprintln(3,"DistPass arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
        return node
        # lhs is sequential if rhs is unknown
    elseif haskey(state.arrs_dist_info, lhs)
        @dprintln(3,"DistPass assignment unknown rhs:", rhs,", sequentlial lhs: ",lhs)
        state.arrs_dist_info[lhs].partitioning = SEQ
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function get_arr_dist_info_assignment(node::Expr, state::DistPassState, top_level_number, lhs::LHSVar, rhs::Expr)
    if isAllocation(rhs)
        # if allocated already, can't parallelize since we can't replace
        #   the arraysize() calls to global value statically. needs runtime support
        alloc_sizes = map(replaceAllocTypedVar, get_alloc_shape(rhs.args[2:end]))
        if state.arrs_dist_info[lhs].dim_sizes[1]!=0 && state.arrs_dist_info[lhs].dim_sizes!=alloc_sizes
            setSEQ(lhs,state)
            @dprintln(3,"DistPass arr info non-constant allocation found sequential: ", lhs," ",rhs.args[2:end])
        end
        state.arrs_dist_info[lhs].dim_sizes = alloc_sizes
        @dprintln(3,"DistPass arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
    elseif isa(rhs,Expr) && rhs.head==:call && (isa(rhs.args[1],GlobalRef) || isa(rhs.args[1],TopNode)) && in(rhs.args[1].name, dist_ir_funcs)
        func = rhs.args[1]
        if isBaseFunc(func,:convert)
            # handle convert similar to assignment, ignore type for now
            return get_arr_dist_info_assignment(node, state, top_level_number, lhs, rhs.args[3])
        elseif isBaseFunc(func,:reshape)
            # only reshape() with constant tuples handled
            if haskey(state.tuple_table, rhs.args[3])
                state.arrs_dist_info[lhs].dim_sizes = state.tuple_table[rhs.args[3]]
                # TODO: are these required?
                #=
                state.arrs_dist_info[lhs].starts = state.arrs_dist_info[rhs.args[2]].starts
                state.arrs_dist_info[lhs].counts = state.arrs_dist_info[rhs.args[2]].counts
                state.arrs_dist_info[lhs].strides = state.arrs_dist_info[rhs.args[2]].strides
                state.arrs_dist_info[lhs].blocks = state.arrs_dist_info[rhs.args[2]].blocks
                state.arrs_dist_info[lhs].local_sizes = state.arrs_dist_info[rhs.args[2]].local_sizes
                state.arrs_dist_info[lhs].leftovers = state.arrs_dist_info[rhs.args[2]].leftovers
                =#

                @dprintln(3,"DistPass arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
                # lhs and rhs are sequential if either is sequential
                # partitioning based on precedence, SEQ has highest precedence
                partitioning = min(state.arrs_dist_info[lhs].partitioning, state.arrs_dist_info[toLHSVar(rhs.args[2])].partitioning)
                state.arrs_dist_info[lhs].partitioning = state.arrs_dist_info[toLHSVar(rhs.args[2])].partitioning = partitioning
            else
                @dprintln(3,"DistPass arr info reshape tuple not found: ", rhs.args[3]," therefore sequential: ",lhs," ",toLHSVar(rhs.args[2]))
                setSEQ(lhs,state)
                setSEQ(toLHSVar(rhs.args[2]),state)
            end
        elseif isBaseFunc(rhs.args[1],:tuple)
            state.tuple_table[lhs] = [  toLHSVarOrNum(s) for s in rhs.args[2:end] ]
            # TODO: do tuples need to be constant?
            #=
            ok = true
            for s in rhs.args[2:end]
            if !(isa(s,SymbolNode) || isa(s,Int))
            ok = false
            end
            end
            if ok
            state.tuple_table[lhs] = [  toLHSVarOrNum(s) for s in rhs.args[2:end] ]
            @dprintln(3,"DistPass arr info tuple constant: ", lhs," ",rhs.args[2:end])
            else
            @dprintln(3,"DistPass arr info tuple not constant: ", lhs," ",rhs.args[2:end])
            end
            =#
        elseif isBaseFunc(func,:gemm_wrapper!)
            return get_arr_dist_info_gemm(node, state, top_level_number, lhs, rhs)
        elseif isBaseFunc(func,:gemv!)
            return get_arr_dist_info_gemv(node, state, top_level_number, lhs, rhs)
            # TODO: Check why toLHSVar(rhs.args[2])].dim_sizes[1] is zero for all arrays
        elseif isBaseFunc(func,:cumsum!)
            @assert length(rhs.args)==3 "cumsum case not handled $rhs"
            out = toLHSVar(rhs.args[2])
            in_arr = toLHSVar(rhs.args[3])
            state.arrs_dist_info[lhs].partitioning = state.arrs_dist_info[out].partitioning = state.arrs_dist_info[in_arr].partitioning
            state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[out].dim_sizes
        elseif func.name == :hcat
            @dprintln(3,"DistPass arr info handling hcat: ", rhs)
            state.arrs_dist_info[lhs].dim_sizes[1] = state.arrs_dist_info[toLHSVar(rhs.args[2])].dim_sizes[1]
            state.arrs_dist_info[lhs].dim_sizes[2] = length(rhs.args) - 1
            min_partitioning = state.arrs_dist_info[lhs].partitioning
            for curr_array_index in 2:length(rhs.args)
                min_partitioning = min(min_partitioning, state.arrs_dist_info[toLHSVar(rhs.args[curr_array_index])].partitioning)
            end
            for curr_array_index in 2:length(rhs.args)
                state.arrs_dist_info[toLHSVar(rhs.args[curr_array_index])].partitioning = min_partitioning
            end
            state.arrs_dist_info[lhs].partitioning = min_partitioning
        elseif func.name == :typed_hcat
            # lhs = Base.typed_hcat(Int64, arr1,...)
            @dprintln(3,"DistPass arr info handling typed_hcat: ", rhs)
            state.arrs_dist_info[lhs].dim_sizes[1] = state.arrs_dist_info[toLHSVar(rhs.args[3])].dim_sizes[1]
            state.arrs_dist_info[lhs].dim_sizes[2] = length(rhs.args) - 2
            min_partitioning = state.arrs_dist_info[lhs].partitioning
            for curr_array_index in 3:length(rhs.args)
                min_partitioning = min(min_partitioning, state.arrs_dist_info[toLHSVar(rhs.args[curr_array_index])].partitioning)
            end
            for curr_array_index in 3:length(rhs.args)
                state.arrs_dist_info[toLHSVar(rhs.args[curr_array_index])].partitioning = min_partitioning
            end
            state.arrs_dist_info[lhs].partitioning = min_partitioning
        elseif func.name == :vcat
            @dprintln(3,"DistPass arr info handling vcat: ", rhs)
            # output and all input arrays have same dim_sizes except first dimension
            state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].dim_sizes
            # HACK: a call expression summing dimension 1 of all input arrays
            state.arrs_dist_info[lhs].dim_sizes[1] =
               Expr(:call,:(+), map(x->state.arrs_dist_info[x].dim_sizes[1], rhs.args[2:end])...)
            min_partitioning = state.arrs_dist_info[lhs].partitioning
            for curr_array_index in 2:length(rhs.args)
                min_partitioning = min(min_partitioning, state.arrs_dist_info[toLHSVar(rhs.args[curr_array_index])].partitioning)
            end
            for curr_array_index in 2:length(rhs.args)
                state.arrs_dist_info[toLHSVar(rhs.args[curr_array_index])].partitioning = min_partitioning
            end
            state.arrs_dist_info[lhs].partitioning = min_partitioning
        elseif isBaseFunc(func,:transpose!)
            # arg1 of transpose!() is also output and needs to be updated
            out_arg = toLHSVar(rhs.args[2])
            state.arrs_dist_info[lhs].dim_sizes[2] = state.arrs_dist_info[toLHSVar(rhs.args[3])].dim_sizes[1]
            state.arrs_dist_info[out_arg].dim_sizes[2] = state.arrs_dist_info[toLHSVar(rhs.args[3])].dim_sizes[1]
            state.arrs_dist_info[lhs].dim_sizes[1] = state.arrs_dist_info[toLHSVar(rhs.args[3])].dim_sizes[2]
            state.arrs_dist_info[out_arg].dim_sizes[1] = state.arrs_dist_info[toLHSVar(rhs.args[3])].dim_sizes[2]
            partitioning = min(state.arrs_dist_info[lhs].partitioning, state.arrs_dist_info[toLHSVar(rhs.args[3])].partitioning)
            state.arrs_dist_info[lhs].partitioning = state.arrs_dist_info[toLHSVar(rhs.args[3])].partitioning = partitioning
            state.arrs_dist_info[out_arg].partitioning = partitioning
        elseif isBaseFunc(func,:transpose)
            in_arr = toLHSVar(rhs.args[2])
            state.arrs_dist_info[lhs].dim_sizes[2] = state.arrs_dist_info[in_arr].dim_sizes[1]
            state.arrs_dist_info[lhs].dim_sizes[1] = state.arrs_dist_info[in_arr].dim_sizes[2]
            partitioning = min(state.arrs_dist_info[lhs].partitioning, state.arrs_dist_info[in_arr].partitioning)
            state.arrs_dist_info[lhs].partitioning = state.arrs_dist_info[in_arr].partitioning = partitioning
        elseif func==GlobalRef(HPAT.API, :__hpat_transpose_hcat)
            # lhs = HPAT.API.__hpat_transpose_hcat(arr1,...)
            @dprintln(3,"DistPass arr info handling transpose_hcat: ", rhs)
            # each column consists of 1 element from each input
            state.arrs_dist_info[lhs].dim_sizes[1] = length(rhs.args)-1
            # take size from one input
            state.arrs_dist_info[lhs].dim_sizes[2] = state.arrs_dist_info[toLHSVar(rhs.args[2])].dim_sizes[1]
            min_partitioning = state.arrs_dist_info[lhs].partitioning
            for curr_array_index in 2:length(rhs.args)
                min_partitioning = min(min_partitioning, state.arrs_dist_info[toLHSVar(rhs.args[curr_array_index])].partitioning)
            end
            for curr_array_index in 2:length(rhs.args)
                state.arrs_dist_info[toLHSVar(rhs.args[curr_array_index])].partitioning = min_partitioning
            end
            state.arrs_dist_info[lhs].partitioning = min_partitioning
        end
    else
        # lhs is sequential if rhs is unknown
        if haskey(state.arrs_dist_info, lhs)
            @dprintln(3,"DistPass assignment unknown rhs:", rhs,", sequentlial lhs: ",lhs)
            state.arrs_dist_info[lhs].partitioning = SEQ
        end
        return CompilerTools.AstWalker.ASTWALK_RECURSE
    end
    # Dead return
    return node
end

get_arr_dist_info_assignment(node::Expr, state::DistPassState, top_level_number, lhs::ANY, rhs::ANY) = CompilerTools.AstWalker.ASTWALK_RECURSE

function get_arr_dist_info_gemm(node::Expr, state::DistPassState, top_level_number, lhs::LHSVar, rhs::Expr)
    # determine output dimensions
    state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].dim_sizes
    state.arrs_dist_info[lhs].starts = state.arrs_dist_info[toLHSVar(rhs.args[2])].starts
    state.arrs_dist_info[lhs].counts = state.arrs_dist_info[toLHSVar(rhs.args[2])].counts
    state.arrs_dist_info[lhs].strides = state.arrs_dist_info[toLHSVar(rhs.args[2])].strides
    state.arrs_dist_info[lhs].blocks = state.arrs_dist_info[toLHSVar(rhs.args[2])].blocks
    state.arrs_dist_info[lhs].local_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].local_sizes
    state.arrs_dist_info[lhs].leftovers = state.arrs_dist_info[toLHSVar(rhs.args[2])].leftovers

    arr1 = toLHSVar(rhs.args[5])
    t1 = (rhs.args[3]=='T')
    arr2 = toLHSVar(rhs.args[6])
    t2 = (rhs.args[4]=='T')

    partitioning=ONE_D

    # result is sequential if both inputs are sequential
    if isSEQ(arr1,state) && isSEQ(arr2,state)
        partitioning=SEQ
        # result is sequential but with reduction if both inputs are partitioned and second one is transposed
        # e.g. labels*points'
    elseif isONE_D(arr1,state) && isONE_D(arr2,state) && t2 && !t1
        partitioning=SEQ
        # first input is sequential but output is parallel if the second input is partitioned but not transposed
        # e.g. w*points
    elseif isONE_D(arr2,state) && !t2 && !isTWO_D(arr1,state)
        @dprintln(3,"DistPass arr info gemm first input is sequential: ", arr1)
        setSEQ(arr1,state)
        # if no array is sequential and any array is 2D, then all are TWO_D
    elseif !isSEQ(arr1,state) && !isSEQ(arr2,state) && !isSEQ(lhs,state) &&
        ( isTWO_D(arr1,state) || isTWO_D(arr2,state) || isTWO_D(lhs,state) )
        @dprintln(3,"DistPass arr info gemm 2D: ", arr1)
        setTWO_D(arr1,state)
        setTWO_D(arr2,state)
        partitioning=TWO_D
        # otherwise, no known pattern found, every array is sequential
    else
        @dprintln(3,"DistPass arr info gemm all sequential: ", arr1," ", arr2)
        setSEQ(arr1,state)
        setSEQ(arr2,state)
        partitioning=SEQ
    end

    if partitioning==SEQ
        @dprintln(3,"DistPass arr info gemm output is sequential: ", lhs," ",rhs.args[2])
    end
    setArrayPartitioning(lhs,partitioning,state)
    setArrayPartitioning(toLHSVar(rhs.args[2]),partitioning,state)
    return node
end

function get_arr_dist_info_gemv(node::Expr, state::DistPassState, top_level_number, lhs::LHSVar, rhs::Expr)
    # determine output dimensions
    state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].dim_sizes
    state.arrs_dist_info[lhs].starts = state.arrs_dist_info[toLHSVar(rhs.args[2])].starts
    state.arrs_dist_info[lhs].counts = state.arrs_dist_info[toLHSVar(rhs.args[2])].counts
    state.arrs_dist_info[lhs].strides = state.arrs_dist_info[toLHSVar(rhs.args[2])].strides
    state.arrs_dist_info[lhs].blocks = state.arrs_dist_info[toLHSVar(rhs.args[2])].blocks
    state.arrs_dist_info[lhs].local_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].local_sizes
    state.arrs_dist_info[lhs].leftovers = state.arrs_dist_info[toLHSVar(rhs.args[2])].leftovers

    arr1 = toLHSVar(rhs.args[4])
    t1 = (rhs.args[3]=='T')
    arr2 = toLHSVar(rhs.args[5])

    partitioning = ONE_D

    # result is sequential if both inputs are sequential
    if isSEQ(arr1,state) && isSEQ(arr2,state)
        partitioning = SEQ
        # result is sequential but with reduction if both inputs are partitioned and matrix is not transposed (X*y)
    elseif !isSEQ(arr1,state) && !isSEQ(arr2,state) && !t1
        partitioning = SEQ
        # result is parallel if matrix is parallel and transposed (X'*x)
    elseif !isSEQ(arr1,state) && t1
        setSEQ(arr2,state)
        #seq = true
        # otherwise, no known pattern found, every array is sequential
    else
        @dprintln(3,"DistPass arr info gemv all sequential: ", arr1," ", arr2)
        setSEQ(arr1,state)
        setSEQ(arr2,state)
        partitioning = SEQ
    end

    if partitioning==SEQ
        @dprintln(3,"DistPass arr info gemv output is sequential: ", lhs," ",rhs.args[2])
    end
    setArrayPartitioning(lhs,partitioning,state)
    setArrayPartitioning(toLHSVar(rhs.args[2]),partitioning,state)
    return node
end

#=
function isEqualDimSize(sizes1::Array{Union{RHSVar,Int,Expr},1} , sizes2::Array{Union{RHSVar,Int,Expr},1})
    if length(sizes1)!=length(sizes2)
        return false
    end
    for i in 1:length(sizes1)
        if !eqSize(sizes1[i],sizes2[i])
            return false
        end
    end
    return true
end

function eqSize(a::Expr, b::Expr)
    if a.head!=b.head || length(a.args)!=length(b.args)
        return false
    end
    for i in 1:length(a.args)
        if !eqSize(a.args[i],b.args[i])
            return false
        end
    end
    return true
end

function eqSize(a::RHSVar, b::RHSVar)
    return toLHSVar(a) == toLHSVar(b)
end

function eqSize(a::Any, b::Any)
    return a==b
end
=#

isMeta(node::Expr) = (node.head==:meta)
isMeta(node::ANY) = false

function recreate_parfor_pre(body, linfo)
    @dprintln(3,"DistPass recreate_parfor_pre ast", linfo, body)
    lives = computeLiveness(body, linfo)
    # arrays that their allocations added to prev
    pre_alloc_arrs = LHSVar[]
    # for each basic block:
    # for each parfor, find its write array, add allocation to prestatements
    # remove meta and line numbers
    for bb in collect(values(lives.basic_blocks))
        for i in length(bb.statements):-1:1
            stmt = bb.statements[i].tls.expr
            if isBareParfor(stmt)
                parfor = stmt.args[1]
                rws = CompilerTools.ReadWriteSet.from_exprs(parfor.body, ParallelAccelerator.ParallelIR.pir_rws_cb, linfo)
                write_arrays = collect(keys(rws.writeSet.arrays))
                # fix to include reduction variable generated by 2nd gemm pattern
                # TODO: generalize rws somehow, or use liveness instead?
                #if length(parfor.preParFor)>=1 && isCall(parfor.preParFor[1]) &&
                #    parfor.preParFor[1].args[1]==GlobalRef(ParallelAccelerator.API,:set_zeros)
                #    push!(write_arrays, parfor.preParFor[1].args[2])
                #end
                @dprintln(3,"DistPass recreate_parfor_pre parfor found: ", parfor, rws, "\nwrite arrs: ",write_arrays)
                # only parfors with one write supported
                # TODO: generalize?
                if length(write_arrays)==1
                    arr = toLHSVar(write_arrays[1])
                    @dprintln(3,"DistPass recreate_parfor_pre write array: ", arr)
                    for j in i-1:-1:1
                        prev_stmt = bb.statements[j].tls.expr
                        if isAllocationAssignment(prev_stmt) && prev_stmt.args[1]==arr
                            @dprintln(3,"DistPass recreate_parfor_pre allocation for parfor found: ", prev_stmt)
                            parfor.preParFor = [prev_stmt; parfor.preParFor]
                            push!(pre_alloc_arrs, arr)
                            break
                        elseif in(arr,bb.statements[j].use) || in(arr,bb.statements[j].def)
                            break
                        end
                    end
                end
            end
        end
    end
    @dprintln(3,"DistPass recreate_parfor_pre pre_alloc_arrs: ", pre_alloc_arrs)
    out = Any[]
    for i in 1:length(body.args)
        node = body.args[i]
        if !isMeta(node) && !(isAllocationAssignment(node) && in(node.args[1], pre_alloc_arrs))
            push!(out, node)
        end
        if isBareParfor(node)
            fix_parfor_for_fusion(node.args[1], length(out), linfo)
        end
    end
    body.args = out
end

function fix_parfor_for_fusion(parfor::PIRParForAst, new_top_level_number, linfo)
    @dprintln(3,"DistPass fix_parfor_for_fusion parfor ", parfor)
    parfor.top_level_number = [new_top_level_number]
    empty!(parfor.array_aliases)
    # remove array assignment from post statements generated by expanding gemm
    if length(parfor.postParFor)==2 && isa(parfor.postParFor[1],Expr) &&
         parfor.postParFor[1].head==:(=) && isa(parfor.postParFor[1].args[2],LHSVar)
        @assert length(parfor.preParFor)>=1 && isAllocationAssignment(parfor.preParFor[1]) "invalid parfor for expanded gemm matched"
        @dprintln(3,"DistPass fix_parfor_for_fusion remove assignment from expanded gemm ")
        lhs_var = parfor.postParFor[1].args[1]
        out_var = parfor.postParFor[1].args[2]
        replaceExprWithDict!(parfor, Dict{LHSVar,Any}(out_var=>lhs_var), linfo, ParallelIR.AstWalk)
        parfor.postParFor = Any[0]
    end
    # use rws to update first_input, which is used for finding correlation in fusion
    # reverse order to match access array
    parfor_indices = [ toLHSVar(parfor.loopNests[i].indexVariable) for i in length(parfor.loopNests):-1:1 ]
    rws = CompilerTools.ReadWriteSet.from_exprs(parfor.body, ParallelAccelerator.ParallelIR.pir_rws_cb, linfo)
    @dprintln(3,"DistPass fix_parfor_for_fusion parfor indices ", parfor_indices)
    @dprintln(3,"DistPass fix_parfor_for_fusion rws arrays ", union(rws.readSet.arrays, rws.writeSet.arrays))

    for (arr,inds) in union(rws.readSet.arrays, rws.writeSet.arrays)
        # TODO: is this sufficient condition for parfor/array correlation?
        indices = map(x->isa(x,Colon)?x:toLHSVar(x), inds[1])
        if indices==parfor_indices
            @dprintln(3,"DistPass fix_parfor_for_fusion updating first_input.array from ",
                 parfor.first_input.array, " to ", arr)
            parfor.first_input.array = arr
            break
        end
    end
end

function dist_optimize(body::Expr, state::DistPassState)
    @assert body.head==:body "invalid body in dist_optimize"
    out_body = Any[]
    for i in 1:length(body.args)
        new_node = dist_optimize_node(body.args[i], i, state)
        push!(out_body, new_node)
    end
    body.args = out_body
    recreate_parfor_pre(body, state.LambdaVarInfo)
    state.LambdaVarInfo, body = ParallelAccelerator.ParallelIR.fusion_pass("dist_opt", state.LambdaVarInfo, body)
    return body
end

function dist_optimize_node(node::Expr, top_level_number, state)
    if node.head==:(=)
        #@dprintln(3,"DistPass optimize assignment: ", node)
        lhs = toLHSVar(node.args[1])
        rhs = node.args[2]
        return dist_optimize_assignment(node, state, top_level_number, lhs, rhs)
    elseif node.head==:parfor
        parfor = node.args[1]
        new_body = dist_optimize(Expr(:body, parfor.body...), state)
        parfor.body = new_body.args
    end
    return node
end

function dist_optimize_node(node::ANY, top_level_number, state)
    return node
end

function dist_optimize_assignment(node::Expr, state::DistPassState, top_level_number, lhs::LHSVar, rhs::RHSVar)
    return node
end

function dist_optimize_assignment(node::Expr, state::DistPassState, top_level_number, lhs::LHSVar, rhs::Expr)
    if rhs.head==:call && isBaseFunc(rhs.args[1],:gemm_wrapper!)
        @dprintln(3,"DistPass optimize gemm found: ", node)
        out = toLHSVar(rhs.args[2])
        arr1 = toLHSVar(rhs.args[5])
        t1 = (rhs.args[3]=='T')
        arr2 = toLHSVar(rhs.args[6])
        t2 = (rhs.args[4]=='T')
        # weight multipied by samples (e.g. w*points)
        if isSEQ(arr1,state) && isONE_D(arr2,state) && !t1 && !t2
            @dprintln(3,"DistPass optimize weight times points pattern found")
            return expand_gemm_sp(lhs, out, arr1, arr2, top_level_number, state)
        # labels multipied by samples (e.g. labels*points')
        elseif isSEQ(out,state) && isONE_D(arr1,state) && isONE_D(arr2,state) && !t1 && t2
            @dprintln(3,"DistPass optimize labels times points transpose pattern found")
            return expand_gemm_pp(lhs, out, arr1, arr2, top_level_number, state)
        end
    end
    return node
end

function dist_optimize_assignment(node::Expr, state::DistPassState, top_level_number, lhs::ANY, rhs::ANY)
    return node
end

function expand_gemm_sp(lhs, out, arr1, arr2, top_level_number, state)
    size1 = state.arrs_dist_info[arr2].dim_sizes[end]
    size2 = state.arrs_dist_info[arr1].dim_sizes[1]
    size3 = state.arrs_dist_info[arr1].dim_sizes[end]
    # outer loop over samples, inner loop over functions
    parfor_index1 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_parfor_"*string(getDistNewID(state))*"_index1"), Int, ISASSIGNED,state.LambdaVarInfo))
    parfor_index2 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_parfor_"*string(getDistNewID(state))*"_index2"), Int, ISASSIGNED,state.LambdaVarInfo))
    loop_index = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_loop_"*string(getDistNewID(state))*"_index"), Int, ISASSIGNED,state.LambdaVarInfo))
    elem_typ = eltype(CompilerTools.LambdaHandling.getType(arr1, state.LambdaVarInfo))
    temp_var1 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_gemm_tmp1_"*string(getDistNewID(state))), elem_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo))
    temp_var2 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_gemm_tmp2_"*string(getDistNewID(state))), elem_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo))
    temp_var3 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_gemm_tmp3_"*string(getDistNewID(state))), elem_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo))

    loopNests = PIRLoopNest[ PIRLoopNest(parfor_index1, 1, size1, 1), PIRLoopNest(parfor_index2, 1, size2, 1) ]
    parfor_id = getDistNewID(state)
    first_input_info = InputInfo(out)
    first_input_info.dim = 2
    #first_input_info.indexed_dims = ones(Int64, first_input_info.dim)
    first_input_info.indexed_dims = [true,true] # loop over last dimension
    first_input_info.out_dim = 2
    first_input_info.elementTemp = temp_var2
    out_body = Any[]
    pre_statements  = Any[]
    post_statements = Any[ Expr(:(=), lhs, out), 0 ]

    push!(out_body, Expr(:(=), temp_var3, 0))
    # inner loop k dimension
    push!(out_body, Expr(:loophead, loop_index, 1, size3))
    # tmp1 = w[j,k]
    push!(out_body, Expr(:(=), temp_var1, mk_call(GlobalRef(Base,:unsafe_arrayref),[arr1, parfor_index2, loop_index])))
    # tmp2 = points[k,i]
    push!(out_body, Expr(:(=), temp_var2, mk_call(GlobalRef(Base,:unsafe_arrayref),[arr2, loop_index, parfor_index1])))
    push!(out_body, Expr(:(=), temp_var3, mk_add_float_expr(temp_var3, mk_mult_float_expr(temp_var1,temp_var2))))
    push!(out_body, Expr(:loopend, loop_index))
    push!(out_body, mk_call(GlobalRef(Base,:unsafe_arrayset),[out, temp_var3, parfor_index2, parfor_index1]))

    new_parfor = ParallelAccelerator.ParallelIR.PIRParForAst(
        first_input_info,
        out_body,
        pre_statements,
        loopNests,
        PIRReduction[],
        post_statements,
        [ParallelAccelerator.ParallelIR.DomainOperation(:mmap!,Any[])], # empty domain_oprs
        top_level_number,
        parfor_id,
        Set{LHSVar}(), #arrays_written_past_index
        Set{LHSVar}()) #arrays_read_past_index
    @dprintln(3,"DistPass optimize new_parfor ", new_parfor)
    state.parfor_partitioning[parfor_id] = ONE_D
    state.parfor_arrays[parfor_id] = [lhs,arr2]
    return Expr(:parfor, new_parfor)
end

function expand_gemm_pp(lhs, out, arr1, arr2, top_level_number, state)
    size1 = state.arrs_dist_info[arr2].dim_sizes[end]
    size2 = state.arrs_dist_info[arr1].dim_sizes[1]
    size3 = state.arrs_dist_info[out].dim_sizes[end]
    # outer loop over samples, inner loop over functions
    parfor_index1 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_parfor_"*string(getDistNewID(state))*"_index1"), Int, ISASSIGNED,state.LambdaVarInfo))
    parfor_index2 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_parfor_"*string(getDistNewID(state))*"_index2"), Int, ISASSIGNED,state.LambdaVarInfo))
    loop_index = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_loop_"*string(getDistNewID(state))*"_index"), Int, ISASSIGNED,state.LambdaVarInfo))
    elem_typ = eltype(CompilerTools.LambdaHandling.getType(arr1, state.LambdaVarInfo))
    temp_var1 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_gemm_tmp1_"*string(getDistNewID(state))), elem_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo))
    temp_var2 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_gemm_tmp2_"*string(getDistNewID(state))), elem_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo))
    temp_var3 = toLHSVar(CompilerTools.LambdaHandling.addLocalVariable(
        Symbol("_dist_gemm_tmp3_"*string(getDistNewID(state))), elem_typ, ISASSIGNED | ISPRIVATEPARFORLOOP, state.LambdaVarInfo))

    loopNests = PIRLoopNest[ PIRLoopNest(parfor_index1, 1, size1, 1), PIRLoopNest(parfor_index2, 1, size2, 1) ]

    parfor_id = getDistNewID(state)
    first_input_info = InputInfo(arr2)
    first_input_info.dim = 2
    #first_input_info.indexed_dims = ones(Int64, first_input_info.dim)
    first_input_info.indexed_dims = [true,false] # loop over last dimension
    first_input_info.out_dim = 1
    first_input_info.elementTemp = temp_var2
    out_body = Any[]
    pre_statements  = Any[ mk_call(GlobalRef(ParallelAccelerator.API,:set_zeros),[out]) ]
    post_statements = Any[ Expr(:(=), lhs, out), 0 ]

    # tmp2 = labels[j,i]
    push!(out_body, Expr(:(=), temp_var2, mk_call(GlobalRef(Base,:unsafe_arrayref),[arr1, parfor_index2, parfor_index1])))
    # inner loop k dimension
    push!(out_body, Expr(:loophead, loop_index, 1, size3))
    # tmp1 = points[k,i]
    push!(out_body, Expr(:(=), temp_var1, mk_call(GlobalRef(Base,:unsafe_arrayref),[arr2, loop_index, parfor_index1])))
    # tmp3 = w[j,w]
    push!(out_body, Expr(:(=), temp_var3, mk_call(GlobalRef(Base,:unsafe_arrayref),[out, parfor_index2, loop_index])))
    push!(out_body, Expr(:(=), temp_var3, mk_add_float_expr(temp_var3, mk_mult_float_expr(temp_var1,temp_var2))))
    push!(out_body, mk_call(GlobalRef(Base,:unsafe_arrayset),[out, temp_var3, parfor_index2, loop_index]))
    push!(out_body, Expr(:loopend, loop_index))

    new_parfor = ParallelAccelerator.ParallelIR.PIRParForAst(
        first_input_info,
        out_body,
        pre_statements,
        loopNests,
        [PIRReduction(out, 0, GlobalRef(Base,:(+)))],
        post_statements,
        [ParallelAccelerator.ParallelIR.DomainOperation(:mmap!,Any[])], # empty domain_oprs
        top_level_number,
        parfor_id,
        Set{LHSVar}(), #arrays_written_past_index
        Set{LHSVar}()) #arrays_read_past_index
    @dprintln(3,"DistPass optimize new_parfor ", new_parfor)
    state.parfor_partitioning[parfor_id] = ONE_D
    state.parfor_arrays[parfor_id] = [lhs,arr2]
    return Expr(:parfor, new_parfor)
end
