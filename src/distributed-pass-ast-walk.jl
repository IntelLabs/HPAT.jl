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
end

function set_user_partitionings(state)
    for (arr,part) in state.user_partitionings
      state.arrs_dist_info[arr].partitioning = part
    end
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
            @dprintln(2,"DistPass arr info walk array sequential since in serial code: ", var, " ", node)
            setSEQ(toLHSVar(var),state)
        end
    end
    return node
end

function get_arr_dist_info_parfor(node, state, top_level_number, parfor)
    rws = parfor.rws
    partitioning = ONE_D

    if length(parfor.arrays_read_past_index)!=0 || length(parfor.arrays_written_past_index)!=0
        @dprintln(2,"DistPass arr info walk parfor sequential: ", node)
        partitioning = SEQ
    end

    indexVariable = toLHSVar(parfor.loopNests[1].indexVariable)

    allArrAccesses = merge(rws.readSet.arrays,rws.writeSet.arrays)
    myArrs = LHSVar[]

    body_lives = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo,
                                                            parfor.body, ParallelIR.pir_live_cb, state.LambdaVarInfo)
    # @dprintln(3, "body_lives = ", body_lives)

    # If an array is accessed with a Parfor's index variable, the parfor and array should have same partitioning
    for arr in keys(allArrAccesses)
        # an array can be accessed multiple times in Pafor
        # for each access:
        for access_indices in allArrAccesses[arr]
            indices = map(toLHSVar,access_indices)
            # if array would be accessed in parallel in this Parfor
            if indices[end]==indexVariable
                push!(myArrs, arr)
            end
            # An array access index can be dependent on parfor's
            # index variable as in nested comprehension case of K-Means.
            # Parfor can't be parallelized in general cases since array can't be partitioned properly.
            # ParallelIR should optimize out the trivial cases where indices are essentially equal (i=1+1*index-1 in k-means)
            if isAccessIndexDependent(indices, indexVariable, body_lives, state)
                #push!(myArrs, arr)
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
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function isAccessIndexDependent(indices::Vector{Any}, indexVariable::LHSVar, body_lives::BlockLiveness, state)
    # if any index is dependent
    return reduce(|, [ isAccessIndexDependent(indices[i], indexVariable, body_lives, state) for i in 1:length(indices)] )
end

"""
For each statement of parfor's body, see if array access index is in Def set and parfor's indexVariable is in Use set.
This is a hack to work around not having dependence analysis in CompilerTools.
"""
function isAccessIndexDependent(index::LHSVar, indexVariable::LHSVar, body_lives::BlockLiveness, state)
    for bb in collect(values(body_lives.basic_blocks))
        for stmt in bb.statements
            if isBareParfor(stmt.tls.expr)
                # inner_fake_body = CompilerTools.LambdaHandling.LambdaVarInfoToLambdaExpr(state.LambdaVarInfo, TypedExpr(nothing, :body, getParforNode(stmt.tls.expr).body...))
                # @dprintln(3,"inner fake_body = ", fake_body)

                inner_body_lives = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo,
                                                                              TypedExpr(nothing, :body, getParforNode(stmt.tls.expr).body...),
                                                                              ParallelIR.pir_live_cb, state.LambdaVarInfo)
                # @dprintln(3, "inner body_lives = ", body_lives)
                return isAccessIndexDependent(index, indexVariable, inner_body_lives, state)
            elseif in(index,stmt.def) && in(indexVariable, stmt.use)
                @dprintln(2,"DistPass arr info walk dependent index found: ", index," ",indexVariable,"  ",stmt)
                return true
            end
        end
    end
    return false
end

function get_arr_dist_info(ast::Any, state::DistPassState, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
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
        if isBaseFunc(func,:reshape)
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
        elseif func.name == :hcat
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
