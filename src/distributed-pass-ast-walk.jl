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
    before_dist_arrays = [arr for arr in keys(state.arrs_dist_info)]
    
    while true
        dist_arrays = []
        @dprintln(3,"DistPass state before array info walk: ",state)
        ParallelIR.AstWalk(ast, get_arr_dist_info, state)
        @dprintln(3,"DistPass state after array info walk: ",state)
            # all arrays not marked sequential are distributable at this point 
        for arr in keys(state.arrs_dist_info)
            if state.arrs_dist_info[arr].isSequential==false
                @dprintln(2,"DistPass distributable parfor array: ", arr)
                push!(dist_arrays,arr)
            end
        end
        # break if no new sequential array discovered
        if length(dist_arrays)==length(before_dist_arrays)
            break
        end
        before_dist_arrays = dist_arrays
    end
    state.dist_arrays = before_dist_arrays
    @dprintln(3,"DistPass state dist_arrays after array info walk: ",state.dist_arrays)
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
        rws = parfor.rws
        seq = false
                
        if length(parfor.arrays_read_past_index)!=0 || length(parfor.arrays_written_past_index)!=0 
            @dprintln(2,"DistPass arr info walk parfor sequential: ", node)
            seq = true
        end
        
        indexVariable::Symbol = toLHSVar(parfor.loopNests[1].indexVariable)
        
        allArrAccesses = merge(rws.readSet.arrays,rws.writeSet.arrays)
        myArrs = LHSVar[]
        
        body_lives = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo, parfor.body, ParallelIR.pir_live_cb, state.LambdaVarInfo)
        #@dprintln(3, "body_lives = ", body_lives)

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
                    seq = true
                end
                # sequential if not accessed column major (last dimension)
                # TODO: generalize?
                if in(indexVariable, indices[1:end-1])
                    @dprintln(2,"DistPass arr info walk arr index sequential: ",arr," ", indices, " ", indexVariable)
                    seq = true
                end
            end
        end

        # keep mapping from parfors to arrays
        state.parfor_info[parfor.unique_id] = myArrs
        @dprintln(3,"DistPass arr info walk parfor arrays: ", myArrs)

        for arr in myArrs
            if state.arrs_dist_info[arr].isSequential ||
                        !eqSize(state.arrs_dist_info[arr].dim_sizes[end], state.arrs_dist_info[myArrs[1]].dim_sizes[end]) 
                    # last dimension of all parfor arrays should be equal since they are partitioned
                    @dprintln(2,"DistPass parfor check array: ", arr," seq: ", state.arrs_dist_info[arr].isSequential)
                    seq = true
            end
        end
        # parfor and all its arrays are sequential
        if seq
            push!(state.seq_parfors, parfor.unique_id)
            for arr in myArrs
                state.arrs_dist_info[arr].isSequential = true
            end
        end
        return CompilerTools.AstWalker.ASTWALK_RECURSE
    # functions dist_ir_funcs are either handled here or do not make arrays sequential  
    elseif head==:call && in(node.args[1].name, dist_ir_funcs)
        func = node.args[1].name
        if func==:__hpat_data_source_HDF5_read || func==:__hpat_data_source_TXT_read
            @dprintln(2,"DistPass arr info walk data source read ", node)
            # will be parallel IO, intentionally do nothing
        elseif func==:Kmeans
            @dprintln(2,"DistPass arr info walk kmeans ", node)
            # first array is cluster output and is sequential
            # second array is input matrix and is parallel
            state.arrs_dist_info[toLHSVar(node.args[2])].isSequential = true
        elseif func==:LinearRegression || func==:NaiveBayes
            @dprintln(2,"DistPass arr info walk LinearRegression/NaiveBayes ", node)
            # first array is cluster output and is sequential
            # second array is input matrix and is parallel
            # third array is responses and is parallel
            state.arrs_dist_info[toLHSVar(node.args[2])].isSequential = true
        end
        return node
    elseif head==:gotoifnot
        @dprintln(3,"DistPass arr info gotoifnot: ", node)
        return CompilerTools.AstWalker.ASTWALK_RECURSE
    # arrays written in sequential code are not distributed
    elseif head!=:body && head!=:block && head!=:lambda
        @dprintln(3,"DistPass arr info walk sequential code: ", node)
        
        live_info = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo, TypedExpr(nothing, :body, node), ParallelIR.pir_live_cb, state.LambdaVarInfo)
        #@dprintln(3, "body_lives = ", body_lives)
        # live_info = CompilerTools.LivenessAnalysis.find_top_number(top_level_number, state.lives)
        # all_vars = union(live_info.def, live_info.use)
        all_vars = []
        
        for bb in collect(values(live_info.basic_blocks))
            for stmt in bb.statements
                append!(all_vars, collect(union(stmt.def, stmt.use)))
            end
        end
        
        @dprintln(3,"DistPass arr info walk sequential code vars: ", all_vars)
        # ReadWriteSet is not robust enough now
        #rws = CompilerTools.ReadWriteSet.from_exprs([node], ParallelIR.pir_live_cb, state.LambdaVarInfo)
        #readArrs = collect(keys(rws.readSet.arrays))
        #writeArrs = collect(keys(rws.writeSet.arrays))
        #allArrs = [readArrs;writeArrs]
        
        for var in all_vars
            if haskey(state.arrs_dist_info, toLHSVar(var))
                @dprintln(2,"DistPass arr info walk array in sequential code: ", var, " ", node)
                
                state.arrs_dist_info[toLHSVar(var)].isSequential = true
            end
        end
        return node
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function isAccessIndexDependent(indices::Vector{Any}, indexVariable::Symbol, body_lives::BlockLiveness, state)
    # if any index is dependent
    return reduce(|, [ isAccessIndexDependent(indices[i], indexVariable, body_lives, state) for i in 1:length(indices)] )
end

"""
For each statement of parfor's body, see if array access index is in Def set and parfor's indexVariable is in Use set.
This is a hack to work around not having dependence analysis in CompilerTools.
"""
function isAccessIndexDependent(index::LHSVar, indexVariable::Symbol, body_lives::BlockLiveness, state)
    for bb in collect(values(body_lives.basic_blocks))
        for stmt in bb.statements
            if isBareParfor(stmt.tls.expr)
                #inner_fake_body = CompilerTools.LambdaHandling.LambdaVarInfoToLambdaExpr(state.LambdaVarInfo, TypedExpr(nothing, :body, getParforNode(stmt.tls.expr).body...))
                #@dprintln(3,"inner fake_body = ", fake_body)

                inner_body_lives = CompilerTools.LivenessAnalysis.from_lambda(state.LambdaVarInfo, TypedExpr(nothing, :body, getParforNode(stmt.tls.expr).body...), ParallelIR.pir_live_cb, state.LambdaVarInfo)
                #@dprintln(3, "inner body_lives = ", body_lives)
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


function get_arr_dist_info_assignment(node::Expr, state::DistPassState, top_level_number, lhs, rhs)
    if isAllocation(rhs)
            state.arrs_dist_info[lhs].dim_sizes = map(toLHSVarOrInt, get_alloc_shape(rhs.args[2:end]))
            @dprintln(3,"DistPass arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
    elseif isa(rhs,RHSVar)
        rhs = toLHSVar(rhs)
        if haskey(state.arrs_dist_info, rhs)
            state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[rhs].dim_sizes
            # lhs and rhs are sequential if either is sequential
            seq = state.arrs_dist_info[lhs].isSequential || state.arrs_dist_info[rhs].isSequential
            state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[rhs].isSequential = seq
            @dprintln(3,"DistPass arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
        end
    elseif isa(rhs,Expr) && rhs.head==:call && in(rhs.args[1].name, dist_ir_funcs)
        func = rhs.args[1]
        if isBaseFunc(func,:reshape)
            # only reshape() with constant tuples handled
            if haskey(state.tuple_table, rhs.args[3])
                state.arrs_dist_info[lhs].dim_sizes = state.tuple_table[rhs.args[3]]
                @dprintln(3,"DistPass arr info dim_sizes update: ", state.arrs_dist_info[lhs].dim_sizes)
                # lhs and rhs are sequential if either is sequential
                seq = state.arrs_dist_info[lhs].isSequential || state.arrs_dist_info[toLHSVar(rhs.args[2])].isSequential
                state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[toLHSVar(rhs.args[2])].isSequential = seq
            else
                @dprintln(3,"DistPass arr info reshape tuple not found: ", rhs.args[3])
                state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[toLHSVar(rhs.args[2])].isSequential = true
            end
        elseif isBaseFunc(rhs.args[1],:tuple)
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
        elseif isBaseFunc(func,:gemm_wrapper!)
            # determine output dimensions
            state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].dim_sizes
            arr1 = toLHSVar(rhs.args[5])
            t1 = (rhs.args[3]=='T')
            arr2 = toLHSVar(rhs.args[6])
            t2 = (rhs.args[4]=='T')
            
            seq = false
            
            # result is sequential if both inputs are sequential 
            if state.arrs_dist_info[arr1].isSequential && state.arrs_dist_info[arr2].isSequential
                seq = true
            # result is sequential but with reduction if both inputs are partitioned and second one is transposed
            # e.g. labels*points'
            elseif !state.arrs_dist_info[arr1].isSequential && !state.arrs_dist_info[arr2].isSequential && t2 && !t1
                seq = true
            # first input is sequential but output is parallel if the second input is partitioned but not transposed
            # e.g. w*points
            elseif !state.arrs_dist_info[arr2].isSequential && !t2
                @dprintln(3,"DistPass arr info gemm first input is sequential: ", arr1)
                state.arrs_dist_info[arr1].isSequential = true
            # otherwise, no known pattern found, every array is sequential
            else
                @dprintln(3,"DistPass arr info gemm all sequential: ", arr1," ", arr2)
                state.arrs_dist_info[arr1].isSequential = true
                state.arrs_dist_info[arr2].isSequential = true
                seq = true
            end
            
            if seq
                @dprintln(3,"DistPass arr info gemm output is sequential: ", lhs," ",rhs.args[2])
            end
            state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[toLHSVar(rhs.args[2])].isSequential = seq
        elseif isBaseFunc(func,:gemv!)
            # determine output dimensions
            state.arrs_dist_info[lhs].dim_sizes = state.arrs_dist_info[toLHSVar(rhs.args[2])].dim_sizes
            arr1 = toLHSVar(rhs.args[4])
            t1 = (rhs.args[3]=='T')
            arr2 = toLHSVar(rhs.args[5])
            
            seq = false
            
            # result is sequential if both inputs are sequential 
            if state.arrs_dist_info[arr1].isSequential && state.arrs_dist_info[arr2].isSequential
                seq = true
            # result is sequential but with reduction if both inputs are partitioned and matrix is not transposed
            elseif !state.arrs_dist_info[arr1].isSequential && !state.arrs_dist_info[arr2].isSequential && !t1
                seq = true
            # result and vector are sequential if matrix is parallel and transposed 
            elseif !state.arrs_dist_info[arr1].isSequential && t1
                state.arrs_dist_info[arr2].isSequential = true
                seq = true
            # otherwise, no known pattern found, every array is sequential
            else
                @dprintln(3,"DistPass arr info gemv all sequential: ", arr1," ", arr2)
                state.arrs_dist_info[arr1].isSequential = true
                state.arrs_dist_info[arr2].isSequential = true
                seq = true
            end
            
            if seq
                @dprintln(3,"DistPass arr info gemv output is sequential: ", lhs," ",rhs.args[2])
            end
            state.arrs_dist_info[lhs].isSequential = state.arrs_dist_info[toLHSVar(rhs.args[2])].isSequential = seq
        end
    else
        return CompilerTools.AstWalker.ASTWALK_RECURSE
    end
    return node
end


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

function eqSize(a::SymbolNode, b::SymbolNode)
    return a.name == b.name
end

function eqSize(a::Any, b::Any)
    return a==b
end

