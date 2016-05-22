module CaptureAPI

using CompilerTools
using CompilerTools.AstWalker
import ..API
import HPAT


"""
At macro level, translate DataSource into function calls so that type inference
and ParallelAccelerator compilation works with knowledge of calls and allocations for arrays. 
"""
function process_node(node::Expr, state, top_level_number, is_top_level, read)
    if node.head == :(=) 
        return process_assignment(node, state, node.args[1], node.args[2])
    end
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_node(node::Any, state, top_level_number, is_top_level, read)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, state, lhs::Symbol, rhs::Expr)
    if rhs.head ==:call && rhs.args[1]==:DataSource
        arr_var_expr = rhs.args[2]
        @assert arr_var_expr.head==:curly "curly syntax expected for DataSource"
        if arr_var_expr.args[1]==:DataTable
            return translate_data_table(lhs, state, arr_var_expr, rhs.args[3], rhs.args[4:end])
        else
            return translate_data_source(lhs, state, arr_var_expr, rhs.args[3], rhs.args[4:end])
        end
   elseif rhs.head==:call && rhs.args[1]==:join
        t1 = rhs.args[2]
        t2 = rhs.args[3]
        @assert rhs.args[4].head==:comparison "invalid join key"
        @assert rhs.args[4].args[2]==:(==) "invalid join key"
        
        key1 = rhs.args[4].args[1].value
        key1_arr = getColName(t1, key1)
        key2 = rhs.args[4].args[3].value
        key2_arr = getColName(t2, key2)
        new_key = rhs.args[5].value
        new_key_arr = getColName(lhs, new_key)
        
        rest_cols1 = filter(x->x!=key1, state[t1])
        rest_cols2 = filter(x->x!=key2, state[t2])
        rest_cols1_arrs = map(x->getColName(t1,x),rest_cols1)
        rest_cols2_arrs = map(x->getColName(t2,x),rest_cols2)
        rest_cols3_arrs = map(x->getColName(lhs,x),[rest_cols1;rest_cols2])
        state[lhs] = [new_key;rest_cols1;rest_cols2]
        return :( ($new_key_arr,$(rest_cols3_arrs...)) = join([$key1_arr;$(rest_cols1_arrs...)], [$key2_arr;$(rest_cols2_arrs...)]) )
        
   elseif rhs.head==:call && rhs.args[1]==:aggregate
        t1 = rhs.args[2]
        c1 = rhs.args[3].value
        c1_arr = getColName(t1, c1)
        c1_out_arr = getColName(lhs, c1)
        out_e = []
        out_aggs = []
        out_arrs = [c1_out_arr]
        out_cols = [c1]
        for col_expr in rhs.args[4:end]
            @assert col_expr.head==:kw "expected assignment for new aggregate column"
            # output column name
            out_col = col_expr.args[1].value
            out_col_arr = getColName(lhs, out_col)
            push!(out_cols, out_col)
            push!(out_arrs, out_col_arr)
            @assert col_expr.args[2].head==:call "expected aggregation function"
            # aggregation function
            func = col_expr.args[2].args[1]
            # aggregation expression
            e = col_expr.args[2].args[2]
            # replace column name with actual array in expression
            AstWalk(e, replace_col_with_array,  (t1, state[t1]))
            out_e_arr = symbol("_$(lhs)_$(out_col)_e")
            push!(out_aggs, :(($out_e_arr, $func)))
            push!(out_e,:($out_e_arr=$e))
        end
        out_call = :($(out_arrs...) = aggregate($c1_arr,[$(out_aggs...)]))
        push!(out_e, out_call)
        state[lhs] = out_cols
        return quote $(out_e...) end
   end
   CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, state, lhs::ANY, rhs::ANY)
end

function replace_col_with_array(node::QuoteNode, table::Tuple{Symbol,Vector{Symbol}}, top_level_number, is_top_level, read)
    if node.value in table[2]
        return getColName(table[1], node.value)
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end
function replace_col_with_array(node::ANY, table::Tuple{Symbol,Vector{Symbol}}, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function translate_data_source(lhs, state, arr_var_expr, source_typ, other_args)
    @assert arr_var_expr.args[1]==:Array || arr_var_expr.args[1]==:Matrix || arr_var_expr.args[1]==:Vector "Data sources need Vector or Array or Matrix as type"
    @assert source_typ==:HDF5 || source_typ==:TXT "Only HDF5 and TXT (text) data sources supported for now."

    # desugar call
    call_name = symbol("data_source_$source_typ")
    # GlobalRef since Julia doesn't resolve the module!
    api_call = GlobalRef(HPAT.API, call_name)
    rhs = Expr(:call, api_call, arr_var_expr, other_args...)
    return Expr(:(=), lhs, rhs)
end

function translate_data_table(lhs, state, arr_var_expr, source_typ, other_args)
    @assert arr_var_expr.args[1]==:DataTable "expected :DataTable"
    out = []
    col_names = Symbol[]
    for column in arr_var_expr.args[2:end]
        @assert column.head==:(=)
        col_name = column.args[1].value
        push!(col_names, col_name)
        col_type = column.args[2]
        col_lhs = getColName(lhs,col_name)
        col_source = translate_data_source(col_lhs, state, :(Vector{$(col_type)}), source_typ, other_args)
        push!(out, col_source)
    end
    state[lhs] = col_names
    return quote $(out...) end
end

function getColName(t::Symbol, c::Symbol)
    return symbol("_$(t)_$(c)")
end

end # module
