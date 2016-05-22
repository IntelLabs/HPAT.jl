module CaptureAPI

using CompilerTools
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
        
   end
   CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, state, lhs::ANY, rhs::ANY)
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
        col_source = translate_data_source(col_lhs, :(Vector{$(col_type)}), source_typ, other_args)
        push!(out, col_source)
    end
    state[lhs] = col_names
    return quote $(out...) end
end

function getColName(t::Symbol, c::Symbol)
    return symbol("_$(t)_$(c)")
end

end # module
