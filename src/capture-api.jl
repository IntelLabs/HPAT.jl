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
        return process_assignment(node, node.args[1], node.args[2])
    end
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_node(node::Any, state, top_level_number, is_top_level, read)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, lhs::Symbol, rhs::Expr)
    if rhs.head ==:call && rhs.args[1]==:DataSource

        arr_var_expr = rhs.args[2]
        @assert arr_var_expr.head==:curly "curly syntax expected for DataSource"
        if arr_var_expr.args[1]==:DataTable
            return translate_data_table(lhs, arr_var_expr, rhs.args[3], rhs.args[4:end])
        else
            return translate_data_source(lhs, arr_var_expr, rhs.args[3], rhs.args[4:end])
        end
   end
   CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, lhs::ANY, rhs::ANY)
end

function translate_data_source(lhs, arr_var_expr, source_typ, other_args)
    @assert arr_var_expr.args[1]==:Array || arr_var_expr.args[1]==:Matrix || arr_var_expr.args[1]==:Vector "Data sources need Vector or Array or Matrix as type"
    @assert source_typ==:HDF5 || source_typ==:TXT "Only HDF5 and TXT (text) data sources supported for now."
    
    # desugar call
    call_name = symbol("data_source_$source_typ")
    # GlobalRef since Julia doesn't resolve the module!
    api_call = GlobalRef(HPAT.API, call_name)
    rhs = Expr(:call, api_call, arr_var_expr, other_args...)
    return Expr(:(=), lhs, rhs)
end

function translate_data_table(lhs, arr_var_expr, source_typ, other_args)
    @assert arr_var_expr.args[1]==:DataTable "expected :DataTable"
    out = []
    for column in arr_var_expr.args[2:end]
        @assert column.head==:(=)
        col_name = column.args[1].value
        col_type = column.args[2]
        col_lhs = symbol("_$(lhs)_$(col_name)")
        col_source = translate_data_source(col_lhs, :(Vector{$(col_type)}), source_typ, other_args)
        push!(out, col_source)
    end
    return quote $(out...) end
end

end # module
