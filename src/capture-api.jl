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
        process_assignment(node, node.args[1], node.args[2])
    end
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_node(node::Any, state, top_level_number, is_top_level, read)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, lhs::Symbol, rhs::Expr)
    if rhs.head ==:call && rhs.args[1]==:DataSource
        arr_var_expr = rhs.args[2]
        
        @assert arr_var_expr.args[1]==:Array || arr_var_expr.args[1]==:Matrix || arr_var_expr.args[1]==:Vector "Data sources need Vector or Array or Matrix as type"

        source_typ = rhs.args[3]
        @assert source_typ==:HDF5 || source_typ==:TXT "Only HDF5 and TXT (text) data sources supported for now."
        # desugar call
        call_name = symbol("data_source_$source_typ")
        # GlobalRef since Julia doesn't resolve the module!
        rhs.args[1] = GlobalRef(HPAT.API, call_name)
        # remove the source type arg
        splice!(rhs.args, 3) 

   end
end

function process_assignment(node, lhs::ANY, rhs::ANY)
end

end # module
