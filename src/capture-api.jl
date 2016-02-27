module CaptureAPI

using CompilerTools

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
        arr_var_expr = node.args[2].args[2]
        
        @assert arr_var_expr.args[1]==:Array || arr_var_expr.args[1]==:Matrix || arr_var_expr.args[1]==:Vector "Data sources need Vector or Array or Matrix as type"
        
        if arr_var_expr.args[1]==:Matrix
            dims = 2 # Matrix type is 2D array
        elseif arr_var_expr.args[1]==:Vector
            dims = 1
        elseif arr_var_expr.args[1]==:Array
            dims = arr_var_expr.args[3]
        end

        node.args[1] = :($(node.args[1])::$arr_var_expr)
        source_typ = node.args[2].args[3]
        @assert source_typ==:HDF5 || source_typ==:TXT "Only HDF5 and TXT (text) data sources supported for now."
        call_name = symbol("__hpat_data_source_$source_typ")
        
        call = Expr(:call)
        
        if source_typ==:HDF5
            hdf_var_name = node.args[2].args[4]
            hdf_file_name = node.args[2].args[5]
            call = :($(call_name)($hdf_var_name,$hdf_file_name))
        else
            txt_file_name = node.args[2].args[4]
            call = :($(call_name)($txt_file_name))
        end
        
        node.args[2] = call
#=        arr_var_expr = node.args[2].args[2]
        dims = arr_var_expr.args[3]
        @assert arr_var_expr.args[1]==:Array "Data sources need arrays as type"
        
        source_typ = node.args[2].args[3]
        @assert source_typ==:HDF5 "Only HDF5 data sources supported for now."
        
        hdf_var_name = node.args[2].args[4]
        hdf_file_name = node.args[2].args[5]

        # return :($(node.args[1]) = zeros($(arr_var_expr.args[2]),$(arr_var_expr.args[3])))
        num = get_unique_data_source_num()
        hps_source_var = symbol("__hpat_data_source_$num")
        hps_source_size_var = symbol("__hpat_data_source_size_$num")
        hps_source_size_call = symbol("__hpat_data_source_get_size_$(dims)d")
        declare_expr = :( $hps_source_var = __hpat_data_source_open($hdf_var_name,$hdf_file_name))
        size_expr = :( $hps_source_size_var = $hps_source_size_call($hps_source_var))
        return [declare_expr; size_expr]
=#
   elseif rhs.head==:call && isa(rhs.args[1],Expr) && rhs.args[1].head==:. && rhs.args[1].args[1]==:HPAT
        hps_call = rhs.args[1].args[2].args[1]
        new_opr = symbol("__hpat_$hps_call")
        node.args[2].args[1] = new_opr
        node.args[1] = :($lhs::Matrix{Float64})
   end
end

end # module
