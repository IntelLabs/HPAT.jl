module CaptureAPI

using CompilerTools
using CompilerTools.AstWalker
import ..API
import HPAT

import CompilerTools.DebugMsg
DebugMsg.init()

"""
At macro level, translate DataSource into function calls so that type inference
and ParallelAccelerator compilation works with knowledge of calls and allocations for arrays. 
"""
function process_node(node::Expr, state, top_level_number, is_top_level, read)
    @dprintln(3,"translating expr, head: ",node.head," node: ",node)
    if node.head == :(=) 
        return process_assignment(node, state, node.args[1], node.args[2])
    end
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_node(node::Any, state, top_level_number, is_top_level, read)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, state, lhs::Symbol, rhs::Expr)
    @dprintln(3,"assignment: ", lhs)
    if rhs.head ==:call && rhs.args[1]==:DataSource
        @dprintln(3,"datasource: ", lhs)
        arr_var_expr = rhs.args[2]
        @assert arr_var_expr.head==:curly "curly syntax expected for DataSource"
        if arr_var_expr.args[1]==:DataTable
            return translate_data_table(lhs, state, arr_var_expr, rhs.args[3], rhs.args[4:end])
        else
            return translate_data_source(lhs, state, arr_var_expr, rhs.args[3], rhs.args[4:end])
        end
   elseif rhs.head==:call && rhs.args[1]==:join
        @dprintln(3,"join: ", lhs)
        # 1st and 2nd args are tables to join
        t1 = rhs.args[2]
        t2 = rhs.args[3]
        @assert rhs.args[4].head==:comparison "invalid join key"
        @assert rhs.args[4].args[2]==:(==) "invalid join key"
        
        key1 = getQuoteValue(rhs.args[4].args[1])
        key1_arr = getColName(t1, key1)
        key2 = getQuoteValue(rhs.args[4].args[3])
        key2_arr = getColName(t2, key2)
        new_key = getQuoteValue(rhs.args[5])
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
        c1 = getQuoteValue(rhs.args[3])
        c1_arr = getColName(t1, c1)
        c1_out_arr = getColName(lhs, c1)
        out_e = []
        out_aggs = []
        out_arrs = [c1_out_arr]
        out_cols = [c1]
        for col_expr in rhs.args[4:end]
            @assert col_expr.head==:kw "expected assignment for new aggregate column"
            # output column name
            out_col = getQuoteValue(col_expr.args[1])
            out_col_arr = getColName(lhs, out_col)
            push!(out_cols, out_col)
            push!(out_arrs, out_col_arr)
            @assert col_expr.args[2].head==:call "expected aggregation function"
            # aggregation function
            func = col_expr.args[2].args[1]
            # aggregation expression
            e = col_expr.args[2].args[2]
            # replace column name with actual array in expression
            e = AstWalk(e, replace_col_with_array,  (t1, state[t1]))
            out_e_arr = symbol("_$(lhs)_$(out_col)_e")
            push!(out_aggs, :(($out_e_arr, $func)))
            push!(out_e,:($out_e_arr=$e))
        end
        out_call = Expr(:(=), Expr(:tuple, out_arrs...), :(aggregate($c1_arr,[$(out_aggs...)])) )
        push!(out_e, out_call)
        state[lhs] = out_cols
        return quote $(out_e...) end
   end
   CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, state, lhs::ANY, rhs::ANY)
   CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Replace column symbols with translated array names in aggregate expressions 
"""
function replace_col_with_array(node::QuoteNode, table::Tuple{Symbol,Vector{Symbol}}, top_level_number, is_top_level, read)
    col_sym = getQuoteValue(node)
    if col_sym in table[2]
        return getColName(table[1], col_sym)
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function replace_col_with_array(node::Expr, table::Tuple{Symbol,Vector{Symbol}}, top_level_number, is_top_level, read)
    if node.head!=:quote
        return CompilerTools.AstWalker.ASTWALK_RECURSE
    end
    col_sym = getQuoteValue(node)
    if col_sym in table[2]
        return getColName(table[1], col_sym)
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

"""
Data tables are broken down to individual column arrays, table meta data is saved
table_name = DataSource(DataTable{:column1=<typeof_column1>, :column2=<typeof_column2>, ...}, HDF5, file_name)
                ->  table_name_column1 = DataSource(...)
                    table_name_column2 = DataSource(...)
                    assertEqShape([table_name_column1, table_name_column2])
                    newTableMeta(:table_name, [:column1,:column2])

Example table to translate: 
        t1 = DataSource(DataTable{:userid = Int64,:val2 = Float64},HDF5,file_name)
"""
function translate_data_table(lhs, state, arr_var_expr, source_typ, other_args)
    @assert arr_var_expr.args[1]==:DataTable "expected :DataTable"
    # arr_var_expr has the form: :(DataTable{:userid = Int64,:val2 = Float64})
    @dprintln(3,"translating data table: ",arr_var_expr)
    out = []
    col_names = Symbol[]
    for column in arr_var_expr.args[2:end]
        @dprintln(3,"table column: ", column)
        @assert column.head==:(=)
        col_name = getQuoteValue(column.args[1])
        push!(col_names, col_name)
        col_type = column.args[2]
        col_lhs = getColName(lhs,col_name)
        col_source = translate_data_source(col_lhs, state, :(Vector{$(col_type)}), source_typ, ["/"*string(col_name);other_args])
        push!(out, col_source)
    end
    # save table info in state
    state[lhs] = col_names
    ret = quote $(out...) end
    @dprintln(3, "data table returns: ",ret)
    return ret
end

"""
Julia sometimes returns a QuoteNode for :column1
"""
function getQuoteValue(exp::QuoteNode)
    return exp.value
end

"""
Julia sometimes returns an Expr for :column1
"""
function getQuoteValue(exp::Expr)
    @assert exp.head==:quote "expected :quote expression"
    return exp.args[1]
end

function getColName(t::Symbol, c::Symbol)
    return symbol("_$(t)_$(c)")
end

end # module
