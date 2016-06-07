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
    elseif node.head==:ref # table column ref like: t1[:c1]
        t1 = node.args[1]
        if haskey(state.tableCols,t1)
            c1 = node.args[2]
            @assert isa(c1,QuoteNode) || (isa(c1,Expr) && c1.head==:quote) "invalid table ref"
             return getColName(t1, getQuoteValue(c1))
        end
        CompilerTools.AstWalker.ASTWALK_RECURSE 
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
        return translate_join(lhs, rhs, state)
   elseif rhs.head==:call && rhs.args[1]==:aggregate
        return translate_aggregate(lhs,rhs,state)
   end
   CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, state, lhs::ANY, rhs::ANY)
   CompilerTools.AstWalker.ASTWALK_RECURSE
end

"""
Basic join will match first column of each column array
  t3 = join(t1, t2, :c1==:c1, :c2)
                ->  t3_c1, t3_c2,... = join([t1_c1,t1_c2,...], [t2_c1,t2_c2,...])
                    assertEqShape(t3_c1, t3_c2,...)
                    newTableMeta(:t3, [:c1,:c2,...])
"""
function translate_join(lhs, rhs, state)
    @dprintln(3,"join: ", lhs)
    # 1st and 2nd args are tables to join
    t1 = rhs.args[2]
    t2 = rhs.args[3]
    @assert rhs.args[4].head==:comparison "invalid join key"
    @assert rhs.args[4].args[2]==:(==) "invalid join key"
    
    # get key columns
    key1 = getQuoteValue(rhs.args[4].args[1])
    key1_arr = getColName(t1, key1)
    key2 = getQuoteValue(rhs.args[4].args[3])
    key2_arr = getColName(t2, key2)
    new_key = getQuoteValue(rhs.args[5])
    new_key_arr = getColName(lhs, new_key)
    
    # get rest of the columns
    rest_cols1 = filter(x->x!=key1, state.tableCols[t1])
    rest_cols2 = filter(x->x!=key2, state.tableCols[t2])
    rest_cols1_arrs = map(x->getColName(t1,x),rest_cols1)
    rest_cols2_arrs = map(x->getColName(t2,x),rest_cols2)
    rest_cols3_arrs = map(x->getColName(lhs,x),[rest_cols1;rest_cols2])
    # save new table
    state.tableCols[lhs] = [new_key;rest_cols1;rest_cols2]
    
    # pass tables as array of columns since [t1_c1,t1_c2...] flattens to single array instead of array of arrays
    # eg. t1 = Array(Vector,n)
    t1_num_cols = length(state.tableCols[t1])
    t1_col_arr = :(_join_t1 = Array(Vector,$(t1_num_cols)))
    t2_num_cols = length(state.tableCols[t2])
    t2_col_arr = :(_join_t2 = Array(Vector,$(t2_num_cols)))
    # assign column arrays
    # e.g. t1[1] = _t1_c1
    assign1 = [ Expr(:(=),:(_join_t1[$i]),getColName(t1,state.tableCols[t1][i])) for i in 1:length(state.tableCols[t1]) ]
    assign2 = [ Expr(:(=),:(_join_t2[$i]),getColName(t2,state.tableCols[t2][i])) for i in 1:length(state.tableCols[t2]) ]
    #out = [t1_col_arr;t2_col_arr]
    # TODO: assign types
    #ret = :( ($new_key_arr,$(rest_cols3_arrs...)) = HPAT.API.join([$key1_arr;$(rest_cols1_arrs...)], [$key2_arr;$(rest_cols2_arrs...)]) )
    join_call = :( _j_out = HPAT.API.join(_join_t1, _join_t2) )
    
    col_types = [ state.tableTypes[t1][1] ]
    col_types1 = [ state.tableTypes[t1][i+1] for i in 1:length(rest_cols1)]
    col_types2 = [ state.tableTypes[t2][i+1] for i in 1:length(rest_cols2)]
    col_types = [col_types;col_types1;col_types2]
    # save new table types
    state.tableTypes[lhs] = col_types
    
    typ_assigns = [ :($new_key_arr::Vector{$(col_types[1])} = _j_out[1]) ]
    typ_assigns1 = [ :($(rest_cols3_arrs[i])::Vector{$(col_types[i+1])} = _j_out[$(i+1)]) for i in 1:length(rest_cols3_arrs)]
    typ_assigns = [typ_assigns;typ_assigns1]
    
    ret = Expr(:block,t1_col_arr,assign1...,t2_col_arr,assign2...,join_call,typ_assigns...)
    @dprintln(3,"join returns: ",ret)
    return ret
end

"""
example: t4 = aggregate(t3, :userid, :sumo2 = sum(:val2==1.1), :size_val3 = size(:val3))

f is a reduction function on grouped data, e is closure for filtering column elements
 t2 = aggregate(t1, :c1, :c3=f(e(:c2,...)),...)
                ->  t2_c3_e = e(t1_c2,...)
                    ...
                    t2_c1, t2_c3,... = aggregate(t1_c1, (t2_c3_e,f),...)
                    assertEqShape(t3_c1, t3_c3,...)
                    newTableMeta(:t3, [:c1,:c3,...])

"""
function translate_aggregate(lhs, rhs, state)
    @dprintln(3,"aggregate: ", lhs)
    t1 = rhs.args[2]
    c1 = getQuoteValue(rhs.args[3])
    c1_arr = getColName(t1, c1)
    c1_out_arr = getColName(lhs, c1)
    out_e = []
    out_aggs = []
    out_arrs = [c1_out_arr]
    out_cols = [c1]
    out_type_assigns = [ :($c1_out_arr::Vector{$(state.tableTypes[t1][1])} = $c1_out_arr) ]
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
        # convert math operations to element-wise versions to work with arrays
        e = AstWalk(e, convert_oprs_to_elementwise,  (t1, state.tableCols[t1]))
        # replace column name with actual array in expression
        e = AstWalk(e, replace_col_with_array,  (t1, state.tableCols[t1]))
        out_e_arr = symbol("_$(lhs)_$(out_col)_e")
        push!(out_aggs, :(($out_e_arr, $func)))
        push!(out_e,:($out_e_arr=$e))
        # to add types to aggregate output
        # make a dummy call to get the type with user's function
        # then use it in type assertion. Julia type inference can infer the type and use it for common functions
        # example: t2_c1, t2_c2 = aggregate(t1_c1, (t1_c2_e,sum))
        #          _T_c2 = typeof(sum(t1_c2_e))
        #          t2_c2::_T_c2 = t2_c2
        typ_name = Symbol("_T_$(out_col)")
        dummy_reduce = :( $typ_name = typeof($(func)($out_e_arr)) )
        push!(out_e, dummy_reduce)
        # typ_assigns = [ :($new_key_arr::Vector{$(col_types[1])} = _j_out[1]) ]
        push!(out_type_assigns,:($out_col_arr::Vector{$typ_name} = $out_col_arr))
    end
    out_call = Expr(:(=), Expr(:tuple, out_arrs...), :(HPAT.API.aggregate($c1_arr,[$(out_aggs...)])) )
    push!(out_e, out_call)
    push!(out_e, out_type_assigns...)
    state.tableCols[lhs] = out_cols
    # TODO: save new table types
    #ret = quote $(out_e...) end
    ret = Expr(:block, out_e...)
    @dprintln(3,"aggregate returns: ",ret)
    return ret
end

agg_oprs_map = Dict{Symbol, Symbol}(
    # comparison operators
    :(>) => :(.>),
    :(<) => :(.<),
    :(<=) => :(.<=),
    :(>=) => :(.>=),
    :(==) => :(.==),
    
    # binary operators
    :(+) => :(.+),
    :(-) => :(.-),
    :(*) => :(.*),
    :(/) => :(./),
    :(%) => :(.%),
    :(^) => :(.^),
    :(<<) => :(.<<),
    :(>>) => :(.>>)
)


"""
Convert math operations to element-wise versions to work with arrays

example: user can write 't4 = aggregate(t3, :userid, :sumo2 = sum(:val2==1.1), :size_val3 = size(:val3))'
the aggregate expression ':val2==1.1' should be translated to '_t4_val2.==1.1' to be valid for arrays.

For simplicity, we assume we can convert all operations with element-wise versions.
This is wrong for rare use cases such as comparison of two arrays([1,2]==[1,3] is not the same as [1,2].==[1,3]),
TODO: make it fully general. 
"""
function convert_oprs_to_elementwise(node::Symbol, table::Tuple{Symbol,Vector{Symbol}}, top_level_number, is_top_level, read)
    if haskey(agg_oprs_map,node)
        return agg_oprs_map[node]
    end
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function convert_oprs_to_elementwise(node::ANY, table::Tuple{Symbol,Vector{Symbol}}, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
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
    col_types = Symbol[]
    for column in arr_var_expr.args[2:end]
        @dprintln(3,"table column: ", column)
        @assert column.head==:(=)
        col_name = getQuoteValue(column.args[1])
        push!(col_names, col_name)
        col_type = column.args[2]
        push!(col_types,col_type)
        col_lhs = getColName(lhs,col_name)
        col_source = translate_data_source(col_lhs, state, :(Vector{$(col_type)}), source_typ, ["/"*string(col_name);other_args])
        push!(out, col_source)
    end
    # save table info in state
    state.tableCols[lhs] = col_names
    state.tableTypes[lhs] = col_types
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
