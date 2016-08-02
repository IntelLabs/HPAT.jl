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

module CaptureAPI

using CompilerTools
using CompilerTools.AstWalker
import ..API
import HPAT
using HPAT.Partitioning
using HPAT.SEQ
using HPAT.TWO_D
using HPAT.ONE_D

import CompilerTools.DebugMsg
DebugMsg.init()
#using Debug

""" At macro level, translate DataSource into function calls so that type inference
    and ParallelAccelerator compilation works with knowledge of calls and allocations for arrays.
"""
function process_node(node::Expr, state, top_level_number, is_top_level, read)
    @dprintln(3,"translating expr, head: ",node.head," node: ",node)
    # rename symbols before translation
    AstWalk(node, rename_symbols,  state.rename_map)

    if node.head == :(=)
        return process_assignment(node, state, node.args[1], node.args[2])
    elseif node.head==:call && node.args[1]==:DataSink
        return translate_data_sink(node.args[2], state, node.args[3], node.args[4:end])
    elseif node.head==:ref
        t1 = node.args[1]
        # table column ref like: t1[:c1]
        # TODO: column assignment like t1[:c2] =... should be processed in assignment
        if haskey(state.tableCols,t1)
            c1 = node.args[2]
            @assert isQuote(c1) " $node = $c1 invalid table ref"
            return getColName(t1, getQuoteValue(c1))
        end
        CompilerTools.AstWalker.ASTWALK_RECURSE
    elseif node.head==:macrocall
        return process_macros(node, state,node.args[1])
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
    elseif rhs.head==:ref
        t1 = rhs.args[1]
        # table filter like t1 = t1[:c1>=2]
        if haskey(state.tableCols, t1)
            return translate_filter(lhs, t1, rhs.args[2], state)
        end
    end
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_assignment(node, state, lhs::ANY, rhs::ANY)
    CompilerTools.AstWalker.ASTWALK_RECURSE
end

function process_macros(node, state, func)
    if func==Symbol("@partitioned")
        state.array_partitioning[node.args[2]] = convert_partitioning(node.args[3])
        return CompilerTools.AstWalker.ASTWALK_REMOVE
    end
    return node
end

function convert_partitioning(p::Symbol)
    if p==:HPAT_2D
        return TWO_D
    elseif p==:HPAT_1D
        return ONE_D
    elseif p==:HPAT_SEQ
        return SEQ
    else
        error("unknown partitioning $p")
        return SEQ
    end
    return SEQ
end

""" Translate filter out_t = t1[cond]

    We create an array of arrays to pass the columns to table_filter since
    arrays are passed by value with tuples.
                _t1_cond_e = cond
                _filter_t1 = Array(Vector,n)
                _filter_t1[1] = _t1_c1
                ...
                (HPAT.API.table_filter!)(_t1_e,_filter_t1)
                _t1_c1 = _filter_t1[1]
                ...
    """

function translate_filter(t_out::Symbol, t_in::Symbol, cond::Expr, state)
    @dprintln(3, "translating filter: ",t_in," ",cond," -> ", t_out)

    # @assert t_out!=t_in "Output table of filter must have different name"
    # rename output if it is the same name
    if t_out==t_in
      id = get_unique_id(state)
      new_t_out = Symbol("$(t_out)_f_$(id)")
      @dprintln(3, "renaming filter output table: ", t_out, " -> ", new_t_out)
      # symbol will be renamed for future AST nodes
      add_symbol_rename(t_out, new_t_out, state)
      t_out = new_t_out
    end

    # Adding new output table to state.tableCols
    state.tableCols[t_out] = state.tableCols[t_in]
    state.tableTypes[t_out] = state.tableTypes[t_in]

    # convert math operations to element-wise versions to work with arrays
    cond = AstWalk(cond, convert_oprs_to_elementwise,  (t_in, state.tableCols[t_in]))
    # replace column name with actual array in expression
    cond = AstWalk(cond, replace_col_with_array,  (t_in, state.tableCols[t_in]))
    # evaluate the condition into a BitArray
    cond_arr = Symbol("#$(t_in)#cond_e")
    cond_assign = :( $cond_arr = $cond )

    t_in_num_cols = length(state.tableCols[t_in])
    arg_arr_in = Symbol("_filter_in_$t_in")
    t_in_col_arr = :($arg_arr_in = Array(Vector,$(t_in_num_cols)))

    # Output table has same number of columns
    arg_arr_out = Symbol("_filter_out_$t_out")
    #t_out_col_arr = :($arg_arr_out = Array(Vector,$(t_in_num_cols)))
    # assign column arrays
    # e.g. t1[1] = _t1_c1
    t_in_col_arrs = map(x->getColName(t_in,x), state.tableCols[t_in])
    # Output table has same column names as in input table
    t_out_col_arrs = map(x->getColName(t_out,x), state.tableCols[t_out])

    assigns_in = [ Expr(:(=),:($arg_arr_in[$i]),:($(t_in_col_arrs[i]))) for i in 1:length(state.tableCols[t_in]) ]
    assigns_out = [ Expr(:(=), :($(t_out_col_arrs[i])::Vector{$(state.tableTypes[t_out][i])}) ,:($arg_arr_out[$i])) for i in 1:length(state.tableCols[t_out]) ]
    #assigns_out = [ Expr(:(=), :($(t_out_col_arrs[i])) ,:($arg_arr_out[$i])) for i in 1:length(state.tableCols[t_out]) ]

    mod_call = GlobalRef(HPAT.API, :table_filter!)
    filter_call = :( $arg_arr_out = ($mod_call)($cond_arr,($arg_arr_in)) )
    ret = Expr(:block, cond_assign, t_in_col_arr, assigns_in..., filter_call, assigns_out...)
    @dprintln(3,"filter returns: ", ret)
    return ret
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
    key1_index = findfirst(state.tableCols[t1],key1)
    key2_index = findfirst(state.tableCols[t2],key2)
    # Make key column first column and then append to it
    t1_col_arrs_sorted = [key1_arr]
    t2_col_arrs_sorted = [key2_arr]
    # get rest of the columns
    rest_cols1 = filter(x->x!=key1, state.tableCols[t1])
    rest_cols2 = filter(x->x!=key2, state.tableCols[t2])
    rest_cols1_arrs = map(x->getColName(t1,x),rest_cols1)
    rest_cols2_arrs = map(x->getColName(t2,x),rest_cols2)
    append!(t1_col_arrs_sorted, rest_cols1_arrs)
    append!(t2_col_arrs_sorted, rest_cols2_arrs)
    rest_cols3_arrs = map(x->getColName(lhs,x),[rest_cols1;rest_cols2])
    # save new table
    state.tableCols[lhs] = [new_key;rest_cols1;rest_cols2]
    #@assert key1==t1_col_arrs_sorted[1] "Join key $key1 of table $t1 is not column 1"
    #@assert key2==t2_col_arrs_sorted[1] "Join key $key2 of table $t2 is not column 1"
    @dprintln(3, "new table join output: ",lhs," ", state.tableCols[lhs])
    # pass tables as array of columns since [t1_c1,t1_c2...] flattens to single array instead of array of arrays
    # eg. t1 = Array(Vector,n)
    # HACK: the table names are extracted from these variable names in DomainPass
    _join_t1 = Symbol("_join_$t1")
    _join_t2 = Symbol("_join_$t2")
    t1_num_cols = length(state.tableCols[t1])
    t1_col_arr = :($_join_t1 = Array(Vector,$(t1_num_cols)))
    t2_num_cols = length(state.tableCols[t2])
    t2_col_arr = :($_join_t2 = Array(Vector,$(t2_num_cols)))
    # assign column arrays
    # e.g. t1[1] = _t1_c1
    assign1 = [ Expr(:(=),:($_join_t1[$i]), t1_col_arrs_sorted[i]) for i in 1:length(t1_col_arrs_sorted) ]
    assign2 = [ Expr(:(=),:($_join_t2[$i]), t2_col_arrs_sorted[i]) for i in 1:length(t2_col_arrs_sorted) ]

    #out = [t1_col_arr;t2_col_arr]
    # TODO: assign types
    #ret = :( ($new_key_arr,$(rest_cols3_arrs...)) = HPAT.API.join([$key1_arr;$(rest_cols1_arrs...)], [$key2_arr;$(rest_cols2_arrs...)]) )
    # GlobalRef since Julia doesn't resolve the module! why does GlobalRef work in surface AST??
    g_call = GlobalRef(HPAT.API,:join)
    _j_out = Symbol("_join_out_$lhs")
    join_call = :( $_j_out = $(g_call)($_join_t1, $_join_t2) )

    col_types = [ get_column_type(state, t1, key1) ]
    col_types1 = [ get_column_type(state, t1, i) for i in rest_cols1]
    col_types2 = [ get_column_type(state, t2, i) for i in rest_cols2]
    col_types = [col_types;col_types1;col_types2]
    # save new table types
    state.tableTypes[lhs] = col_types

    typ_assigns = [ :($new_key_arr::Vector{$(col_types[1])} = $_j_out[1]) ]
    typ_assigns1 = [ :($(rest_cols3_arrs[i])::Vector{$(col_types[i+1])} = $_j_out[$(i+1)]) for i in 1:length(rest_cols3_arrs)]
    typ_assigns = [typ_assigns;typ_assigns1]

    ret = Expr(:block,t1_col_arr,assign1...,t2_col_arr,assign2...,join_call,typ_assigns...)
    @dprintln(3,"join returns: ",ret)
    return ret
end

# Return type for given table and column
function get_column_type(state, t, col)
    index_col = findfirst(state.tableCols[t], col)
    return state.tableTypes[t][index_col]
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
    out_dummies = []
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
        out_e_arr = symbol("#$(lhs)#$(out_col)#e")
        push!(out_aggs, :(($out_e_arr,$e, $func)))
        push!(out_e,:($out_e_arr=$e))
        # to add types to aggregate output
        # make a dummy call to get the type with user's function
        # then use it in type assertion. Julia type inference can infer the type and use it for common functions
        # example: t2_c1, t2_c2 = aggregate(t1_c1, (t1_c2_e,sum))
        #          _T_c2 = typeof(sum(t1_c2_e))
        #          t2_c2::_T_c2 = t2_c2
        typ_name = Symbol("_T_$(out_col)")
        dummy_reduce = :( $typ_name = typeof($(func)($out_e_arr)) )
        push!(out_dummies, dummy_reduce)
        # typ_assigns = [ :($new_key_arr::Vector{$(col_types[1])} = _j_out[1]) ]

    end
    append!(out_e,out_dummies)

    out_var = Symbol("_agg_out_$(lhs)_#_$(t1)")

    # assign types of output columns
    # we know the type of key column already
    out_type_assigns = [ :($c1_out_arr::Vector{$(state.tableTypes[t1][1])} = $(out_var)[1]) ]
    # create type variables for other types
    out_typs = [ Symbol("_T_$(out_cols[i])") for i in 1:length(out_cols) ]
    out_typs[1] = state.tableTypes[t1][1]
    out_type_assigns1 = [ :($(out_arrs[i])::Vector{$(out_typs[i])} = $(out_var)[$i])  for i in 2:length(out_arrs) ]
    out_type_assigns = [out_type_assigns;out_type_assigns1]

    # GlobalRef since Julia doesn't resolve the module! why does GlobalRef work in surface AST??
    agg_call = GlobalRef(HPAT.API,:aggregate)

    out_call = Expr(:(=), out_var, :($(agg_call)($c1_arr,[$(out_aggs...)])) )
    push!(out_e, out_call)
    push!(out_e, out_type_assigns...)
    state.tableCols[lhs] = out_cols
    state.tableTypes[lhs] = out_typs
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


""" Convert math operations to element-wise versions to work with arrays

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

"""
    rename symbols of the node according to rename_map
"""
function rename_symbols(node::Symbol, rename_map::Dict{Symbol,Symbol}, top_level_number, is_top_level, read)
    new_sym = node
    while haskey(rename_map, new_sym)
      new_sym = rename_map[new_sym]
    end
    return new_sym
end

function rename_symbols(node::ANY, rename_map::Dict{Symbol,Symbol}, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function translate_data_sink(var, state, source_typ, other_args)
    @assert source_typ==:HDF5 || source_typ==:TXT "Only HDF5 and TXT (text) data sources supported for now."

    # desugar call
    call_name = symbol("data_sink_$source_typ")
    # GlobalRef since Julia doesn't resolve the module!
    api_call = GlobalRef(HPAT.API, call_name)
    rhs = Expr(:call, api_call, var, other_args...)
    return rhs
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

""" Data tables are broken down to individual column arrays, table meta data is saved
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

function isQuote(e::Expr)
    return e.head==:quote
end

function isQuote(e::QuoteNode)
    return true
end

function isQuote(e::ANY)
    return false
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


"""
    Convert a table column to an array name
"""
function getColName(t::Symbol, c::Symbol)
    return symbol("#$(t)#$(c)")
end

"""
    reverse of getColName()
    get column name from array name
"""
function revColName(t::Symbol, c_arr::Symbol)
    t_str = string(t)
    c_arr_str = string(c_arr)
    before_col_len = 2+length(t_str) # two underscores + length of table name
    return c_arr_str[before_col_len+1:end]
end

"""
    get a unique id for renaming
"""
function get_unique_id(state)
  state.unique_id += 1
  return state.unique_id
end

"""
    add symbol to rename map, to be applied from next AST node
"""
function add_symbol_rename(t::Symbol, new_t::Symbol, state)
  state.rename_map[t] = new_t
end

end # module
