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

module DataTablePass

#using Debug

using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()
using CompilerTools.Helper
using CompilerTools.LambdaHandling
import CompilerTools.ReadWriteSet
import CompilerTools.LambdaHandling.matchVarDef

import HPAT
import HPAT.CaptureAPI.getColName
using HPAT.DomainPass.get_table_meta

using ParallelAccelerator
using ParallelAccelerator.DomainIR.AstWalk
import ParallelAccelerator.ParallelIR
import ParallelAccelerator.ParallelIR.computeLiveness

mk_call(fun,args) = Expr(:call, fun, args...)

# ENTRY to datatable-pass
type QueryTreeNode
    expr::Expr
    children::Vector{QueryTreeNode}
    # positions in original AST
    start_pos::Int
    end_pos::Int
    parent::Union{QueryTreeNode,Void}

    # empty node constructor
    function QueryTreeNode()
        new(Expr(:null), QueryTreeNode[], -1, -1, nothing)
    end
    # Constructor
    function QueryTreeNode(expr::Expr, sp, ep)
        new(expr, QueryTreeNode[], sp, ep, nothing)
    end
end

islast(n::QueryTreeNode) = (n == n.child)

function print_tree(root_qtn)
    println(string("     Tree::", string(root_qtn.expr.head)))
    for qn in root_qtn.children
        print_tree(qn)
    end
end

QueryTreeNode(data::Expr) = QueryTreeNode(data)
QueryTreeNode(data::Expr, parent::QueryTreeNode,sp,ep) = QueryTreeNode(data,parent,sp,ep)

function from_root(function_name, ast::Tuple)
    @dprintln(1, "Starting main DataTablePass.from_root.  function = ", function_name, " ast = ", ast)
    (linfo, body) = ast
    lives = computeLiveness(body, linfo)
    tableCols, tableTypes = get_table_meta(body)

    tree = make_query_tree(body.args)
    print_tree(tree)
    # transform body
    body.args = from_toplevel_body(body.args, tree, tableCols, linfo)
    @dprintln(1,"DataTablePass.from_root returns function = ", function_name, " ast = ", body)
    return LambdaVarInfoToLambda(linfo, body.args, ParallelAccelerator.DomainIR.AstWalk)
end

# nodes are :body of AST
function from_toplevel_body(nodes::Array{Any,1}, tree, tableCols, linfo)
    res::Array{Any,1} = []
    # TODO Handle optimization; Need to replace all uses of filter output table after pushing up with new filter output table
    perform_opts(nodes, tree, tableCols, linfo)
    @dprintln(3,"Datatable pass: Body after query optimizations ", nodes)

    # After optimizations make actuall call nodes for cgen
    for (index, node) in enumerate(nodes)
        if isa(node, Expr) && node.head==:filter
            append!(res, translate_hpat_filter(node))
        elseif isa(node, Expr) && node.head==:join
            append!(res, translate_hpat_join(node,linfo))
       elseif isa(node, Expr) && node.head==:aggregate
            append!(res, translate_hpat_aggregate(node,linfo))
        else
            append!(res, [node])
        end
    end
    return res
end

"""
    Build a query tree of relational operations.
"""
function make_query_tree(nodes::Array{Any,1})
    # node of trees where new nodes are added
    q_nodes = QueryTreeNode[]
    root = QueryTreeNode()
    # TODO: fix index
    for (index, node) in enumerate(reverse(nodes))
        new_qt_node = QueryTreeNode()
        # ignore non-expression nodes
        # TODO: handle having different basic blocks
        if !isa(node, Expr) continue end
        if node.head==:filter
            # -2 because id and condition are above it and these are part of filter
            new_qt_node = QueryTreeNode(node, index-2, index)
        elseif node.head==:join
            # -1 because id is above it which part of join
            new_qt_node = QueryTreeNode(node, index-1, index)
        elseif node.head==:aggregate
            # node.args[4] gives length of expression nodes list and they are part of aggregate
            # * 2 because each expression nodes occupy two places
            # TODO Fix this. It might not work in future. Aggregate node indexes are variable
            new_qt_node = QueryTreeNode(node, (index-(length(node.args[4]) * 2)),index)
        else
            continue
        end
        # the first node is root of the tree
        # TODO: handle projections as root
        if length(q_nodes)==0
            root = new_qt_node
            push!(q_nodes, root)
        else
            add_child!(q_nodes, new_qt_node)
        end
    end
    return root
end

"""
    adds new node to tree and updates q_nodes accordingly
"""
function add_child!(q_nodes::Vector{QueryTreeNode}, new_qt_node::QueryTreeNode)
    for q_node in q_nodes
        # if any of q_node's inputs are output of new_qt_node
        if isDependent(q_node, new_qt_node)
            push!(q_node.children, new_qt_node)
            new_qt_node.parent = q_node
        end
    end
    push!(q_nodes, new_qt_node)
end

"""
    if any of t2's inputs are output of t1
"""
function isDependent(t2::QueryTreeNode, t1::QueryTreeNode)
    local t1_out_table::Symbol
    if t1.expr.head==:filter
        t1_out_table = t1.expr.args[2]
    elseif t1.expr.head==:aggregate || t1.expr.head==:join
        t1_out_table = t1.expr.args[1]
    else
        throw("invalid input query node: $t1")
    end
    local t2_in_tables::Vector{Symbol}
    if t2.expr.head==:filter
        t2_in_tables = [t2.expr.args[3]]
    elseif t2.expr.head==:join
        t2_in_tables = t2.expr.args[2:3]
    elseif t2.expr.head==:aggregate
        t2_in_tables = [t2.expr.args[2]]
    else
        throw("invalid input query node: $t2")
    end
    return t1_out_table in t2_in_tables
end


"""
    Translate join node so that backend can translate
"""
function translate_hpat_join(node,linfo)
    # args: id, length of output table columns, length of 1st input table columns,
    #       length of 2nd input table columns, output columns, input1 columns, input2 columns
    open_call = mk_call(GlobalRef(HPAT.API,:__hpat_join),
                        [node.args[10]; length(node.args[7]); length(node.args[8]); length(node.args[9]); node.args[7]; node.args[8]; node.args[9]])
    dprintln(3, "Datatable pass: join translated: ", open_call)
    return [open_call]
end

"""
    Make filter :call node with the following layout
    condition expression lhs, columns length, columns names(#t1#c1) ...
"""
function translate_hpat_filter(node)
    num_cols = length(node.args[4])
    # args: id, condition, number of columns, output table columns, input table columns
    open_call = mk_call(GlobalRef(HPAT.API,:__hpat_filter),
                        [node.args[8]; node.args[1]; num_cols; node.args[5]; node.args[6]])
    dprintln(3, "Datatable pass: filter translated: ", open_call)
    return [open_call]
end

"""
    Translate aggragte node so that backend can translate
"""
function translate_hpat_aggregate(node,linfo)
    # args: id, key, number of expressions, expression list
    open_call = mk_call(GlobalRef(HPAT.API,:__hpat_aggregate),
                        [node.args[7]; node.args[3]; length(node.args[4]); node.args[4]; node.args[5]; node.args[6]])
    dprintln(3, "Datatable pass: aggregate translated: ", open_call)
    return [open_call]
end

"""
    Starting point of all optimization performed on query tree
    Add rules to perform more optimizations
"""
function perform_opts(nodes, root_qtn, tableCols, linfo)
    # TODO Add more rules of optimizations
    curr_qtn = root_qtn
    while true
        islast(curr_qtn) && break
        # save child pointer because swap may happen otherwise concurrent modifiction error would be thrown
        child_qtn = curr_qtn.child
        # RULE 1: PUSH FILTER UP
        # if parent is join and child is filter then switch the order
        if curr_qtn.parent.data.head==:join && curr_qtn.data.head==:filter
            rename_map = push_filter_up(nodes, curr_qtn, tableCols, linfo)
            call_astwalk_down_tree(nodes, curr_qtn.end_pos, rename_map)
            swap_querytree_nodes(curr_qtn.parent, curr_qtn)
        end
        curr_qtn = child_qtn
    end
end

"""
    Swap the positions and data of two query tree nodes
"""
function swap_querytree_nodes(qtn1, qtn2)
    if qtn1.data.head==:join && qtn2.data.head==:filter
        # filter node is shoved above join and filter has 3 nodes
        qtn2.start_pos = qtn1.start_pos
        qtn2.end_pos = qtn2.start_pos + 2
        qtn1.start_pos = qtn1.start_pos + 3
        qtn1.end_pos = qtn1.end_pos + 3
    else
        @assert false "Write rules to handle swaping"
    end
    qtn2_child = qtn2.child
    qtn1_parent = qtn1.parent

    qtn1.parent.child = qtn2
    qtn1.child = qtn2_child

    qtn2.child = qtn1
    qtn2_child.parent = qtn1
    qtn2.parent = qtn1_parent
    qtn1.parent = qtn2

end

"""
    OPTIMIZATION: PUSH FILTER UP
        if there is a join before a filter then move that filter above join
        Assign new output to filter operation and use that in join node
        Old output of filter needs to be replaced with whatever table operation output right above filter
"""

function push_filter_up(nodes::Array{Any,1}, curr_qtn, tableCols, linfo)
    rename_map::Dict{LHSVar,LHSVar} = Dict{LHSVar,LHSVar}()
    join_node = curr_qtn.parent.data
    filter_node = curr_qtn.data
    # args[2] and args[3] are join input tables (symbol names of tables)
    # Search in these input tables on which filter condition can be applied.
    join_input_tables = [join_node.args[2];join_node.args[3]]
    filter_output_table = filter_node.args[2]
    # start_pos is the cond pos
    new_cond_node = nodes[curr_qtn.start_pos]
    filter_id_node = nodes[curr_qtn.start_pos+1]
    # args[7] is the conditional
    cond_rhs = string(filter_node.args[7].args[2])
    # Extract column name from filter condition and check in join input tables on which it was applied
    # It will return table name and join input table index 1 or 2 which should be filtered
    table_name,j_ind = find_table_from_cond(tableCols,join_input_tables,cond_rhs)

    # Change condition rhs input table; Does not change output table
    replace_table_in_cond(new_cond_node,table_name)
    # return output filter table name which will be replaced in join
    out_filter_table, out_filter_table_cols, in_filter_table_cols = replace_table_in_filter_node(filter_node,table_name,tableCols)
    # As new table columns are made after pushing. We need to add to linfo
    for co in 1:length(out_filter_table_cols)
        CompilerTools.LambdaHandling.addLocalVariable(out_filter_table_cols[co],
                                                      CompilerTools.LambdaHandling.getType(in_filter_table_cols[co],linfo),
                                                      ISASSIGNEDONCE | ISASSIGNED, linfo)
    end
    # Adding mapping from filter table output to join output
    # In future change generalize it because any node output above filter node should be used instead of join output
    add_mapping(filter_output_table, join_node.args[1], tableCols, rename_map)
    # + 1 because we need right index in join expression node
    join_node.args[j_ind + 1] = out_filter_table
    # Also replace table columns list with appropriate name
    # + 7 because we need right index in join expression node
    join_node.args[j_ind + 7] = out_filter_table_cols
    # remove condition,id and filter node
    for ind in curr_qtn.start_pos:curr_qtn.end_pos
        nodes=deleteat!(nodes,ind)
    end
    # Move condition, id and filter node
    # curr_qtn parents start_pos give me place to put filter
    splice!(nodes, curr_qtn.parent.start_pos:1, [new_cond_node, filter_id_node, filter_node])

    return rename_map
end

"""
    Utility function to perform renaming astwalk starting from the argument pos
"""
function call_astwalk_down_tree(nodes, pos, rename_map::Dict{LHSVar,LHSVar})
    for i in pos:length(nodes)
        AstWalk(nodes[i], rename_symbols, rename_map)
    end
end

function rename_symbols(node::LHSVar, rename_map::Dict{LHSVar,LHSVar}, top_level_number, is_top_level, read)
    new_sym = node
    while haskey(rename_map, new_sym)
      new_sym = rename_map[new_sym]
    end
    return new_sym
end

function rename_symbols(node::TypedVar, rename_map::Dict{LHSVar,LHSVar}, top_level_number, is_top_level, read)
    typ = node.typ
    new_sym = toLHSVar(node)
    while haskey(rename_map, new_sym)
      new_sym = rename_map[new_sym]
    end
    return TypedVar(new_sym,typ)
end

function rename_symbols(node::ANY, rename_map::Dict{LHSVar,LHSVar}, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end

function add_mapping(old_t::Symbol, new_t::Symbol, tableCols::Dict{Symbol,Array{Symbol,1}}, mapping::Dict{LHSVar,LHSVar})
    mapping[old_t]=new_t
    old_t_cols = tableCols[old_t]
    new_t_cols = tableCols[new_t]
    for (ind,old_col) in enumerate(old_t_cols)
        new_col = new_t_cols[ind]
        @assert new_col==old_col "Both table cols should be same"
        mapping[getColName(old_t,old_col)] = getColName(new_t,new_col)
    end
    @dprintln(3,"Datatable pass: After adding new mappings = ",mapping)
end


"""
    OPTIMIZATION: Remove extra columns.
    Insert Project(select) above aggregate and join
"""
function prune_column(nodes::Array{Any,1})
end

"""
    OPTIMIZATION: Combine two or more filters using AND on condition which are on
    same table without any other inner query operations
"""
function combine_filters()
end

"""
    OPTIMIZATION: If select/projection operator only use one of the table columns after join then
    we can safely remove that join.
    TODO check with ParallelAccelerator
"""
function remove_unnessary_joins()
end

"""
    Find table name for the given filter condition
    This necessary to replace the table name if you move filter
    above join or aggregates
    TODO make it short
"""
function find_table_from_cond(tableCols,join_input_tables,cond)
    # I am assuming that second element has column name
    arr = split(cond,'#')
    col_name = arr[3]
    for (j_ind,k) in enumerate(join_input_tables)
        arr = tableCols[k]
        for (index, value) in enumerate(arr)
            curr_col = string(value)
            if curr_col == col_name
                return string(k),j_ind
            end
        end
    end
    @assert false "Could not find column in join input tables"
end

"""
    Replaces table name in the filter condition(mmap) rhs
    e.g
     table1#cond_e = table1#col1 > 1 => table1#cond_e = table2#col1 > 1
"""
function replace_table_in_cond(node,table_name)
    arr2 = split(string(node.args[2].args[1][1].name),'#')
    node.args[2].args[1][1] = Symbol(string("@",table_name,"@",arr2[3]))
end

"""
    Replaces table name and columns accordingly in the filter node after moving
"""
function replace_table_in_filter_node(node, table_name, tableCols)
    # node.args[2] gives output table
    out_table_name = string(table_name, "_fil")
    node.args[2] = Symbol(out_table_name)
    # New table must be added in tableCols which will have same columns as input
    tableCols[Symbol(out_table_name)] = tableCols[Symbol(table_name)]
    # node.args[3] gives input table
    node.args[3] = Symbol(table_name)
    # replace in condition expression (could be removed in future)
    # args[7] gives cond_lhs
    arr2 = split(string(node.args[7].args[2].name),'#')
    node.args[7].args[2] = getColName(Symbol(table_name), Symbol(arr2[3]))
    # Replace columns list with new columns
    node.args[4] = tableCols[Symbol(table_name)]
    # Replace output and input table columns array with new table columns
    node.args[5] = []
    node.args[6] = []
    for (index, col) in enumerate(tableCols[node.args[2]])
        append!(node.args[5], [getColName(Symbol(out_table_name), col)])
        append!(node.args[6], [getColName(Symbol(table_name), col)])
    end
    # Return output filter table and its columns list and input tables columns list
    return node.args[2], node.args[5], node.args[6]
end

end # DataTablePass
