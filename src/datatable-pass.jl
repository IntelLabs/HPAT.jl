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

import Base.show

using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools.AstWalker
import CompilerTools.ReadWriteSet
using CompilerTools.LambdaHandling
using CompilerTools.Helper

import HPAT
using HPAT.DomainPass.get_table_meta
import HPAT.CaptureAPI.getColName

using ParallelAccelerator
import ParallelAccelerator.ParallelIR
import ParallelAccelerator.ParallelIR.isArrayType
import ParallelAccelerator.ParallelIR.getParforNode
import ParallelAccelerator.ParallelIR.isBareParfor
import ParallelAccelerator.ParallelIR.isAllocation
import ParallelAccelerator.ParallelIR.TypedExpr
import ParallelAccelerator.ParallelIR.get_alloc_shape
import ParallelAccelerator.ParallelIR.computeLiveness

import ParallelAccelerator.ParallelIR.ISCAPTURED
import ParallelAccelerator.ParallelIR.ISASSIGNED
import ParallelAccelerator.ParallelIR.ISASSIGNEDBYINNERFUNCTION
import ParallelAccelerator.ParallelIR.ISCONST
import ParallelAccelerator.ParallelIR.ISASSIGNEDONCE
import ParallelAccelerator.ParallelIR.ISPRIVATEPARFORLOOP
import ParallelAccelerator.ParallelIR.PIRReduction

mk_call(fun,args) = Expr(:call, fun, args...)
# ENTRY to datatable-pass
type QueryTreeNode{T}
    data::T
    parent::QueryTreeNode{T}
    child::QueryTreeNode{T}
    # positions in original AST
    start_pos::Int
    end_pos::Int
    # Constructor for root
    function QueryTreeNode(data::T)
        n = new(data)
        n.parent = n
        n.child = n
        n.start_pos = 0
        n.end_pos = 0
        n
    end
    # Constructor
    function QueryTreeNode(data::T, parent::QueryTreeNode, sp, ep)
        n = new(data, parent)
        n.child = n
        n.start_pos = sp
        n.end_pos = ep
        n
    end
end

function add_child{T}(data::T,parent::QueryTreeNode{T},sp,ep)
    newc = QueryTreeNode(data,parent,sp,ep)
    parent.child = newc
    newc
end
function print_tree(root_qtn)
    # TODO recursive function to pretty print the query plan tree
end

QueryTreeNode{T}(data::T) = QueryTreeNode{T}(data)
QueryTreeNode{T}(data::T, parent::QueryTreeNode{T},sp,ep) = QueryTreeNode{T}(data,parent,sp,ep)

function from_root(function_name, ast::Tuple)
    @dprintln(1,"Starting main DataTablePass.from_root.  function = ", function_name, " ast = ", ast)
    (linfo, body) = ast
    lives = computeLiveness(body, linfo)
    tableCols, tableTypes = get_table_meta(body)
    # transform body
    root_qtn = QueryTreeNode("root")
    plan = make_query_plan(body.args,root_qtn)
    body.args = from_toplevel_body(body.args,tableCols, linfo)
    @dprintln(1,"DataTablePass.from_root returns function = ", function_name, " ast = ", body)
    return LambdaVarInfoToLambda(linfo, body.args)
end

# nodes are :body of AST
function from_toplevel_body(nodes::Array{Any,1},tableCols,linfo)
    res::Array{Any,1} = []
    # TODO Handle optimization; Need to replace all uses of filter output table after pushing up with new filter output table
    # nodes = push_filter_up(nodes,tableCols,linfo)
    @dprintln(3,"body after query optimizations ", nodes)
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

function make_query_plan(nodes::Array{Any,1},root_qtn)
    last_child = root_qtn
    for (index, node) in enumerate(nodes)
        if isa(node, Expr) && node.head==:filter
            last_child = add_child("filter",last_child,index-1,index)
        elseif isa(node, Expr) && node.head==:join
            last_child = add_child("join",last_child,index,index)
        elseif isa(node, Expr) && node.head==:aggregate
            last_child = add_child("aggregate",last_child,(index-(length(node.args[4]))),index)
        else
        end
    end
end

#=

=#
function translate_hpat_join(node,linfo)
    res = Any[]
    # args: id, length of output table columns, length of 1st input table columns,
    #       length of 2nd input table columns, output columns, input1 columns, input2 columns
    open_call = mk_call(GlobalRef(HPAT.API,:__hpat_join),
                        [node.args[10]; length(node.args[7]); length(node.args[8]); length(node.args[9]); node.args[7]; node.args[8]; node.args[9]])
    push!(res, open_call)
    return res
end

#=
Make filter :call node with the following layout
condition expression lhs, columns length, columns names(#t1#c1) ...
=#
function translate_hpat_filter(node)
    num_cols = length(node.args[4])
    # args: id, condition, number of columns, output table columns, input table columns
    open_call = mk_call(GlobalRef(HPAT.API,:__hpat_filter),
                        [node.args[8]; node.args[1]; num_cols; node.args[5]; node.args[6]])
    return [open_call]
end

#=

=#
function translate_hpat_aggregate(node,linfo)
    res = Any[]
    # args: id, key, number of expressions, expression list
    open_call = mk_call(GlobalRef(HPAT.API,:__hpat_aggregate),
                        [node.args[8]; node.args[3]; length(node.args[4]); node.args[4]; node.args[6]; node.args[7]])
    push!(res, open_call)
    return res
end

#=
if there is a join before a filter then move that filter above join
=#
function push_filter_up(nodes::Array{Any,1},tableCols,linfo)
    new_nodes = []
    hit_join = false
    pos = 0
    for i in 1:length(nodes)
        #println(nodes[i])
        if isa(nodes[i], Expr) && nodes[i].head==:join
            hit_join = true
            # move above join id
            pos=i-1
        end
        if isa(nodes[i], Expr) && nodes[i].head==:filter && hit_join
            # TODO change condition in filter expression node too
            join_node = nodes[pos+1]
            # args[2] and args[3] are join input table.
            # Search in these input tables on which filter condition can be applied.
            join_input_tables = [join_node.args[2];join_node.args[3]]
            new_filter_node = nodes[i]
            new_id_node = nodes[i-1]
            new_cond_node = nodes[i-2]
            cond_lhs = string(nodes[i].args[1])
            # args[7] is the conditional
            cond_rhs = string(nodes[i].args[7].args[2].name)
            # Extract column name from filter condition and check in join input tables on which it was applied
            # It will return table name and join input table index 1 or 2 which should be filtered
            table_name,j_ind = find_table_from_cond(tableCols,join_input_tables,cond_rhs)

            replace_cond_in_linfo(linfo,cond_lhs,table_name)
            replace_table_in_cond(new_cond_node,table_name)
            # return output filter table name which will be replaced in join
            out_filter_table, out_filter_table_cols = replace_table_in_filter_node(new_filter_node,table_name,tableCols)
            # + 1 because we need right index in join expression node
            join_node.args[j_ind + 1] = out_filter_table
            # Also replace table columns list with appropriate name 
            # + 7 because we need right index in join expression node
            join_node.args[j_ind + 7] = out_filter_table_cols
            # remove condition and id node above filter node
            pop!(new_nodes)
            pop!(new_nodes)
            splice!(new_nodes,pos:1,[new_cond_node,new_id_node,new_filter_node])
            hit_join=false
            continue
        end
        push!(new_nodes, nodes[i])
    end
    return new_nodes
end

#=
Remove extra columns.
Insert Project(select) above aggregate and join
=#
function prune_column(nodes::Array{Any,1})
end

#=
Combine two or more filters using AND on condition which are on
same table without any other inner query operations
=#
function combine_filters()
end

#=
If select/projection operator only use one of the table columns after join then
we can safely remove that join.
TODO check with ParallelAccelerator
=#
function remove_unnessary_joins()
end

#=
Remove always true and false filters
=#
function simplify_filters()
end

#=
Simplify expressions whose results can be determined from only one side.
TODO check with ParallelAccelerator
=#
function simplify_booleans()
end

#=
Find table name for the given filter condition
This necessary to replace the table name if you move filter
above join or aggregates
TODO make it short
=#
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

#=
Replaces condition variable in symbol table with correct table
TODO make it short
=#
function replace_cond_in_linfo(linfo,cond_var,table_name)
    for i = 1:length(linfo.var_defs)
        if string(linfo.var_defs[i].name) == cond_var
            arr = split(string(linfo.var_defs[i].name),'#')
            if length(arr) > 2
                linfo.var_defs[i].name = string("#",table_name,"#",arr[3])
            end
        end
    end
end

#=
Replaces table name in the filter condition(mmap)
e.g
 table1#cond_e = table1#col1 > 1 => table2#cond_e = table2#col1 > 1
=#
function replace_table_in_cond(node,table_name)
    arr1 = split(string(node.args[1]),'#')
    arr2 = split(string(node.args[2].args[1][1].name),'#')

    node.args[1] = Symbol(string("#",table_name,"#",arr1[3]))
    node.args[2].args[1][1] = Symbol(string("#",table_name,"#",arr2[3]))
end

#=
Replaces table name and columns accordingly in the filter node after moving
=#
function replace_table_in_filter_node(node,table_name,tableCols)

    # node.args[2] gives output table
    out_table_name = string(table_name, "_fil")
    node.args[2] = Symbol(out_table_name)
    # New table must be added in tableCols which will have same columns as input
    tableCols[Symbol(out_table_name)] = tableCols[Symbol(table_name)]
    # node.args[3] gives input table
    node.args[3] = Symbol(table_name)
    # replace in condition expression (could be removed in future)
    # args[1] gives cond_rhs
    # args[7] gives cond_lhs
    arr1 = split(string(node.args[1]),'#')
    arr2 = split(string(node.args[7].args[2].name),'#')
    node.args[1] = getColName(Symbol(table_name),Symbol(arr1[3]))
    node.args[7].args[2] = getColName(Symbol(table_name),Symbol(arr2[3]))
    # Replace columns list with new columns
    node.args[4] = tableCols[Symbol(table_name)]
    # Replace output and input table columns array with new table columns
    node.args[5] = []
    node.args[6] = []
    for (index, col) in enumerate(tableCols[node.args[2]])
        append!(node.args[5], [getColName(Symbol(out_table_name),col)])
        append!(node.args[6], [getColName(Symbol(table_name),col)])
    end
    # Return output filter table and its columns list
    return node.args[2], node.args[5]
end

end # DataTablePass
