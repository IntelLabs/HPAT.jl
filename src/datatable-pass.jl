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

# ENTRY to datatable-pass
function from_root(function_name, ast::Tuple)
    @dprintln(1,"Starting main DataTablePass.from_root.  function = ", function_name, " ast = ", ast)
    (linfo, body) = ast
    lives = computeLiveness(body, linfo)
    tableCols, tableTypes = get_table_meta(body)
    # transform body
    body.args = from_toplevel_body(body.args,tableCols,linfo)
    @dprintln(1,"DataTablePass.from_root returns function = ", function_name, " ast = ", body)
    return LambdaVarInfoToLambda(linfo, body.args)
end

# nodes are :body of AST
function from_toplevel_body(nodes::Array{Any,1},tableCols,linfo)
    res::Array{Any,1} = []
    nodes = push_filter_up(nodes,tableCols,linfo)
    @dprintln(3,"body after query optimizations ", nodes)
    return nodes
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
            pos=i
        end
        if isa(nodes[i], Expr) && nodes[i].head==:filter && hit_join
            # TODO change condition in filter expression too
            new_filter_node = nodes[i]
            cond = nodes[i-1]
            table_name = find_table_from_cond(tableCols,cond)
            replace_cond_in_linfo(linfo,cond,table_name)
            new_cond_node =  AstWalk(nodes[i-1], replace_table_in_cond,table_name)
            # remove condition node above filter node
            pop!(new_nodes)
            splice!(new_nodes,pos:1,[new_cond_node,new_filter_node])
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
function find_table_from_cond(tableCols,cond)
    # TODO : This hack should be changes.
    # I am assuming that second element has column name
    s = string(cond.args[2].args[2].name)
    arr = split(s,'#')
    col_name = arr[3]
    for k in keys(tableCols)
        arr = tableCols[k]
        for (index, value) in enumerate(arr)
            curr_col = string(value)
            if curr_col == col_name
                return string(k)
            end
        end
    end
end

#=
Replaces condition variable in symbol table with correct table
TODO make it short
=#
function replace_cond_in_linfo(linfo,cond,table_name)
    cond_var = string(cond.args[1])
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
Replaces table name in the filter condition
e.g
 table1#cond_e = table1#col1 > 1 => table2#cond_e = table2#col1 > 1
=#
function replace_table_in_cond(node::Symbol, table_name, top_level_number, is_top_level, read)
    arr = split(string(node),'#')
    return (length(arr) > 2) ? Symbol(string("#",table_name,"#",arr[3])) : CompilerTools.AstWalker.ASTWALK_RECURSE

end
function replace_table_in_cond(node::ANY, table_name, top_level_number, is_top_level, read)
    return CompilerTools.AstWalker.ASTWALK_RECURSE
end
function replace_table_in_cond(node::SymbolNode, table_name, top_level_number, is_top_level, read)
    arr = split(string(node.name),'#')
    return (length(arr) > 2) ? Symbol(string("#",table_name,"#",arr[3])) : CompilerTools.AstWalker.ASTWALK_RECURSE
end

end # DataTablePass
