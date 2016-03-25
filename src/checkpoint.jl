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

module Checkpointing

import ParallelAccelerator

using CompilerTools
import CompilerTools.DebugMsg
DebugMsg.init()

import HPAT

# information about AST gathered and used in Checkpointing
type DomainState
    LambdaVarInfo :: CompilerTools.LambdaHandling.LambdaVarInfo
end

unique_num = 1
"""
If we need to generate a name and make sure it is unique then include an monotonically increasing number.
"""
function get_unique_num()
    ret = unique_num
    global unique_num = unique_num + 1
    ret
end

single_node_mttf = 4000.0   # default to 4000 hours MTTF of a single node
single_node_faults_per_million = 1000000.0 / single_node_mttf
checkpoint_time = 1.0 / 3600.0  # 1 minute checkpoint time converted to hours

# ENTRY to checkpointing
function from_root(function_name, ast :: Expr, with_restart :: Bool)
    @assert ast.head == :lambda "Input to Checkpointing should be :lambda Expr"
    @dprintln(1,"Starting main Checkpointing.from_root.  function = ", function_name, " ast = ", ast, " with_restart = ", with_restart)

    state::DomainState = DomainState(CompilerTools.LambdaHandling.lambdaExprToLambdaVarInfo(ast))
    
    body = CompilerTools.LambdaHandling.getBody(ast)

    lives = ParallelAccelerator.ParallelIR.computeLiveness(ast)
    @dprintln(3,"lives = ", lives)
    loop_info = CompilerTools.Loops.compute_dom_loops(lives.cfg)
    @dprintln(3,"loop_info = ", loop_info)

    loops = loop_info.loops

    if length(loops) != 1
        @dprintln(0,"Checkpointing.from_root currently only supports functions with exactly one loop.  ", function_name, " has ", length(loops), " loops.")
        return ast
    end
    the_loop = loops[1]

    loop_entry = the_loop.head
    loop_entry_bb = lives.cfg.basic_blocks[loop_entry]
    @dprintln(3,"loop_entry = ", loop_entry)
    liveness_loop_entry = CompilerTools.LivenessAnalysis.getBasicBlockFromBlockNumber(loop_entry, lives)
    @dprintln(3,"liveness_loop_entry = ", liveness_loop_entry)
    bb_loop_members = map(x -> CompilerTools.LivenessAnalysis.getBasicBlockFromBlockNumber(x, lives), the_loop.members)
    @dprintln(3,"bb_loop_members = ", bb_loop_members)
    loop_live_in = liveness_loop_entry.live_in
    @dprintln(3,"loop_live_in = ", loop_live_in)
    loop_def = reduce((x,y) -> union(x,y), Set{CompilerTools.LambdaHandling.SymGen}(), map(x -> x.def, bb_loop_members))
    @dprintln(3,"loop_def = ", loop_def)
    live_in_and_def = intersect(loop_live_in, loop_def)
    @dprintln(3,"live_in_and_def = ", live_in_and_def)
    assert(!isempty(live_in_and_def))
    
    pre_loop_stmts = Any[]
    checkpoint_timer_sn = ParallelAccelerator.ParallelIR.createStateVar(state, "__hpat_checkpoint_timer", Int32, ParallelAccelerator.ParallelIR.ISASSIGNED)

    push!(pre_loop_stmts, ParallelAccelerator.ParallelIR.mk_assignment_expr(checkpoint_timer_sn, ParallelAccelerator.ParallelIR.TypedExpr(Int32, :call, TopNode(:hpat_get_sec_since_epoch))))

    CompilerTools.Loops.insertNewBlockBeforeLoop(the_loop, lives.cfg, pre_loop_stmts)
    @dprintln(3,"CFG after insert = ", lives.cfg)

    # Create the checkpoint function as a string and then parse/eval to force it into existence.
    # The function takes the last checkpoint time.  If enough time has expired then do the checkpoint and return the current time.
    # If checkpoint time has not arrived then return the last checkpoint time.
    # The time between checkpoints is sqrt( 2 * time_to_checkpoint * system_mttf).
    # We get system_mttf by assuming some reasonable single_node_mttf, converting to failures per million hours, multiple by the
    # number of nodes in the system to get full system failures per million hours and then convert back to full system mttf (in hours).
    checkpoint_func_name = string("__hpat_checkpoint_func_", unique_num)

    # Creates the first line of the function with these characteristics.
    #   1. The function name is __hpat_checkpoint_func_ followed by a unique number to make the function name unique.
    #   2. The first parameter is the last checkpoint time.
    #   3. Other parameters are the elements that need to go in the checkpoint file.
    argument_names = [ string("arg",i) for i = 1:length(live_in_and_def) ]
    checkpoint_func_str = string(                     "function ", checkpoint_func_name, "(start_time, num_pes, ", foldl((a,b) -> "$a, $b", argument_names), ")\n")
    checkpoint_func_str = string(checkpoint_func_str, "    system_faults_per_million_hours = num_pes * ", single_node_faults_per_million, "\n")
    checkpoint_func_str = string(checkpoint_func_str, "    system_mttf::Float64 = 1000000.0 / system_faults_per_million_hours\n")
    checkpoint_func_str = string(checkpoint_func_str, "    cur_time = hpat_get_sec_since_epoch()\n")
    checkpoint_func_str = string(checkpoint_func_str, "    if ((cur_time - start_time) / 3600.0) > sqrt(2 * system_mttf * ", checkpoint_time, ")\n")
    checkpoint_func_str = string(checkpoint_func_str, "        return cur_time\n")
    checkpoint_func_str = string(checkpoint_func_str, "    end\n")
    checkpoint_func_str = string(checkpoint_func_str, "    return start_time\n")
    checkpoint_func_str = string(checkpoint_func_str, "end\n")
    @dprintln(3,"checkpoint_func_str = ", checkpoint_func_str)
    Main.eval(parse(checkpoint_func_str))   # Force the new checkpoint function into existence.

    assert(!isempty(loop_entry_bb.statements))
    first_loop_entry_stmt = loop_entry_bb.statements[1]
    # We can use this version with the __hpat_num_pes variable once we are able to run the checkpointing pass after the distributed pass.
    #call_checkpoint_expr  = ParallelAccelerator.ParallelIR.mk_assignment_expr(checkpoint_timer_sn, ParallelAccelerator.ParallelIR.TypedExpr(Uint64, :call, TopNode(symbol(checkpoint_func_name)), checkpoint_timer_sn, :__hpat_num_pes, live_in_and_def...))
    pes_expr = ParallelAccelerator.ParallelIR.TypedExpr(Int32, :call, TopNode(:hps_dist_num_pes))
    #pes_expr = 8
    call_checkpoint_expr  = ParallelAccelerator.ParallelIR.mk_assignment_expr(checkpoint_timer_sn, ParallelAccelerator.ParallelIR.TypedExpr(Int32, :call, GlobalRef(Main,symbol(checkpoint_func_name)), checkpoint_timer_sn, pes_expr, live_in_and_def...))
    CompilerTools.CFGs.insertStatementBefore(lives.cfg, loop_entry_bb, first_loop_entry_stmt.index, call_checkpoint_expr)

    # Reconstitute the body with the new basic block.
    body.args = CompilerTools.CFGs.createFunctionBody(lives.cfg)
    # Re-create the lambda.
    new_ast = CompilerTools.LambdaHandling.LambdaVarInfoToLambdaExpr(state.LambdaVarInfo, body)

    @dprintln(1,"Checkpointing.from_root returns function = ", function_name, " ast = ", new_ast)
    return new_ast
end

end # module

