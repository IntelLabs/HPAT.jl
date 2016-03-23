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

using CompilerTools.LambdaHandling
using CompilerTools.Helper

import HPAT

# ENTRY to checkpointing
function from_root(function_name, ast :: Expr, with_restart :: Bool)
    @assert ast.head == :lambda "Input to Checkpointing should be :lambda Expr"
    @dprintln(1,"Starting main Checkpointing.from_root.  function = ", function_name, " ast = ", ast, " with_restart = ", with_restart)

    state::DomainState = DomainState(CompilerTools.LambdaHandling.lambdaExprToLambdaVarInfo(ast))
    
    old_body = CompilerTools.LambdaHandling.getBody(ast)

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
    @dprintln(3,"loop_entry = ", loop_entry)
    liveness_loop_entry = CompilerTools.LivenessAnalysis.getBasicBlockFromBlockNumber(loop_entry, lives)
    @dprintln(3,"liveness_loop_entry = ", liveness_loop_entry)
    bb_loop_members = map(x -> CompilerTools.LivenessAnalysis.getBasicBlockFromBlockNumber(x, lives), the_loop.members)
    @dprintln(3,"bb_loop_members = ", bb_loop_members)
    loop_live_in = liveness_loop_entry.live_in
    @dprintln(3,"loop_live_in = ", loop_live_in)
    loop_def = reduce((x,y) -> union(x,y), Set{SymGen}(), map(x -> x.def, bb_loop_members))
    @dprintln(3,"loop_def = ", loop_def)
    live_in_and_def = intersect(loop_live_in, loop_def)
    @dprintln(3,"live_in_and_def = ", live_in_and_def)

#    body = TypedExpr(CompilerTools.LambdaHandling.getReturnType(state.linfo), :body, from_toplevel_body(CompilerTools.LambdaHandling.getBody(state.linfo), state)...)
#    new_ast = CompilerTools.LambdaHandling.LambdaVarInfoToLambdaExpr(state.linfo, body)
    new_ast = ast
    @dprintln(1,"Checkpointing.from_root returns function = ", function_name, " ast = ", new_ast)
    return new_ast
end

# information about AST gathered and used in Checkpointing
type DomainState
    linfo  :: LambdaVarInfo
end


end # module

