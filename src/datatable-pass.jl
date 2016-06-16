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

    # transform body
    #body.args = from_toplevel_body(body.args, state)
    @dprintln(1,"DataTablePass.from_root returns function = ", function_name, " ast = ", body)
    return linfo, body
end

end # DataTablePass
