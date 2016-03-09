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


module HPAT

ENV["CGEN_MPI_COMPILE"]=1

using CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools
using CompilerTools.OptFramework
using CompilerTools.OptFramework.OptPass
using ParallelAccelerator
using ParallelAccelerator.Driver
using CompilerTools.AstWalker
using MPI

export hpat, @acc, @noacc


include("distributed-pass.jl")
include("domain-pass.jl")
include("capture-api.jl")
include("cgen-hpat-pattern-match.jl")

# add HPAT pattern matching code generators to CGen
ParallelAccelerator.CGen.setExternalPatternMatchCall(CGenPatternMatch.pattern_match_call)
ParallelAccelerator.CGen.setExternalPatternMatchAssignment(CGenPatternMatch.from_assignment_match_dist)

function ns_to_sec(x)
  x / 1000000000.0
end

function runDistributedPass(func :: GlobalRef, ast :: Expr, signature :: Tuple)
  dir_start = time_ns()
  code = DistributedPass.from_root(string(func.name), ast)
  dir_time = time_ns() - dir_start
  @dprintln(2, "Distributed code = ", code)
  @dprintln(1, "accelerate: DistributedPass conversion time = ", ns_to_sec(dir_time))
  return code
end

function runDomainPass(func :: GlobalRef, ast :: Expr, signature :: Tuple)
  dir_start = time_ns()
  code = DomainPass.from_root(string(func.name), ast)
  dir_time = time_ns() - dir_start
  @dprintln(2, "Domain code = ", code)
  @dprintln(1, "accelerate: DomainPass conversion time = ", ns_to_sec(dir_time))
  return code
end

"""
A macro pass that translates extensions such as DataSource()
"""
function captureHPAT(func, ast, sig)
  AstWalk(ast, CaptureAPI.process_node, nothing)
  return ast
end

# initialize set of compiler passes HPAT runs
const hpat = [ OptPass(captureHPAT, PASS_MACRO),
               OptPass(captureOperators, PASS_MACRO),
               OptPass(toCartesianArray, PASS_MACRO),
               OptPass(runDomainPass, PASS_TYPED),
               OptPass(toDomainIR, PASS_TYPED),
               OptPass(toParallelIR, PASS_TYPED),
               OptPass(runDistributedPass, PASS_TYPED),
               OptPass(toFlatParfors, PASS_TYPED),
               OptPass(toCGen, PASS_TYPED) ]

append!(ParallelAccelerator.DomainIR.funcIgnoreList, DomainPass.generatedFuncs)


HPAT_default_datapath = "input_data/"

if haskey(ENV, "HPAT_DEFAULT_DATA")
    HPAT_default_datapath = ENV["HPAT_DEFAULT_DATA"]
elseif haskey(ENV, "SCRATCH")
    HPAT_default_datapath = ENV["SCRATCH"]*"/input_data/"
else
    HPAT_default_datapath = joinpath(dirname(@__FILE__), "..")*"/input_data/"
end

if !isdir(HPAT_default_datapath)
    mkdir(HPAT_default_datapath)
end

function getDefaultDataPath()
    return HPAT_default_datapath
end

function setDefaultDataPath(path::AbstractString)
    HPAT_default_datapath = path
end

function HPAT_finalize()
    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
end

atexit(HPAT_finalize)

end # module
