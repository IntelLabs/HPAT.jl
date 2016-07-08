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
using ParallelAccelerator.CGen
using CompilerTools.AstWalker
import MPI
using Base.pushmeta!

export hpat, hpat_checkpoint, hpat_debug, @acc, @noacc

# disable OMP if not set by user since it is slower than pure MPI
if !haskey(ENV, "CGEN_NO_OMP")
    ParallelAccelerator.CGen.disableOMP()
end

function enableOMP()
    if !haskey(ENV, "CGEN_NO_OMP")
        ParallelAccelerator.CGen.enableOMP()
    end
end

OptFramework.setSaveOriginalFunction(false)

force_parallel = true

function setForceParallel(v::Bool)
  global force_parallel=v
end

# constant block size for 2D partitioning
BLOCK_SIZE = 128

function setBlockSize(v::Int)
  global BLOCK_SIZE=v
end

# smaller value means higher precedence
@enum Partitioning SEQ=1 TWO_D=2 ONE_D=3

include("api.jl")
using HPAT.API
export data_source_HDF5, data_source_TXT
include("checkpoint.jl")
include("capture-api.jl")
include("domain-pass.jl")
include("datatable-pass.jl")
include("distributed-pass.jl")
include("cgen-hpat-pattern-match.jl")

# overwrite print so only rank 0 prints
#=function Base.print(xs...)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    if rank==0 Base.print(STDOUT, xs) end
end

function Base.println(xs...)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    if rank==0 Base.println(STDOUT, xs) end
end=#


# add HPAT pattern matching code generators to CGen
ParallelAccelerator.CGen.setExternalPatternMatchCall(CGenPatternMatch.pattern_match_call)
ParallelAccelerator.CGen.setExternalPatternMatchAssignment(CGenPatternMatch.from_assignment_match_dist)

HPAT_path = joinpath(dirname(@__FILE__), "..")

HPAT_includes = string("#include <ctime>\n#include <unordered_map>\n#include \"", HPAT_path, "/deps/include/hpat.h\"\n")
ParallelAccelerator.CGen.addCgenUserOptions(ParallelAccelerator.CGen.CgenUserOptions(HPAT_includes))

function addHpatInclude(stmts)
  ParallelAccelerator.CGen.addCgenUserOptions(ParallelAccelerator.CGen.CgenUserOptions(stmts))
end

function ns_to_sec(x)
  x / 1000000000.0
end

function runDistributedPass(func :: GlobalRef, ast, signature :: Tuple)
  dir_start = time_ns()
  code = DistributedPass.from_root(string(func.name), ast)
  dir_time = time_ns() - dir_start
  @dprintln(2, "Distributed code = ", code)
  @dprintln(1, "accelerate: DistributedPass conversion time = ", ns_to_sec(dir_time))
  return code
end

function runDataTablePass(func :: GlobalRef, ast, signature :: Tuple)
  dir_start = time_ns()
  code = DataTablePass.from_root(string(func.name), ast)
  dir_time = time_ns() - dir_start
  @dprintln(2, "DataTable code = ", code)
  @dprintln(1, "accelerate: DataTablePass conversion time = ", ns_to_sec(dir_time))
  return code
end

function runDomainPass(func :: GlobalRef, ast, signature :: Tuple)
  dir_start = time_ns()
  code = DomainPass.from_root(string(func.name), ast)
  dir_time = time_ns() - dir_start
  @dprintln(2, "Domain code = ", code)
  @dprintln(1, "accelerate: DomainPass conversion time = ", ns_to_sec(dir_time))
  return code
end

function addCheckpointing(func :: GlobalRef, ast, signature :: Tuple)
  dir_start = time_ns()
  code = Checkpointing.from_root(string(func.name), ast, false)
  dir_time = time_ns() - dir_start
  @dprintln(2, "Added checkpointing = ", code)
  @dprintln(1, "accelerate: addCheckpointing conversion time = ", ns_to_sec(dir_time))
  return code
end

function addCheckpointingRestart(func :: GlobalRef, ast, signature :: Tuple)
  dir_start = time_ns()
  code = Checkpointing.from_root(string(func.name), ast, true)
  dir_time = time_ns() - dir_start
  @dprintln(2, "Added checkpointing and restart = ", code)
  @dprintln(1, "accelerate: addCheckpointingRestart conversion time = ", ns_to_sec(dir_time))
  return code
end

type MacroState
    # columns of each table
    tableCols::Dict{Symbol,Vector{Symbol}}
    # Types of each table column
    tableTypes::Dict{Symbol,Vector{Symbol}}
    # partitioning of arrays
    array_partitioning::Dict{Symbol,Partitioning}
    function MacroState()
        new(Dict{Symbol,Vector{Symbol}}(),Dict{Symbol,Vector{Symbol}}(),Dict{Symbol,Symbol}())
    end
end

"""
A macro pass that translates extensions such as DataSource()
"""
function captureHPAT(func, ast, sig)
  macro_state = MacroState()
  AstWalk(ast, CaptureAPI.process_node, macro_state)
  pushmeta!(ast,:hpat_tables, macro_state.tableCols, macro_state.tableTypes)
  pushmeta!(ast,:hpat_partitioning, macro_state.array_partitioning)
  #println(ast)
  return ast
end

""" Adds a duplicate of the function to be checkpointed with "_restart" appended to the name.
"""
function createCheckpointFunc(func, ast, sig)
  @dprintln(1, "createCheckpointFunc func = ", func)
  new_func = deepcopy(ast)
  new_func.args[1].args[1] = symbol(string(new_func.args[1].args[1],"_restart"))
  @dprintln(1, "createCheckpointFunc new_func = ", new_func.args[1].args[1])
  return CompilerTools.OptFramework.MoreWork(ast, CompilerTools.OptFramework.WorkItem[CompilerTools.OptFramework.WorkItem(hpat_checkpoint_internal, hpat_checkpoint_internal, Any[], new_func)])
end

const hpat_debug = [ OptPass(captureHPAT, PASS_MACRO) ]

# initialize set of compiler passes HPAT runs
const hpat = [ OptPass(captureHPAT, PASS_MACRO),
               OptPass(captureOperators, PASS_MACRO),
               OptPass(toCartesianArray, PASS_MACRO),
               OptPass(runDomainPass, PASS_TYPED),
               OptPass(toDomainIR, PASS_TYPED),
               OptPass(runDataTablePass, PASS_TYPED),
               OptPass(toParallelIR, PASS_TYPED),
               OptPass(runDistributedPass, PASS_TYPED),
               OptPass(toFlatParfors, PASS_TYPED),
               OptPass(toCGen, PASS_TYPED) ]

const hpat_checkpoint =
             [ OptPass(createCheckpointFunc, PASS_MACRO),
               OptPass(captureHPAT, PASS_MACRO),
               OptPass(captureOperators, PASS_MACRO),
               OptPass(toCartesianArray, PASS_MACRO),
               OptPass(runDomainPass, PASS_TYPED),
               OptPass(addCheckpointing, PASS_TYPED),
               OptPass(toDomainIR, PASS_TYPED),
               OptPass(runDataTablePass, PASS_TYPED),
               OptPass(toParallelIR, PASS_TYPED),
               OptPass(runDistributedPass, PASS_TYPED),
               OptPass(toFlatParfors, PASS_TYPED),
               OptPass(toCGen, PASS_TYPED) ]

const hpat_checkpoint_internal =
             [ OptPass(captureHPAT, PASS_MACRO),
               OptPass(captureOperators, PASS_MACRO),
               OptPass(toCartesianArray, PASS_MACRO),
               OptPass(runDomainPass, PASS_TYPED),
               OptPass(addCheckpointingRestart, PASS_TYPED),
               OptPass(toDomainIR, PASS_TYPED),
               OptPass(runDataTablePass, PASS_TYPED),
               OptPass(toParallelIR, PASS_TYPED),
               OptPass(runDistributedPass, PASS_TYPED),
               OptPass(toFlatParfors, PASS_TYPED),
               OptPass(toCGen, PASS_TYPED) ]

append!(ParallelAccelerator.DomainIR.funcIgnoreList, [DomainPass.generatedFuncs; API.operators])
append!(ParallelAccelerator.DomainIR.exprHeadIgnoreList, DomainPass.generatedExprHeads)
ParallelAccelerator.DomainIR.setExternalCallback(DomainPass.AstWalkCallback)
ParallelAccelerator.DomainIR.setExternalLiveCB(DomainPass.live_cb)
ParallelAccelerator.DomainIR.setExternalAliasCB(DomainPass.alias_cb)


HPAT_default_datapath = "input_data/"

if haskey(ENV, "HPAT_DEFAULT_DATA")
    global HPAT_default_datapath = ENV["HPAT_DEFAULT_DATA"]
elseif haskey(ENV, "SCRATCH")
    global HPAT_default_datapath = ENV["SCRATCH"]*"/input_data/"
else
    global HPAT_default_datapath = joinpath(dirname(@__FILE__), "..")*"/input_data/"
end

if MPI.Comm_rank(MPI.COMM_WORLD)==0 && !isdir(HPAT_default_datapath)
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

function restart(func, args...)
  @dprintln(1, "HPAT restart func = ", func, " type = ", typeof(func), " args = ", args...)
  arg_type_tuple_expr = Expr(:tuple)
  arg_type_tuple_expr.args = map(x -> typeof(x), [args...])
  arg_type_tuple = eval(arg_type_tuple_expr)
  @dprintln(2, "arg_type_tuple = ", arg_type_tuple)
  gr = GlobalRef(Base.function_module(func,arg_type_tuple), symbol(func))
  @dprintln(2, "GlobalRef = ", gr)
  if haskey(CompilerTools.OptFramework.gOptFrameworkDict, gr)
    res = CompilerTools.OptFramework.gOptFrameworkDict[gr]
    @dprintln(2, "CompilerTools.OptFramework internally mapped ", gr, " to ", res)
    assert(typeof(res) == GlobalRef)
    new_gr = GlobalRef(res.mod, symbol(string(res.name,"_restart")))
    @dprintln(2, "Preparing to call the restart version of the function with name = ", new_gr)
    eval(new_gr)(args...)
  else
    throw(string("HPAT restart function could not find function to restart."))
  end
end

@noinline function __hpat_dist_broadcast(variable, size)
   convert(Int32, 1)
end

end # module
