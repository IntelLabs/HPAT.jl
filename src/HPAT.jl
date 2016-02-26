module HPAT

using CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools
using CompilerTools.OptFramework
using CompilerTools.OptFramework.OptPass
using ParallelAccelerator
using ParallelAccelerator.Driver
using CompilerTools.AstWalker

export hpat, @acc, @noacc


include("distributed-pass.jl")
include("capture-api.jl")

function ns_to_sec(x)
  x / 1000000000.0
end

function runDistributedPass(func :: GlobalRef, ast :: Expr, signature :: Tuple)
  dir_start = time_ns()
  code = DistributedPass.from_root(string(func.name), ast)
  dir_time = time_ns() - dir_start
  @dprintln(3, "Distributed code = ", code)
  @dprintln(1, "accelerate: DistributedPass conversion time = ", ns_to_sec(dir_time))
  return code
end

function runDomainPass(func :: GlobalRef, ast :: Expr, signature :: Tuple)
  dir_start = time_ns()
  code = DistributedPass.from_root(string(func.name), ast)
  dir_time = time_ns() - dir_start
  @dprintln(3, "Distributed code = ", code)
  @dprintln(1, "accelerate: DistributedPass conversion time = ", ns_to_sec(dir_time))
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


end # module
