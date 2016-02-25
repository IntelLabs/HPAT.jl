module HPAT

using CompilerTools.DebugMsg
DebugMsg.init()

using CompilerTools
using CompilerTools.OptFramework
using CompilerTools.OptFramework.OptPass
using ParallelAccelerator
using ParallelAccelerator.Driver


export hpat, @acc, @noacc


# initialize set of compiler passes HPAT runs
const hpat = [ OptPass(captureOperators, PASS_MACRO),
               OptPass(toCartesianArray, PASS_MACRO),
               OptPass(toDomainIR, PASS_TYPED),
               OptPass(toParallelIR, PASS_TYPED),
               OptPass(toDistributedIR, PASS_TYPED),
               OptPass(toFlatParfors, PASS_TYPED),
               OptPass(toCGen, PASS_TYPED) ]

end # module
