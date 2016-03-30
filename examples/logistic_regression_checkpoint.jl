#=
Copyright (c) 2015, Intel Corporation
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

using HPAT 
using MPI
using DocOpt

#CompilerTools.OptFramework.set_debug_level(3)
#CompilerTools.CFGs.set_debug_level(3)
#CompilerTools.Loops.set_debug_level(3)
#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.ParallelIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)
#HPAT.Checkpointing.set_debug_level(3)
#HPAT.CGenPatternMatch.set_debug_level(3)
#HPAT.set_debug_level(3)
#HPAT.Checkpointing.setCheckpointDebug(50)    # Do checkpoints every 50 seconds.
#CompilerTools.LivenessAnalysis.set_debug_level(5)

@acc hpat_checkpoint function logistic_regression(iterations, file_name)
    points = DataSource(Matrix{Float32},HDF5,"/points", file_name)
    responses = DataSource(Vector{Float32},HDF5,"/responses", file_name)
    D = size(points,1) # number of features
    N = size(points,2) # number of instances
    labels = reshape(responses,1,N)
    w = reshape(2.0f0.*rand(Float32,D)-1.0f0,1,D)

    for i in 1:iterations
       w -= ((1.0f0./(1.0f0.+exp(-labels.*(w*points))).-1.0f0).*labels)*points'
    end
    w
end

function main()
    doc = """Logistic regression statistical method.

Usage:
  logistic_regression.jl -h | --help
  logistic_regression.jl [--iterations=<iterations>] [--file=<file>] [--restart]

Options:
  -h --help                  Show this screen.
  --iterations=<iterations>  Specify number of iterations; defaults to 20.
  --file=<file>              Specify input file; defaults to HPAT's default generated data file.
  --restart                  Restart the program from the last checkpoint taken.

"""
    arguments = docopt(doc)

    if (arguments["--iterations"] != nothing)
        iterations = parse(Int, arguments["--iterations"])
    else
        iterations = 20
    end

    if (arguments["--file"] != nothing)
        file_name::ASCIIString = arguments["--file"]
    else
        file_name = HPAT.getDefaultDataPath()*"logistic_regression.hdf5"
    end 

    do_restart = arguments["--restart"]

    srand(0)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    pes = MPI.Comm_size(MPI.COMM_WORLD)

    if rank==0 println("iterations = ", iterations) end
    if rank==0 println("file= ", file_name) end

    tic()
    if do_restart
        HPAT.restart(logistic_regression, 0, file_name)
    else
        logistic_regression(0,file_name)
    end
    time = toq()
    if rank==0 println("SELFPRIMED ", time) end
    MPI.Barrier(MPI.COMM_WORLD)

    tic()
    if do_restart
        W = HPAT.restart(logistic_regression, iterations, file_name)
    else
        W = logistic_regression(iterations, file_name)
    end
    time = toq()
    if rank==0 println("result = ", W) end
    if rank==0 println("rate = ", iterations / time, " iterations/sec") end
    if rank==0 println("SELFTIMED ", time) end
end

main()

