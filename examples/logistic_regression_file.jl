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

@acc hpat function logistic_regression(iterations::Int64, file_name)

    points = DataSource(Matrix{Float64},HDF5,"/points", file_name)
    responses = DataSource(Vector{Float64},HDF5,"/responses", file_name)
    D = size(points,1) # number of features
    N = size(points,2) # number of instances
    labels = reshape(responses,1,N)
    w = reshape(2.0.*rand(D)-1.0,1,D)

    for i in 1:iterations
       w -= ((1.0./(1.0.+exp(-labels.*(w*points))).-1.0).*labels)*points'
    end
    w
end

function main()
    doc = """logistic_regression.jl

Logistic regression statistical method.

Usage:
  logistic_regression.jl -h | --help
  logistic_regression.jl [--iterations=<iterations>] [--instances=<instances>]

Options:
  -h --help                  Show this screen.
  --iterations=<iterations>  Specify number of iterations; defaults to 20.
  --instances=<instances>    Specify number of instances; defaults to 10^7.
"""
    arguments = docopt(doc)

    if (arguments["--iterations"] != nothing)
        iterations = parse(Int, arguments["--iterations"])
    else
        iterations = 20
    end

    if (arguments["--instances"] != nothing)
        instances = parse(Int, arguments["--instances"])
    else
        instances = 10^7
    end

    srand(0)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    pes = MPI.Comm_size(MPI.COMM_WORLD)

    if rank==0 println("iterations = ", iterations) end
    if rank==0 println("instances = ", instances) end

    tic()
    logistic_regression(2,4096)
    time = toq()
    if rank==0 println("SELFPRIMED ", time) end
    MPI.Barrier(MPI.COMM_WORLD)

    tic()
    W = logistic_regression(iterations, instances)
    time = toq()
    if rank==0 println("result = ", W) end
    if rank==0 println("rate = ", iterations / time, " iterations/sec") end
    if rank==0 println("SELFTIMED ", time) end

end

main()

