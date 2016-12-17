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

@acc hpat function linear_regression(iterations, file_name)
    points = DataSource(Matrix{Float64},HDF5,"/points", file_name)
    labels = DataSource(Matrix{Float64},HDF5,"/labels", file_name)
    D,N = size(points) # number of features,samples
    p = size(labels,1) # number of functions
    w = zeros(p,D)
    alphaN = 0.01/N

    for i in 1:iterations
       w -= alphaN*((w*points)-labels)*points'
    end
    w
end

function main()
    doc = """linear regression statistical method.

Usage:
  linear_regression.jl -h | --help
  linear_regression.jl [--iterations=<iterations>] [--file=<file>]

Options:
  -h --help                  Show this screen.
  --iterations=<iterations>  Specify number of iterations; defaults to 20.
  --file=<file>              Specify input file; defaults to HPAT's default generated data file.

"""
    arguments = docopt(doc)
    iterations = 20
    file_name = HPAT.getDefaultDataPath()*"linear_regression.hdf5"

    if (arguments["--iterations"] != nothing)
        iterations = parse(Int, arguments["--iterations"])
    end

    if (arguments["--file"] != nothing)
        file_name = arguments["--file"]
    end 

    W = linear_regression(iterations, file_name)

    if MPI.Comm_rank(MPI.COMM_WORLD)==0 println("result = ", W) end
end

main()

