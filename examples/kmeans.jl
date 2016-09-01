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

@acc hpat function kmeans(numCenter, iterNum, file_name)
    points = DataSource(Matrix{Float64},HDF5,"/points", file_name)
    D,N = size(points) # number of features, instances
    centroids = rand(D, numCenter)

    for l in 1:iterNum
        dist::Array{Array{Float64,1},1} = [ [sqrt(sum((points[:,i]-centroids[:,j]).^2)) for j in 1:numCenter] for i in 1:N]
        labels :: Array{Int, 1} = [indmin(dist[i]) for i in 1:N]
        centroids :: Array{Float64, 2} = [ sum(points[j,labels.==i])/sum(labels.==i) for j in 1:D, i in 1:numCenter]
    end 
    return centroids
end

function main()
    doc = """K-means clustering algorithm.

Usage:
  kmeans.jl -h | --help
  kmeans.jl [--iterations=<iterations>] [--file=<file>] [--centers=<centers>]

Options:
  -h --help                  Show this screen.
  --iterations=<iterations>  Specify number of iterations; defaults to 20.
  --file=<file>              Specify input file; defaults to HPAT's default generated data file.
  --centers=<centers>        Specify number of centers; defaults to 5.

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
        file_name = HPAT.getDefaultDataPath()*"kmeans_large.hdf5"
    end 

    if (arguments["--centers"] != nothing)
        numCenter = parse(Int, arguments["--centers"])
    else
        numCenter = 5
    end

    srand(0)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    pes = MPI.Comm_size(MPI.COMM_WORLD)

    if rank==0 println("iterations = ", iterations) end
    if rank==0 println("file= ", file_name) end
    if rank==0 println("centers= ", numCenter) end

    tic()
    kmeans(numCenter, 2, file_name)
    time = toq()
    if rank==0 println("SELFPRIMED ", time) end
    MPI.Barrier(MPI.COMM_WORLD)

    tic()
    centroids_out = kmeans(numCenter, iterations, file_name)
    time = toq()
    if rank==0 println("result = ", centroids_out) end
    if rank==0 println("rate = ", iterations / time, " iterations/sec") end
    if rank==0 println("SELFTIMED ", time) end

end

main()
