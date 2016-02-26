using HPAT 
using MPI

@acc hpat function calcKmeans(k::Int64, file_name)
    points = DataSource(Matrix{Float64},HDF5,"/points",file_name)
    clusters = HPS.kmeans(points, k)
    return clusters
end

K = 3

if (length(ARGS) > 0)
	K = parse(Int, ARGS[1])
end

c2 = calcKmeans(K,ENV["SCRATCH"]*"/benchmark_data/kmeans_small.hdf5")
rank = MPI.Comm_rank(MPI.COMM_WORLD)
pes = MPI.Comm_size(MPI.COMM_WORLD)


MPI.Barrier(MPI.COMM_WORLD)
t1 = time_ns()
c2 = calcKmeans(K,ENV["SCRATCH"]*"/benchmark_data/kmeans_large.hdf5")
t2 = time_ns()

if rank==0
	println("nodes: ", pes, "\ntime: ", (t2-t1)/1.0e9, "\nCentroids: ", c2)
end

MPI.Barrier(MPI.COMM_WORLD)
