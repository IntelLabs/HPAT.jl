using HPAT 
using MPI

@acc hpat function calcKmeans(k::Int64, file_name)
    points = DataSource(Matrix{Float64},HDF5,"/points",file_name)
    clusters = Kmeans(points, k, 10)
    return clusters
end

K = 3

if (length(ARGS) > 0)
	K = parse(Int, ARGS[1])
end

c2 = calcKmeans(K, HPAT.getDefaultDataPath()*"/kmeans_large.hdf5")

if MPI.Comm_rank(MPI.COMM_WORLD)==0	println("Centroids: ", c2) end

