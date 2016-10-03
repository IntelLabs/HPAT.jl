module IndexTest

using HPAT

#HPAT.DistributedPass.set_debug_level(3)

@acc hpat function index_test(numCenter, file_name)
    points = DataSource(Matrix{Float64},HDF5,"/points", file_name)
    D,N = size(points) # number of features, instances
    centroids :: Array{Float64,2} = rand(D, numCenter)

    dist::Array{Array{Float64,1},1} = [ Float64[sqrt(sum((points[:,i+j]-centroids[:,j]).^2)) for j in 1:numCenter] for i in 1:(N-numCenter)]
    labels::Array{Int,1} = [indmin(dist[i]) for i in 1:(N-numCenter)]
    return sum(labels)
end

function main()

end

using HDF5
using MPI
using Base.Test

using HPAT.Partitioning
using HPAT.SEQ
using HPAT.TWO_D
using HPAT.ONE_D


function main()
    if MPI.Comm_rank(MPI.COMM_WORLD)==0 
        h5write("test1_1.hdf5", "/points", [1 2 3 4; 5 6 7 8]) 
    end
    a = index_test(2, "test1_1.hdf5")
    println(HPAT.get_saved_array_partitionings())

    # every partitoning should be sequential
    for p in values(HPAT.get_saved_array_partitionings())
        @test p==SEQ
    end

    if MPI.Comm_rank(MPI.COMM_WORLD)==0 
        rm("test1_1.hdf5")
    end

end

end

IndexTest.main()
