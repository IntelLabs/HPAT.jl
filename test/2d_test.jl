using HPAT

HPAT.setBlockSize(2)

@acc hpat function mult(file1,file2)
    @partitioned(M,HPAT_2D);
    M = DataSource(Matrix{Float64},HDF5,"/M", file1)
    DataSink(M,HDF5,"/y", file2)
    return 0
end


using MPI
rank = MPI.Comm_rank(MPI.COMM_WORLD)
using HDF5
if rank==0 h5write("dtest.hdf5","/M",convert(Vector{Float64},collect(1:16))) end

mult("dtest.hdf5","testout.hdf5")
if rank==0
    A = h5read("testout.hdf5","/y")
    @test_approx_eq A convert(Vector{Float64},collect(1:16))
    rm("dtest.hdf5")
    rm("testout.hdf5")
end


