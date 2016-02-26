using HPAT 
using MPI

@acc hpat function calc1Dsum(file_name)
    arr = DataSource(Vector{Float64},HDF5,"/labels",file_name)
    return sum(arr)
end

rank = MPI.Comm_rank(MPI.COMM_WORLD)
pes = MPI.Comm_size(MPI.COMM_WORLD)

# warm up
sm2 = calc1Dsum(ENV["SCRATCH"]*"/benchmark_data/1D_small.hdf5")
MPI.Barrier(MPI.COMM_WORLD)
t1 = time_ns()
sm2 = calc1Dsum(ENV["SCRATCH"]*"/benchmark_data/1D_large.hdf5")
t2 = time_ns()

if rank==0
	println("nodes: ", pes, "\n1D exec time: ", (t2-t1)/1.0e9, "\nchecksum: ", sm2)
end

