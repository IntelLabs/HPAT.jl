using HPAT 
using MPI

@acc hpat function calcSum(file_name)
    vals = DataSource(Vector{Float64},TXT, file_name)
    return sum(vals)
end

rank = MPI.Comm_rank(MPI.COMM_WORLD)
pes = MPI.Comm_size(MPI.COMM_WORLD)


sm1 = calcSum(ENV["SCRATCH"]*"/benchmark_data/1D_small.csv")
MPI.Barrier(MPI.COMM_WORLD)
t1 = time_ns()
sm2 = calcSum(ENV["SCRATCH"]*"/benchmark_data/1D_large.csv")
t2 = time_ns()

if rank==0
	println("nodes: ", pes, "\ntime: ", (t2-t1)/1.0e9, "\nchecksum: ", sm1)
end

