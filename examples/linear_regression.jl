using HPAT 
using MPI

@acc hpat function calcLinearRegression(file_name)
    points = DataSource(Matrix{Float64},HDF5,"/points", file_name)
    responses = DataSource(Vector{Float64},HDF5,"/responses", file_name)
    coeffs = HPS.LinearRegression(points, responses)
    return coeffs
end

c2 = calcLinearRegression(ENV["SCRATCH"]*"/benchmark_data/linear_regression_train_small.hdf5")
rank = MPI.Comm_rank(MPI.COMM_WORLD)
pes = MPI.Comm_size(MPI.COMM_WORLD)


MPI.Barrier(MPI.COMM_WORLD)
t1 = time_ns()
c2 = calcLinearRegression(ENV["SCRATCH"]*"/benchmark_data/linear_regression_train_large.hdf5")
t2 = time_ns()

if rank==0
	println("nodes: ", pes, "\nlinear regression exec time: ", (t2-t1)/1.0e9, "\nCoeffs: ", c2)
end

MPI.Barrier(MPI.COMM_WORLD)
