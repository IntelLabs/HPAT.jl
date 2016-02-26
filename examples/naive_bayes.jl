using HPAT 
using MPI

@acc hpat function calcNaiveBayes(file_name)
    num_classes = 20
    points = DataSource(Matrix{Float64},HDF5,"/points", file_name)
    labels = DataSource(Vector{Float64},HDF5,"/labels", file_name)
    coeffs = HPS.NaiveBayes(points, labels, num_classes)
    return coeffs
end

c2 = calcNaiveBayes(ENV["SCRATCH"]*"/benchmark_data/naivebayes_train_small.hdf5")
rank = MPI.Comm_rank(MPI.COMM_WORLD)
pes = MPI.Comm_size(MPI.COMM_WORLD)


MPI.Barrier(MPI.COMM_WORLD)
t1 = time_ns()
c2 = calcNaiveBayes(ENV["SCRATCH"]*"/benchmark_data/naivebayes_train_large.hdf5")
t2 = time_ns()

if rank==0
	println("nodes: ", pes, "\nnaive bayes exec time: ", (t2-t1)/1.0e9, "\nCoeffs: ", c2)
end

MPI.Barrier(MPI.COMM_WORLD)
