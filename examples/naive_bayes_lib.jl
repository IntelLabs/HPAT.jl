using HPAT 
using HPAT.API.NaiveBayes
using MPI

@acc hpat function calcNaiveBayes(file_name)
    num_classes = 20
    points = DataSource(Matrix{Float64},HDF5,"/points", file_name)
    labels = DataSource(Vector{Float64},HDF5,"/labels", file_name)
    coeffs = NaiveBayes(points, labels, num_classes)
    return coeffs
end

c2 = calcNaiveBayes(HPAT.getDefaultDataPath()*"/naivebayes_train_large.hdf5")

if MPI.Comm_rank(MPI.COMM_WORLD)==0	println("Coeffs: ", c2) end

