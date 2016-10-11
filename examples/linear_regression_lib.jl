using HPAT 
using HPAT.API.LinearRegression
using MPI

@acc hpat function calcLinearRegression(file_name)
    points = DataSource(Matrix{Float64},HDF5,"/points", file_name)
    responses = DataSource(Vector{Float64},HDF5,"/responses", file_name)
    coeffs = LinearRegression(points, responses)
    return coeffs
end

c2 = calcLinearRegression(HPAT.getDefaultDataPath()*"/linear_regression_train_large.hdf5")

if MPI.Comm_rank(MPI.COMM_WORLD)==0	println("linear regression Coeffs: ", c2) end

