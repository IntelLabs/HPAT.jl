module SinkTest

using HPAT

@acc hpat function sink(file3)
    y = randn(10)
    DataSink(y,HDF5,"/y", file3)
end


end

SinkTest.sink("y.hdf5")
using MPI
if MPI.Comm_rank(MPI.COMM_WORLD)==0 rm("y.hdf5") end
