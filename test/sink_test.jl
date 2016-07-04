module SinkTest

using HPAT

@acc hpat function sink(file3)
    y = randn(10)
    DataSink(y,HDF5,"/y", file3)
end


end


SinkTest.sink("y.hdf5")
rm("y.hdf5")
