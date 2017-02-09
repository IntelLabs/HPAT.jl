module SinkTest

using HPAT

@acc hpat function sink(file3)
    y = randn(10)
    DataSink(y,HDF5,"/y", file3)
end

@acc hpat function sink2(file3)
    y = randn(3,10)
    DataSink(y,HDF5,"/y2", file3)
end

@acc hpat function sink_txt(file3)
    y = randn(10)
    DataSink(y, TXT, file3)
end

@acc hpat function sink_txt2(file3)
    y = randn(3,10)
    DataSink(y, TXT, file3)
end


using MPI

function main()
    SinkTest.sink("y.hdf5")
    SinkTest.sink2("y.hdf5")
    SinkTest.sink_txt("y.txt")
    SinkTest.sink_txt2("y2.txt")
    if MPI.Comm_rank(MPI.COMM_WORLD)==0 rm("y.hdf5") end
end

end

SinkTest.main()
