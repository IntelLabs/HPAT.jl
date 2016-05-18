module API
using HDF5
import HPAT

export data_source_HDF5, data_source_TXT
operators = [GlobalRef(HPAT.API,:data_source_HDF5),GlobalRef(HPAT.API,:data_source_TXT)]

@noinline function data_source_HDF5(T::DataType, var::AbstractString, file_name::AbstractString)
    arr::T = h5read(file_name, var)
    return arr
end

@noinline function data_source_TXT(T::DataType, file_name::AbstractString)
    arr::T = readdlm(file_name, ' ', eltype(T))
    return arr
end

end
