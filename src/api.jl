module API
using HDF5
import HPAT

 
operators = [:data_source_HDF5,:data_source_TXT,:Kmeans,:LinearRegression,:NaiveBayes]

for op in operators
    @eval export $op
end

@noinline function data_source_HDF5(T::DataType, var::AbstractString, file_name::AbstractString)
    arr::T = h5read(file_name, var)
    return arr
end

@noinline function data_source_TXT(T::DataType, file_name::AbstractString)
    arr::T = readdlm(file_name, ' ', eltype(T))
    return arr
end

@noinline function Kmeans{T}(points::Matrix{T}, numCenter::Int)
    # naive backup sequential implementation
    D = size(points,1) # number of features
    N = size(points,2) # number of instances
    centroids = rand(T, D, numCenter)
    iterNum=10
    for l in 1:iterNum
        dist = [ T[sqrt(sum((points[:,i].-centroids[:,j]).^2)) for j in 1:numCenter] for i in 1:N]
        labels = Int[indmin(dist[i]) for i in 1:N]
        centroids = T[ sum(points[j,labels.==i])/sum(labels.==i) for j in 1:D, i in 1:numCenter]
    end 
    return centroids
end

@noinline function LinearRegression{T}(points::Matrix{T}, responses::Vector{T})
    # TODO: actual sequential implementation
    # return random result to enable type inference
    return rand(T, size(points,1))
end

@noinline function LinearRegression{T}(points::Matrix{T}, responses::Vector{T})
    # TODO: write actual sequential implementation
    # return random result to enable type inference
    return rand(T, size(points,1),2)
end

@noinline function NaiveBayes{T}(points::Matrix{T}, responses::Vector{T})
    # TODO: write actual sequential implementation
    # return random result to enable type inference
    return rand(T, size(points,1),2)
end

end
