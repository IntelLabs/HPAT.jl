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

@noinline function data_sink_HDF5(y, var::AbstractString, file_name::AbstractString)
    h5write(file_name, var,y)
    return nothing
end

@noinline function data_source_TXT(T::DataType, file_name::AbstractString)
    arr::T = readdlm(file_name, ' ', eltype(T))
    return arr
end

@noinline function Kmeans{T}(points::Matrix{T}, numCenter::Int, iterNum::Int)
    # naive backup sequential implementation
    D = size(points,1) # number of features
    N = size(points,2) # number of instances
    centroids = rand(T, D, numCenter)
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

@noinline function join(t1::Vector{Vector},t2::Vector{Vector})
    t1_cols = length(t1)
    t2_cols = length(t2)
    # 1 key column for output
    out_size = t1_cols+t2_cols-1;
    out = [ [] for i in 1:out_size]
    # for each key in table 1
    for i in 1:length(t1[1])
        # for each key in table 2
        for j in 1:length(t2[1])
            if t1[1][i]==t2[1][j]
                # save key to output
                push!(out[1],t1[1][i])
                # save table 1 values
                for ii in 2:length(t1)
                    push!(out[ii], t1[ii][i])
                end
                # save table 2 values
                for jj in 2:length(t2)
                    push!(out[t1_cols+jj-1], t2[jj][j])
                end
            end
        end
    end
    return out
end

@noinline function aggregate{T}(t1k::Vector{T}, new_cols::Vector{ANY})
    new_cols_len = length(new_cols)
    # group values for each key into a separate array
    # key => [ [...] , [...] ...]
    groups = Dict{T,Vector{Vector}}()
    # for each key
    for i in 1:length(t1k)
        # initialize key if seen for first time
        if !haskey(groups,t1k[i])
            # one array for each output
            groups[t1k[i]] = Array(Vector,new_cols_len)
            for j in 1:new_cols_len
                groups[t1k[i]][j] = []
            end
        end
        # save group values
        for j in 1:new_cols_len
            push!(groups[t1k[i]][j], new_cols[j][1][i])
        end
    end
    #dprintln(groups)
    # output is array of column vectors
    out = Array(Vector,new_cols_len+1)
    # first array is keys
    out[1] = collect(keys(groups))
    for j in 1:new_cols_len
        out[j+1] = []
    end
    for j in 1:new_cols_len
        func = new_cols[j][2]
        #println(func)
        for key in out[1]
            push!(out[j+1], func(groups[key][j]))
        end
    end
    return out
end

@noinline function table_filter!(cond::BitArray{1}, columns::Vector{Vector})
    for i in 1:length(columns)
        columns[i] = columns[i][cond]
    end
end

#=
@doc """
function join{T1,T12,T22}(t1c1::Vector{T1}, t1c2::Vector{T12}, t2c1::Vector{T1}, t2c2::Vector{T22})
       out1 = t1c1 .+ t2c1
       out2 = t1c2
       return out1,out2
end

"""
# join up to 50 column tables
const MAX_COLUMNS = 50

for k in 1:MAX_COLUMNS
    for l in 1:MAX_COLUMNS
        # create type symbol names
        # T11 T12 ... T21 T22...
        typ_names1 = Symbol[Symbol("T1$j") for j in 1:k]
        typ_names2 = Symbol[Symbol("T2$j") for j in 1:l]
        typ_names = [typ_names1;typ_names2]
        # join{T1,T11,T12...T21,T22...}
        fexpr = Expr(:curly,:join,:T1,typ_names...)
        # t1c1::Vector{T11}, t1c2::Vector{T12} ....
        arrs1 = Expr[ Expr(:(::),Symbol("t1c$j"), Expr(:curly,:Vector,typ_names1[j])) for j in 1:k]
        arrs2 = Expr[ Expr(:(::),Symbol("t2c$j"), Expr(:curly,:Vector,typ_names2[j])) for j in 1:l]
        # key array: t1k::Vector{T1}
        arrk1 = Expr(:(::),Symbol("t1k"), Expr(:curly,:Vector,:T1))
        arrk2 = Expr(:(::),Symbol("t2k"), Expr(:curly,:Vector,:T1))
        arrs = [arrk1; arrs1;arrk2; arrs2]
        # fcall = Expr(:call, fexpr,arrs...)
        # out1=T1[], out2=T2[]
        out_decls1 = Expr[ Expr(:(=),Symbol("out1$j"), Expr(:ref,Symbol("T1$j"))) for j in 1:k]
        out_decls2 = Expr[ Expr(:(=),Symbol("out2$j"), Expr(:ref,Symbol("T2$j"))) for j in 1:l]
        outk_decl = Expr(:(=),:outk, Expr(:ref,:T1))
        out_decls = [outk_decl; out_decls1; out_decls2]
        out_block = Expr(:block, out_decls...)
        # list of output symbols: outk out11 out12... out21 out22...
        outs1 = Symbol[Symbol("out1$j") for j in 1:k]
        outs2 = Symbol[Symbol("out1$j") for j in 1:k]
        outs = [:outk;outs1;outs2]
        # push!(out11,t1c1[i])
        push_calls1 = [ Expr(:call,:push!,Symbol("out1$j"),Symbol("t1c$j[i]"))  for j in 1:k]
        # push!(out21,t2c1[j])
        push_calls2 = [ Expr(:call,:push!,Symbol("out2$j"),Symbol("t2c$j[j]"))  for j in 1:l]
        push_calls = Expr(:block, push_calls1..., push_calls2...)
        @eval begin
            @noinline function ($fexpr)($(arrs...))
                $out_block
                for i in 1:length(t1k)
                    for j in length(t2k)
                        if t1k[i]==t2k[j]
                            push!(outk,t1k[i])
                            $push_calls
                        end
                    end
                end
                return $(Expr(:tuple,outs...))
            end
        end

    end # for k
end # for l
=#

end # module
