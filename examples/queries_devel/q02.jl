using HPAT

function session_split(A::Vector{Tuple{Int64,Int64}}, timeout)
    out = Vector{Int64}[]
    if length(A)==0 return out end

    sort!(A)
    start_time = A[1][1]

    session = Int64[]
    for (a,b) in A
        if b-start_time>timeout
            push!(out, session)
            start_time = b
            session = Int64[]
        end
        push!(session,a)
    end
    push!(out, session)
    return out
end

function removeItem(S::Vector{Vector{Int64}}, a)
    out = Vector{Int64}[]
    for arr in S
        while findfirst(arr, a)!=0
            deletat!(arr, findfirst(arr, a))
        end
        push!(out, arr)
    end
    return out
end


@acc hpat function q02(my_item, timeout, num_top_items, file_name)

    web_clickstreams = DataSource(DataTable{:wcs_user_sk=Int64,:wcs_item_sk=Int64,wcs_click_time_sk=Int64,wcs_click_date_sk=Int64}, HDF5, file_name)
    web_clickstreams[:tstamp_inSec] = web_clickstreams[:wcs_click_date_sk]*24*60*60 .+ web_clickstreams[:wcs_click_time_sk]
    
    user_clicks = aggregate(web_clickstreams, :wcs_user_sk, collect(:tstamp_inSec),collect(:wcs_item_sk))
    map!(sort!, user_clicks)
    session_clicks = map(x->session_split(x,timeout), user_clicks)
    map!(session-> filter!(x->(my_item in x), session), session_clicks)
    map!(x->removeItem(my_item, x), session_clicks)
   
    all_items = flatten(session_clicks)
    item_counts = collect(countmap(all_items))
    sort!(item_counts, by=a->a[2])
    return item_counts[1:num_top_items]
end

println(q02(11, 60*60, 30, "data.hdf5"))
