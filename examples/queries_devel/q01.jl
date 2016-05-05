using HPAT

function makePairs(A::Vector{Int64})
    out = []
    for i in 1:length(A)-1
        for j in i+1:length(A)
            push!(out, (A[i],A[j]))
        end
    end
    return out
end

@acc hpat function q01(my_categories, my_stores, num_top_items, count_threshod, file_name)

    store_sales = DataSource(DataTable{:ss_item_sk=Int64,:ss_ticket_number=Int64,:ss_store_sk=Int64}, HDF5, file_name)
    item = DataSource(DataTable{:item_sk=Int64,:i_category_id=Int64}, HDF5, file_name)

    sale_items = join(store_sales, item, :ss_item_sk==:i_item_sk, :item_sk)

    sale_items = sale_items[:i_category_id in my_categories && :ss_store_sk in my_stores]

    # get array of items sold in a transaction
    sold_in_ticket = aggregate(sale_items, :ss_ticket_number, :item_arr = collect(:item_sk))

    # remove duplicates in each sale
    map!(union, sold_in_ticket[:item_arr])

    # sort items of sale
    map!(sort!, sold_in_ticket[:item_arr])

    # make pair of items sold
    map!(makePairs, sold_in_ticket[:item_arr])

    all_pairs = flatten(sold_in_ticket[:item_arr])
    pair_counts = countmap(all_pairs)

    high_counts = collect(filter((a,b)->b>count_threshod, pair_counts))
    sort!(high_counts)
    return high_counts[1:num_top_items]
end


println(q01([1,2,3], [20,30], 100, 50, "data.hdf5"))
