using HPAT

function get_view_items(A::Vector{Tuple{Int64,Int64,Int64}}, days_in_sec_before_purchase, views_before_purchase, purchased_item)
    sort!(A, by=a->a[1], rev=true)
    out = Int64[]
    sale = false
    sale_time = 0
    count = 0
    for (t,i,s) in A
        # new sale happened, reset
        if s!=0
            count = 0
            if i==purchased_item
                sale = true
                sale_time = t
            else
                sale = false
            end
        # just a click by user
        else
            if sale==true && count<views_before_purchase && t>=sale_time-days_in_sec_before_purchase
                push!(out, i)
                count += 1
            end
        end
    end
    return out
end


@acc hpat function q03(days_in_sec_before_purchase, views_before_purchase, purchased_item, my_categories, file_name)

    web_clickstreams = DataSource(DataTable{:wcs_user_sk=Int64,:wcs_sales_sk=Int64,:wcs_item_sk=Int64,wcs_click_time_sk=Int64,wcs_click_date_sk=Int64}, HDF5, file_name)
    web_clickstreams[:tstamp_inSec] = web_clickstreams[:wcs_click_date_sk]*24*60*60 .+ web_clickstreams[:wcs_click_time_sk]
    item = DataSource(DataTable{:item_sk=Int64,:i_category_id=Int64}, HDF5, file_name)
    
    user_clicks = aggregate(web_clickstreams, :wcs_user_sk, collect(:tstamp_inSec),collect(:wcs_item_sk),collect(:wcs_sales_sk))
    view_items = map(a->get_view_items(a, days_in_sec_before_purchase, views_before_purchase, purchased_item), user_clicks)

    view_items = flatten(view_items)
    view_items_category = join(view_items, item, :view_items==:i_item_sk, :item_sk)
    view_items_category = view_items_category[:i_category_id in my_categories]
    item_counts = collect(countmap(view_items_category[:item_sk]))
    sort!(item_counts, rev=true)
    return item_counts[1:100]
    
end

println(q02(10*24*60*60, 5, 11, [3,4,5], "data.hdf5"))
