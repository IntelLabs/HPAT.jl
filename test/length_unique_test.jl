using HPAT

HPAT.CaptureAPI.set_debug_level(3)
HPAT.DomainPass.set_debug_level(3)
HPAT.DataTablePass.set_debug_level(3)

@acc hpat function q25(d_date, file_name)
    store_sales = DataSource(DataTable{:ss_customer_sk=Int64, :ss_ticket_number=Int64, :ss_sold_date_sk=Int64, :ss_net_paid=Float64}, HDF5, file_name)

    store_agg = aggregate(store_sales, :cid = :ss_customer_sk,
                                           :frequency = length(unique(:ss_ticket_number)),
                                           :most_recent_date = maximum(:ss_sold_date_sk),
                                           :amount = sum(:ss_net_paid))

    return store_agg[:cid], store_agg[:frequency], store_agg[:most_recent_date], store_agg[:amount]
end

println(q25("33000", "test_q25.hdf5"))
