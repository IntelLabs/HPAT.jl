using HPAT

#HPAT.CaptureAPI.set_debug_level(3)
#HPAT.DomainPass.set_debug_level(3)
#HPAT.DataTablePass.set_debug_level(3)

@acc hpat function q25(d_date, file_name)
    store_sales = DataSource(DataTable{:ss_customer_sk=Int64, :ss_ticket_number=Int64, :ss_sold_date_sk=Int64, :ss_net_paid=Float64}, HDF5, file_name)

    store_agg = aggregate(store_sales, :cid = :ss_customer_sk,
                                           :frequency = length(unique(:ss_ticket_number)),
                                           :most_recent_date = maximum(:ss_sold_date_sk),
                                           :amount = sum(:ss_net_paid))

    return store_agg[:cid], store_agg[:frequency], store_agg[:most_recent_date], store_agg[:amount]
end

using HDF5
using MPI
if MPI.Comm_rank(MPI.COMM_WORLD)==0 
    h5write("test_q25.hdf5", "/ss_customer_sk", [1,1,2,2,1])
    h5write("test_q25.hdf5", "/ss_ticket_number", [0,3,5,6,0])
    h5write("test_q25.hdf5", "/ss_sold_date_sk", [37580,34000,35000,36000,37600])
    h5write("test_q25.hdf5", "/ss_net_paid", [101.0,24.0,3.5,50.0,3.2])
end

c, f, m, a = q25("33000", "test_q25.hdf5")
println(c,f,m,a)

using Base.Test
@test c==[1,2]
@test f==[2,2]
@test m==[37600,36000]
@test_approx_eq a [128.2,53.5]
