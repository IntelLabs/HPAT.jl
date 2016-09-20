using HPAT
#using HPAT.API.Kmeans
#HPAT.CaptureAPI.set_debug_level(3)
#HPAT.DomainPass.set_debug_level(3)
#HPAT.DataTablePass.set_debug_level(3)
#HPAT.DistributedPass.set_debug_level(3)
#ParallelAccelerator.DomainIR.set_debug_level(3)
#ParallelAccelerator.CGen.set_debug_level(3)


@acc hpat function q26(category, item_count, num_centroids, iterations, file_name)
    store_sales = DataSource(DataTable{:ss_item_sk=Int64,:ss_customer_sk=Int64}, HDF5, file_name)
    item = DataSource(DataTable{:i_item_sk=Int64,:i_category=Int64,:i_class_id=Int64}, HDF5, file_name)

    sale_items = join(store_sales, item, :ss_item_sk==:i_item_sk, :ss_item_sk)

    sale_items = sale_items[:i_category==category]

    customer_i_class = aggregate(sale_items, :ss_customer_sk, :ss_item_count = length(:ss_item_sk),
                                                               :id1 = sum(:i_class_id==1),
                                                               :id2 = sum(:i_class_id==2),
                                                               :id3 = sum(:i_class_id==3),
                                                               :id4 = sum(:i_class_id==4),
                                                               :id5 = sum(:i_class_id==5),
                                                               :id6 = sum(:i_class_id==6),
                                                               :id7 = sum(:i_class_id==7),
                                                               :id8 = sum(:i_class_id==8),
                                                               :id9 = sum(:i_class_id==9),
                                                               :id10 = sum(:i_class_id==10),
                                                               :id11 = sum(:i_class_id==11),
                                                               :id12 = sum(:i_class_id==12),
                                                               :id13 = sum(:i_class_id==13),
                                                               :id14 = sum(:i_class_id==14),
                                                               :id15 = sum(:i_class_id==15))

    customer_i_class = customer_i_class[:ss_item_count>item_count]

    points = transpose(typed_hcat(Float64,customer_i_class[:ss_item_count], customer_i_class[:id1], customer_i_class[:id2],
    customer_i_class[:id3], customer_i_class[:id4], customer_i_class[:id5], customer_i_class[:id6],
    customer_i_class[:id7], customer_i_class[:id8], customer_i_class[:id9], customer_i_class[:id10],
    customer_i_class[:id11], customer_i_class[:id12], customer_i_class[:id13], customer_i_class[:id14], customer_i_class[:id15]))

    model = Kmeans(points, num_centroids, iterations)
    return model
end


#println(q26(1, 1, 8, 20, "test_q26.hdf5"))
println(q26(90882, 1, 8, 20, "test_q26.hdf5"))
