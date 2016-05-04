using HPAT

@acc hpat function q26(category, item_count, num_centroids, file_name)
    store_sales = DataSource(DataTable{:ss_item_sk=Int64,:ss_customer_sk=Int64}, HDF5, file_name)
    item = DataSource(DataTable{:item_sk=Int64,:i_category=ASCIIString,:i_class_id=Int64}, HDF5, file_name)

    sale_items = join(store_sales, item, :ss_item_sk==:i_item_sk && :i_category==category)

    customer_i_class = aggregate(sale_items, :ss_customer_sk, :ss_item_count = size(:ss_item_sk),
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

    model = HPAT.Kmeans(customer_i_class, num_centroids)
    return model
end


println(q26("food", 20, 10, "data.hdf5"))
