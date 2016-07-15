using HPAT
using HDF5
using Base.Test
using CompilerTools
using ParallelAccelerator

function create_dataset(dataset_name)
    #=
    customer    iterm
    1       1
    1       2
    2       1
    1       3
    2       3
    3       1
    =#
    customer = [1,1,2,1,2,3]
    sale = [1,2,1,3,3,1]

    #= item class category
    1   3   1
    2   1   2
    3   2   1
    =#
    item = [1,2,3]
    class = [3,1,2]
    category = [1,2,1]
    file_name = dataset_name

    h5write(file_name ,"/ss_customer_sk",customer)
    h5write(file_name, "/ss_item_sk",sale)

    h5write(file_name, "/i_item_sk", item)
    h5write(file_name, "/i_class_id", class)
    h5write(file_name, "/i_category", category)

end
                          
@acc hpat function q26(category, item_count, num_centroids, file_name)
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
    return customer_i_class[:ss_item_count],customer_i_class[:id3]
end
println("     Testing Query 26")
create_dataset("test_q26.hdf5")
return_arr= q26(1, 1, 10, "test_q26.hdf5")
@test return_arr[1] == [2,2]
@test return_arr[2] == [1,1]
rm("test_q26.hdf5")
