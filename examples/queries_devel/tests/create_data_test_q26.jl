using HDF5



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


file_name = "test_q26.hdf5"

h5write(file_name ,"/ss_customer_sk",customer)
h5write(file_name, "/ss_item_sk",sale)

h5write(file_name, "/i_item_sk", item)
h5write(file_name, "/i_class_id", class)
h5write(file_name, "/i_category", category)


