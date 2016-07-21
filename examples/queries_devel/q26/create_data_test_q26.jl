using HDF5
using DataFrames
# Arguments:
# ARGS[1] = option
# ARGS[2] = table store_sales path e.g."/home/whassan/tmp/csv/store_sales_sanitized.csv"
# ARGS[3] = table item path e.g. "/home/whassan/tmp/csv/item_sanitized.csv"

#num_rows = 2000000000000

function example_dataset(customer,sale,item,class,category)
    #=
    customer    iterm
    1       1
    1       2
    2       1
    1       3
    2       3
    3       1
    =#
    customer_tmp = [1,1,2,1,2,3]
    append!(customer,customer_tmp)
    sale_tmp = [1,2,1,3,3,1]
    append!(sale,sale_tmp)

    #= item class category
    1   3   1
    2   1   2
    3   2   1
    =#
    item_tmp = [1,2,3]
    append!(item,item_tmp)
    class_tmp = [3,1,2]
    append!(class, class_tmp)
    category_tmp = [1,2,1]
    append!(category, category_tmp)
    
end

function generate_dataset(customer,sale,item,class,category,table1_path,table2_path)
    store_sales_file = open(table1_path)
    item_file = open(table2_path)
    counter = 0
    item_df = readtable(item_file)
    append!(item,item_df[1])
    append!(class,item_df[2])
    append!(category,item_df[3])

    ss_df = readtable(store_sales_file)
    append!(sale,ss_df[1])
    append!(customer,ss_df[2])
end

function main()

    option = parse(Int,ARGS[1])
    if  option == 1 && length(ARGS) >= 3
        table1_path=ARGS[2]
        table2_path=ARGS[3]
    end

    customer = Int64[]
    sale = Int64[]
    item = Int64[]
    class = Int64[]
    category = Int64[]

    if option == 0
        example_dataset(customer,sale,item,class,category)
    end

    if option == 1
        generate_dataset(customer,sale,item,class,category,table1_path,table2_path)
    end
    println(":D Done reading into arrays whole dataset")
    file_name = "test_q26.hdf5"

    if isfile(file_name)
        rm(file_name)
    end
    h5write(file_name ,"/ss_customer_sk",customer)
    h5write(file_name, "/ss_item_sk",sale)

    h5write(file_name, "/i_item_sk", item)
    h5write(file_name, "/i_class_id", class)
    h5write(file_name, "/i_category", category)
end
main()