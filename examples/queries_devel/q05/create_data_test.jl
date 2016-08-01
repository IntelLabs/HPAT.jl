using HDF5
using DataFrames
# Arguments:
# ARGS[1] = option
# ARGS[2] = table web_clickstreams path e.g."/home/whassan/tmp/csv/store_sales_sanitized.csv"
# ARGS[3] = table item path e.g. "/home/whassan/tmp/csv/item_sanitized.csv"
# ARGS[4] = table customer path e.g. "/home/whassan/tmp/csv/item_sanitized.csv"
# ARGS[5] = table customer_demographics path e.g. "/home/whassan/tmp/csv/item_sanitized.csv"


function example_dataset(wcs_user_sk, wcs_item_sk, i_item_sk, i_category_id, i_category, c_customer_sk, c_current_cdemo_sk, cd_demo_sk, cd_gender, cd_education_status)
    append!(wcs_user_sk, [1,1,2,1,2,3])
    append!(wcs_item_sk, [1,1,2,1,2,3])
    append!(i_item_sk, [1,1,2,1,2,3])
    append!(i_category_id, [1,1,2,1,2,3])
    append!(i_category, [1,1,2,1,2,3])
    append!(c_customer_sk, [1,1,2,1,2,3])
    append!(c_current_cdemo_sk, [1,1,2,1,2,3])
    append!(cd_demo_sk, [1,1,2,1,2,3])
    append!(cd_gender, [1,1,2,1,2,3])
    append!(cd_education_status, [1,1,2,1,2,3])
end

function generate_dataset(wcs_user_sk, wcs_item_sk, i_item_sk, i_category_id, i_category, c_customer_sk, c_current_cdemo_sk, cd_demo_sk, cd_gender, cd_education_status, table_wcs_path, table_i_path, table_c_path, table_cd_path)

    wcs_df = readtable(open(table_wcs_path))
    i_df = readtable(open(table_i_path))
    c_df = readtable(open(table_c_path))
    cd_df = readtable(open(table_cd_path))

    append!(wcs_user_sk,wcs_user_df[1])
    append!(wcs_item_sk,wcs_user_df[2])

    append!(i_item_sk,item_df[1])
    append!(i_category_id,item_df[2])
    append!(i_category,item_df[3])

    append!(c_customer_sk,item_df[1])
    append!(c_current_cdemo_sk,item_df[2])

    append!(cd_demo_sk,item_df[1])
    append!(cd_gender,item_df[2])
    append!(cd_education_status,item_df[3])
end

function main()

    option = parse(Int,ARGS[1])
    if  option == 1 && length(ARGS) >= 5
        table_wcs_path=ARGS[2]
        table_i_path=ARGS[3]
        table_c_path=ARGS[4]
        table_cd_path=ARGS[5]
    end
    wcs_user_sk = Int64[]
    wcs_item_sk = Int64[]
    i_item_sk = Int64[]
    i_category_id = Int64[]
    i_category = Int64[]
    c_customer_sk = Int64[]
    c_current_cdemo_sk = Int64[]
    cd_demo_sk = Int64[]
    cd_gender = Int64[]
    cd_education_status = Int64[]

    if option == 0
        example_dataset(wcs_user_sk, wcs_item_sk, i_item_sk, i_category_id, i_category, c_customer_sk, c_current_cdemo_sk, cd_demo_sk, cd_gender, cd_education_status)
    end

    if option == 1
        generate_dataset(wcs_user_sk, wcs_item_sk, i_item_sk, i_category_id, i_category, c_customer_sk, c_current_cdemo_sk, cd_demo_sk, cd_gender, cd_education_status)
    end
    println(":D Done reading into arrays whole dataset")
    file_name = "test_q05.hdf5"

    if isfile(file_name)
        rm(file_name)
    end
    h5write(file_name ,"/wcs_user_sk", wcs_user_sk)
    h5write(file_name ,"/wcs_item_sk", wcs_item_sk)
    h5write(file_name ,"/i_item_sk", i_item_sk)
    h5write(file_name ,"/i_category_id",i_category_id)
    h5write(file_name ,"/i_category", i_category)
    h5write(file_name ,"/c_customer_sk", c_customer_sk)
    h5write(file_name ,"/c_current_cdemo_sk", c_current_cdemo_sk)
    h5write(file_name ,"/cd_demo_sk", cd_demo_sk)
    h5write(file_name ,"/cd_gender", cd_gender)
    h5write(file_name ,"/cd_education_status", cd_education_status)
end
main()
