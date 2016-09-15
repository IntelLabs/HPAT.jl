using HDF5
using DataFrames
# Arguments:
# ARGS[1] = option
# ARGS[2] = table store_sales path e.g."/home/whassan/tmp/csv/q25/store_sales_sanitized.csv"
# ARGS[5] = table web_sales path e.g. "/home/whassan/tmp/csv/q25/web_sales_sanitized.csv"


function example_dataset(ss_customer_sk, ss_ticket_number, ss_sold_date_sk, ss_net_paid,
       ws_bill_customer_sk, ws_order_number, ws_sold_date_sk, ws_net_paid)
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

function generate_dataset(ss_customer_sk, ss_ticket_number, ss_sold_date_sk, ss_net_paid, 
        ws_bill_customer_sk, ws_order_number, ws_sold_date_sk, ws_net_paid, 
        table_ss_path, table_ws_path)

    wcs_df = readtable(open(table_wcs_path))
    i_df = readtable(open(table_i_path))
    append!(wcs_item_sk, convert(Array, wcs_df[1], typemax(Int32)))
    #replace NA values in Dataframes with 2147483648
    append!(wcs_user_sk, convert(Array, wcs_df[2], typemax(Int32)))

    append!(i_item_sk, i_df[1])
    append!(i_category_id, i_df[2])
    append!(i_category, i_df[3])

    append!(c_customer_sk, c_df[1])
    #replace NA values in Dataframes with 2147483648
    append!(c_current_cdemo_sk, convert(Array, c_df[2], typemax(Int32)))

    append!(cd_demo_sk, cd_df[1])
    #replace NA values in Dataframes with 2147483648
    append!(cd_gender, convert(Array, cd_df[2],  typemax(Int32)))
    append!(cd_education_status, convert(Array, cd_df[3], typemax(Int32)))
end

function main()

    option = parse(Int,ARGS[1])
    if  option == 1 && length(ARGS) >= 3
        table_ss_path=ARGS[2]
        table_ws_path=ARGS[3]
    end

    ss_customer_sk = Int64[]
    ss_ticket_number = Int64[]
    ss_sold_date_sk = Int64[]
    ss_net_paid = Float64[]

    ws_bill_customer_sk = Int64[]
    ws_order_number = Int64[]
    ws_sold_date_sk = Int64[]
    ws_net_paid = Float64[]

    if option == 0
        example_dataset(ss_customer_sk, ss_ticket_number, ss_sold_date_sk, ss_net_paid, ws_bill_customer_sk, ws_order_number, ws_sold_date_sk, ws_net_paid)
    end

    if option == 1
        generate_dataset(ss_customer_sk, ss_ticket_number, ss_sold_date_sk, ss_net_paid, ws_bill_customer_sk, ws_order_number, ws_sold_date_sk, ws_net_paid, table_ss_path, table_ws_path)
    end
    println(":D Done reading into arrays whole dataset")
    file_name = "test_q25.hdf5"

    if isfile(file_name)
        rm(file_name)
    end
    h5write(file_name ,"/ss_customer_sk", ss_customer_sk)
    h5write(file_name ,"/ss_ticket_number", ss_ticket_number)
    h5write(file_name ,"/ss_sold_date_sk", ss_sold_date_sk)
    h5write(file_name ,"/ss_net_paid", ss_net_paid)
    h5write(file_name ,"/ws_bill_customer_sk", ws_bill_customer_sk)
    h5write(file_name ,"/ws_order_number", ws_order_number)
    h5write(file_name ,"/ws_sold_date_sk", ws_sold_date_sk)
    h5write(file_name ,"/ws_net_paid", ws_net_paid)
end
main()
