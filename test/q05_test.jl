using HPAT
using HDF5
using Base.Test
using CompilerTools
using ParallelAccelerator

function create_dataset_small(dataset_name)
    wcs_item_sk = [2,3,5,7,9,11,1,1,9]
    wcs_user_sk = [3,3,3,3,8,2,2,3,3]

    i_item_sk = [1,2,2,2,2,6,7,8,9]
    i_category_id = [3,3,3,3,8,2,2,3,3]
    i_category = [2,3,4,4,8,8,7,7,7]

    c_customer_sk = [1,2,2,2,8,6,7,8,9]
    c_current_cdemo_sk = [3,3,3,3,8,2,2,3,3]

    cd_demo_sk = [1,12,2,2,2,3,3,8,9]
    cd_gender = [3,3,3,3,8,9,9,9,9]
    cd_education_status = [2,7,7,4,8,8,7,7,7]

    file_name = dataset_name
    h5write(file_name ,"/wcs_item_sk", wcs_item_sk)
    h5write(file_name ,"/wcs_user_sk", wcs_user_sk)

    h5write(file_name ,"/i_item_sk", i_item_sk)
    h5write(file_name ,"/i_category_id",i_category_id)
    h5write(file_name ,"/i_category", i_category)

    h5write(file_name ,"/c_customer_sk", c_customer_sk)
    h5write(file_name ,"/c_current_cdemo_sk", c_current_cdemo_sk)

    h5write(file_name ,"/cd_demo_sk", cd_demo_sk)
    h5write(file_name ,"/cd_gender", cd_gender)
    h5write(file_name ,"/cd_education_status", cd_education_status)
end

@acc hpat function q05(category, education, gender, file_name)
    web_clickstreams = DataSource(DataTable{:wcs_item_sk=Int64, :wcs_user_sk=Int64}, HDF5, file_name)
    item = DataSource(DataTable{:i_item_sk=Int64,:i_category_id=Int64,:i_category=Int64}, HDF5, file_name)
    customer = DataSource(DataTable{:c_customer_sk=Int64,:c_current_cdemo_sk=Int64}, HDF5, file_name)
    customer_demographics = DataSource(DataTable{:cd_demo_sk=Int64,:cd_gender=Int64,:cd_education_status=Int64}, HDF5, file_name)
    # Used for Not NULL
    web_clickstreams = web_clickstreams[:wcs_item_sk>typemin(Int32)]

    user_items = join(web_clickstreams, item, :wcs_item_sk==:i_item_sk, :user_items_sk)

    user_clicks_in_cat = aggregate(user_items, :wcs_user_sk, :clicks_in_category = sum(:i_category==category),
                                                         :clicks_in_1 = sum(:i_category_id==1),
                                                         :clicks_in_2 = sum(:i_category_id==2),
                                                         :clicks_in_3 = sum(:i_category_id==3),
                                                         :clicks_in_4 = sum(:i_category_id==4),
                                                         :clicks_in_5 = sum(:i_category_id==5),
                                                         :clicks_in_6 = sum(:i_category_id==6),
                                                         :clicks_in_7 = sum(:i_category_id==7))
    customer_clicks = join(user_clicks_in_cat, customer, :wcs_user_sk==:c_customer_sk, :customer_clicks_sk)
    customer_demo_clicks = join(customer_clicks, customer_demographics, :c_current_cdemo_sk==:cd_demo_sk , :customer_demo_clicks_sk)
    return customer_demo_clicks[:customer_clicks_sk], customer_demo_clicks[:customer_demo_clicks_sk] , customer_demo_clicks[:clicks_in_2], customer_demo_clicks[:clicks_in_3]
end

println("     Testing Query 05[small]")
create_dataset_small("test_q05.hdf5")
return_arr= q05(3, 7, 9, "test_q05.hdf5")

@test return_arr[1] == [2,2,2,8,2,2,2,8,8]
@test return_arr[2] == [3,3,3,3,3,3,3,3,8]
@test return_arr[3] == [0,0,0,0,0,0,0,0,0]
@test return_arr[4] == [1,1,1,1,1,1,1,1,1]

rm("test_q05.hdf5")
