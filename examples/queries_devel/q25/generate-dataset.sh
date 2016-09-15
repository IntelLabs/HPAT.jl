#!/usr/bin/env bash
#set -e
set -x

BIG_DATA_BENCHMARK_PATH=/home/whassan/Big-Data-Benchmark-for-Big-Bench/
DATASET_PATH=/home.old/whassan/tmp/csv/q25/
mkdir -p ${DATASET_PATH}
NUM_MAP_TASKS=2

for dataset_factor in 20; do
    if [ -f ${DATASET_PATH}/store_sales_${dataset_factor}f.dat ]; then
        rm ${DATASET_PATH}/store_sales_${dataset_factor}f.dat
    fi
    if [ -f ${DATASET_PATH}/web_sales_${dataset_factor}f.dat ]; then
        rm ${DATASET_PATH}/web_sales_${dataset_factor}f.dat
    fi

    ${BIG_DATA_BENCHMARK_PATH}/bin/bigBench dataGen -U -m ${NUM_MAP_TASKS} -f $dataset_factor

    for i in $(seq ${NUM_MAP_TASKS}); do
	
    hadoop dfsadmin -safemode leave
	hadoop fs -cat /home.old/whassan/user/whassan/benchmarks/bigbench/data/store_sales/store_sales_${i}.dat > ${DATASET_PATH}/store_sales_${i}_${dataset_factor}f.dat
	hadoop fs -rmr /home.old/whassan/user/whassan/benchmarks/bigbench/data/store_sales/store_sales_${i}.dat
    hadoop fs -cat /home.old/whassan/user/whassan/benchmarks/bigbench/data/web_sales/web_sales_${i}.dat > ${DATASET_PATH}/web_sales_${i}_${dataset_factor}f.dat
    hadoop fs -rmr /home.old/whassan/user/whassan/benchmarks/bigbench/data/web_sales/web_sales_${i}.dat
	
    # schema in data-generator/config/bigbench-schema.xml
    # 1: ss_sold_date_sk, 4: ss_customer_sk, 10: ss_ticket_number, 21: ss_net_paid
    cat ${DATASET_PATH}/store_sales_${i}_${dataset_factor}f.dat  | cut -d'|' -f1,4,10,21 > ${DATASET_PATH}/store_sales_sanitized_${i}_${dataset_factor}f.dat
	rm ${DATASET_PATH}/store_sales_${i}_${dataset_factor}f.dat
	sed -i 's/|/,/g' ${DATASET_PATH}/store_sales_sanitized_${i}_${dataset_factor}f.dat

    # 1: ws_sold_date_sk, 5: ws_bill_customer_sk, 18: ws_order_number, 30: ws_net_paid
    cat ${DATASET_PATH}/web_sales_${i}_${dataset_factor}f.dat  | cut -d'|' -f1,5,18,30 > ${DATASET_PATH}/web_sales_sanitized_${i}_${dataset_factor}f.dat
    rm ${DATASET_PATH}/web_sales_${i}_${dataset_factor}f.dat
    sed -i 's/|/,/g' ${DATASET_PATH}/web_sales_sanitized_${i}_${dataset_factor}f.dat

    done

    cat ${DATASET_PATH}/store_sales_sanitized_[0-9]_${dataset_factor}f.dat > ${DATASET_PATH}/store_sales_sanitized_${dataset_factor}f.csv
    cat ${DATASET_PATH}/web_sales_sanitized_[0-9]_${dataset_factor}f.dat > ${DATASET_PATH}/web_sales_sanitized_${dataset_factor}f.csv
    hadoop dfsadmin -safemode leave
    hadoop fs -rmr /home.old/whassan/user/whassan/benchmarks/bigbench/data/

    rm ${DATASET_PATH}/store_sales_sanitized_[0-9]_${dataset_factor}f.dat
    rm ${DATASET_PATH}/web_sales_sanitized_[0-9]_${dataset_factor}f.dat

done
