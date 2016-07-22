#!/usr/bin/env bash
set -e

BIG_DATA_BENCHMARK_PATH=/home/whassan/Big-Data-Benchmark-for-Big-Bench/
DATASET_PATH=/home/whassan/tmp/csv/

for dataset_factor in 400 , 500; do
    if [ -f ${DATASET_PATH}/store_sales_${dataset_factor}f.dat ]; then
	rm ${DATASET_PATH}/store_sales_${dataset_factor}f.dat
    fi
    if [ -f ${DATASET_PATH}/item_${dataset_factor}f.dat ]; then
	rm ${DATASET_PATH}/item_${dataset_factor}f.dat
    fi

    ${BIG_DATA_BENCHMARK_PATH}/bin/bigBench dataGen -U -m 1 -f $dataset_factor
    hadoop fs -copyToLocal /user/whassan/benchmarks/bigbench/data/store_sales/store_sales.dat ${DATASET_PATH}/store_sales_${dataset_factor}f.dat
    hadoop fs -copyToLocal /user/whassan/benchmarks/bigbench/data/item/item.dat ${DATASET_PATH}/item_${dataset_factor}f.dat
    hadoop fs -rmr /user/whassan/benchmarks/bigbench/data

    sed -i 's/|/,/g' ${DATASET_PATH}/store_sales_${dataset_factor}f.dat && cat ${DATASET_PATH}/store_sales_${dataset_factor}f.dat | cut -d, -f3,4 > ${DATASET_PATH}/store_sales_sanitized_${dataset_factor}f.csv

    sed -i 's/|/,/g' ${DATASET_PATH}/item_5f.dat && cat ${DATASET_PATH}/item_${dataset_factor}f.dat | cut -d, -f1,10,13 |  sed 's/[.!@#$%^&*()-]//g' | sed -e 's,_,,g' | tr abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 012345678901234567890123456789  | cut -d ' ' -f1  > ${DATASET_PATH}/item_sanitized_${dataset_factor}f.csv

    rm ${DATASET_PATH}/store_sales_${dataset_factor}f.dat
    rm ${DATASET_PATH}/item_${dataset_factor}f.dat
done