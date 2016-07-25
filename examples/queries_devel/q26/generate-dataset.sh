#!/usr/bin/env bash
set -e
set -x

BIG_DATA_BENCHMARK_PATH=/home/whassan/Big-Data-Benchmark-for-Big-Bench/
DATASET_PATH=/home.old/whassan/tmp/csv/
NUM_MAP_TASKS=2

for dataset_factor in 600 ; do
    if [ -f ${DATASET_PATH}/store_sales_${dataset_factor}f.dat ]; then
        rm ${DATASET_PATH}/store_sales_${dataset_factor}f.dat
    fi
    if [ -f ${DATASET_PATH}/item_${dataset_factor}f.dat ]; then
        rm ${DATASET_PATH}/item_${dataset_factor}f.dat
    fi

    ${BIG_DATA_BENCHMARK_PATH}/bin/bigBench dataGen -U -m ${NUM_MAP_TASKS} -f $dataset_factor

    for i in $(seq ${NUM_MAP_TASKS}); do
        hadoop fs -cat /user/whassan/benchmarks/bigbench/data/store_sales/store_sales_${i}.dat > ${DATASET_PATH}/store_sales_${i}_${dataset_factor}f.dat

        hadoop fs -cat /user/whassan/benchmarks/bigbench/data/item/item_${i}.dat > ${DATASET_PATH}/item_${i}_${dataset_factor}f.dat
        cat ${DATASET_PATH}/store_sales_${i}_${dataset_factor}f.dat | cut -d'|' -f3,4 > ${DATASET_PATH}/store_sales_sanitized_${i}_${dataset_factor}f.csv &&  sed -i 's/|/,/g' ${DATASET_PATH}/store_sales_sanitized_${i}_${dataset_factor}f.csv

         cat ${DATASET_PATH}/item_${i}_${dataset_factor}f.dat | cut -d'|' -f1,10,13 |  sed 's/[.!@#$%^&*()-]//g' | sed -e 's,_,,g' | tr abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 012345678901234567890123456789  | cut -d ' ' -f1  > ${DATASET_PATH}/item_sanitized_${i}_${dataset_factor}f.csv && sed -i 's/|/,/g' ${DATASET_PATH}/item_sanitized_${i}_${dataset_factor}f.csv
    done
    cat ${DATASET_PATH}/store_sales_sanitized_[0-9]_${dataset_factor}f.csv > ${DATASET_PATH}/store_sales_sanitized_${dataset_factor}f.csv
    cat ${DATASET_PATH}/item_sanitized_[0-9]_${dataset_factor}f.csv > ${DATASET_PATH}/item_sanitized_${dataset_factor}f.csv

    hadoop fs -rmr /user/whassan/benchmarks/bigbench/data
    rm ${DATASET_PATH}/store_sales_[0-9]_${dataset_factor}f.dat
    rm ${DATASET_PATH}/item_[0-9]_${dataset_factor}f.dat
done