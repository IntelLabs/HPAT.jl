#!/usr/bin/env bash
#set -e
set -x

BIG_DATA_BENCHMARK_PATH=/home/etotoni/Downloads/Big-Data-Benchmark-for-Big-Bench/
DATASET_PATH=/srv/data/tmp/q05/
mkdir -p ${DATASET_PATH}
NUM_MAP_TASKS=2

for dataset_factor in 300 400 500; do
    if [ -f ${DATASET_PATH}/web_clickstreams_${dataset_factor}f.dat ]; then
        rm ${DATASET_PATH}/web_clickstreams_${dataset_factor}f.dat
    fi
    if [ -f ${DATASET_PATH}/item_${dataset_factor}f.dat ]; then
        rm ${DATASET_PATH}/item_${dataset_factor}f.dat
    fi
    if [ -f ${DATASET_PATH}/customer_${dataset_factor}f.dat ]; then
        rm ${DATASET_PATH}/customer_${dataset_factor}f.dat
    fi
    if [ -f ${DATASET_PATH}/customer_demographics_${dataset_factor}f.dat ]; then
        rm ${DATASET_PATH}/customer_demographics_${dataset_factor}f.dat
    fi

    ${BIG_DATA_BENCHMARK_PATH}/bin/bigBench dataGen -U -m ${NUM_MAP_TASKS} -f $dataset_factor

    for i in $(seq ${NUM_MAP_TASKS}); do
	hadoop fs -cat /user/etotoni/benchmarks/bigbench/data/web_clickstreams/web_clickstreams_${i}.dat > ${DATASET_PATH}/web_clickstreams_${i}_${dataset_factor}f.dat
	hadoop dfsadmin -safemode leave
	hadoop fs -rmr /user/etotoni/benchmarks/bigbench/data/web_clickstreams/web_clickstreams_${i}.dat
	hadoop fs -cat /user/etotoni/benchmarks/bigbench/data/item/item_${i}.dat > ${DATASET_PATH}/item_${i}_${dataset_factor}f.dat
	hadoop fs -rmr /user/etotoni/benchmarks/bigbench/data/item/item_${i}.dat
	hadoop fs -cat /user/etotoni/benchmarks/bigbench/data/customer/customer_${i}.dat > ${DATASET_PATH}/customer_${i}_${dataset_factor}f.dat
	hadoop fs -rmr /user/etotoni/benchmarks/bigbench/data/customer/customer_${i}.dat
	# For some reasons second data is not generated
	#if [ $i -lt 2 ]; then
            hadoop fs -cat /user/etotoni/benchmarks/bigbench/data/customer_demographics/customer_demographics_${i}.dat > ${DATASET_PATH}/customer_demographics_${i}_${dataset_factor}f.dat
	    hadoop fs -rmr /user/etotoni/benchmarks/bigbench/data/customer_demographics/customer_demographics_${i}.dat
	#fi
	cat ${DATASET_PATH}/web_clickstreams_${i}_${dataset_factor}f.dat  | cut -d'|' -f4,6 > ${DATASET_PATH}/web_clickstreams_sanitized_${i}_${dataset_factor}f.dat
	rm ${DATASET_PATH}/web_clickstreams_${i}_${dataset_factor}f.dat
	sed -i 's/|/,/g' ${DATASET_PATH}/web_clickstreams_sanitized_${i}_${dataset_factor}f.dat

	cat ${DATASET_PATH}/item_${i}_${dataset_factor}f.dat  | cut -d'|' -f1,12,13 |  sed 's/[.!@#$%^&*()-]//g' | sed -e 's,_,,g' | tr abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 012345678901234567890123456789  | cut -d ' ' -f1 > ${DATASET_PATH}/item_sanitized_${i}_${dataset_factor}f.dat
	rm ${DATASET_PATH}/item_${i}_${dataset_factor}f.dat
	sed -i 's/|/,/g' ${DATASET_PATH}/item_sanitized_${i}_${dataset_factor}f.dat


	cat ${DATASET_PATH}/customer_${i}_${dataset_factor}f.dat | cut -d'|' -f1,3 >  ${DATASET_PATH}/customer_sanitized_${i}_${dataset_factor}f.dat
	rm ${DATASET_PATH}/customer_${i}_${dataset_factor}f.dat
	sed -i 's/|/,/g' ${DATASET_PATH}/customer_sanitized_${i}_${dataset_factor}f.dat

	#if [ $i -lt 2 ]; then
	    cat ${DATASET_PATH}/customer_demographics_${i}_${dataset_factor}f.dat  | cut -d'|' -f1,2,4 | sed 's/[.!@#$%^&*()-]//g' | sed -e 's,_,,g' | tr abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 012345678901234567890123456789012345678  | cut -d ' ' -f1 > ${DATASET_PATH}/customer_demographics_sanitized_${i}_${dataset_factor}f.dat
	    rm ${DATASET_PATH}/customer_demographics_${i}_${dataset_factor}f.dat
	    sed -i 's/|/,/g' ${DATASET_PATH}/customer_demographics_sanitized_${i}_${dataset_factor}f.dat
	#fi
     done

    cat ${DATASET_PATH}/web_clickstreams_sanitized_[0-9]_${dataset_factor}f.dat > ${DATASET_PATH}/web_clickstreams_sanitized_${dataset_factor}f.csv
    cat ${DATASET_PATH}/item_sanitized_[0-9]_${dataset_factor}f.dat > ${DATASET_PATH}/item_sanitized_${dataset_factor}f.csv
    cat ${DATASET_PATH}/customer_sanitized_[0-9]_${dataset_factor}f.dat > ${DATASET_PATH}/customer_sanitized_${dataset_factor}f.csv
    cat ${DATASET_PATH}/customer_demographics_sanitized_[0-9]_${dataset_factor}f.dat > ${DATASET_PATH}/customer_demographics_sanitized_${dataset_factor}f.csv
    hadoop dfsadmin -safemode leave
    hadoop fs -rmr /user/etotoni/benchmarks/bigbench/data/

    rm ${DATASET_PATH}/web_clickstreams_sanitized_[0-9]_${dataset_factor}f.dat
    rm ${DATASET_PATH}/item_sanitized_[0-9]_${dataset_factor}f.dat
    rm ${DATASET_PATH}/customer_sanitized_[0-9]_${dataset_factor}f.dat
    rm ${DATASET_PATH}/customer_demographics_sanitized_[0-9]_${dataset_factor}f.dat

done
