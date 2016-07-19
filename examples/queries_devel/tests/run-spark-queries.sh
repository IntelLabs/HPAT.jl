#!/usr/bin/env bash

# Uncomment following line for debuggin
# set -x

# Set DIRs according to your environment
ROOT_DIR=/home/whassan/
SPARK_DIR=/home/whassan/spark/
SPARK_QUERY_DIR=/home/whassan/spark-sql-query-tests/


if [[ ! -d $SPARK_QUERY_DIR ]]; then
    cd $ROOT_DIR
    git clone https://github.com/Wajihulhassan/spark-sql-query-tests.git
fi

cd $SPARK_QUERY_DIR

if [[ "$HOSTNAME" == *PSEPHI* || "$HOSTNAME" == *psephi* ]]; then
    sbt -DproxySet=true -DproxyHost=proxy.jf.intel.com -DproxyPort=911 -v package
else
    sbt -v package
fi

if [[ ! -d $SPARK_DIR ]]; then
    echo "Download and install spark"
    exit
fi

cd  $SPARK_DIR

# TODO generalize this for every query
./bin/spark-submit --master spark://172.16.144.26:7080 --executor-memory 8G --driver-memory 8G  --jars /home/whassan/commons-csv-1.1.jar,/home/whassan/spark-csv_2.10-1.4.0.jar   --class Query26  ~/spark-sql-query-tests/target/scala-2.10/query26_2.10-0.1.jar &> tmp.txt

time_q26=`cat tmp.txt  | grep '\*\*\*\*\*\*' | cut -d ' ' -f 6`

rm tmp.txt

echo "TIME it took for Query 26: "$time_q26
