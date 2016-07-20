
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import com.databricks.spark.csv._
import scala.language.existentials

import org.apache.spark.sql.catalyst.analysis.UnresolvedRelation
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.execution.joins._
/**
  *  q26
command to run
./bin/spark-submit --num-executors 4  --jars /home/whassan/Downloads/commons-csv-1.2.jar,/home/whassan/Downloads/spark-csv_2.10-1.4.0.jar  --class Query26  ~/hps/query-examples/target/scala-2.10/query26_2.10-0.1.jar
  */
object Query26 {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Query")
    val sc = new SparkContext(sparkConf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val schema_store_sales = StructType(Array(
      StructField("ss_sold_date_sk", IntegerType,true),
      StructField("ss_sold_time_sk",IntegerType,true),
      StructField("ss_item_sk",IntegerType,true),
      StructField("ss_customer_sk",IntegerType,true),
      StructField("ss_cdemo_sk",IntegerType,true),
      StructField("ss_hdemo_sk",IntegerType,true),
      StructField("ss_addr_sk",IntegerType,true),
      StructField("ss_store_sk",IntegerType,true),
      StructField("ss_promo_sk",IntegerType,true),
      StructField("ss_ticket_number",IntegerType,true),
      StructField("ss_quantity",IntegerType,true),
      StructField("ss_wholesale_cost",DoubleType,true),
      StructField("ss_list_price",DoubleType,true),
      StructField("ss_sales_price",DoubleType,true),
      StructField("ss_ext_discount_amt",DoubleType,true),
      StructField("ss_ext_sales_price",DoubleType,true),
      StructField("ss_ext_wholesale_cost",DoubleType,true),
      StructField("ss_ext_list_price",DoubleType,true),
      StructField("ss_ext_tax",DoubleType,true),
      StructField("ss_coupon_amt",DoubleType,true),
      StructField("ss_net_paid",DoubleType,true),
      StructField("ss_net_paid_inc_tax",DoubleType,true),
      StructField("ss_net_profit",DoubleType,true)))
    val df_store_sales_temp = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(schema_store_sales).load("/home/whassan/tmp/csv/store_sales.csv")
    val df_store_sales = broadcast(df_store_sales_temp)
    //df_store_sales.show()
    val schema_item = StructType(Array(
      StructField("i_item_sk",IntegerType,true),
      StructField("i_item_id",StringType,true),
      StructField("i_rec_start_date",StringType,true),
      StructField("i_rec_end_date",StringType,true),
      StructField("i_item_desc",StringType,true),
      StructField("i_current_price",DoubleType,true),
      StructField("i_wholesale_cost",DoubleType,true),
      StructField("i_brand_id",IntegerType,true),
      StructField("i_brand",StringType,true),
      StructField("i_class_id",IntegerType,true),
      StructField("i_class",StringType,true),
      StructField("i_category_id",IntegerType,true),
      StructField("i_category",StringType,true),
      StructField("i_manufact_id",IntegerType,true),
      StructField("i_manufact",StringType,true),
      StructField("i_size",StringType,true),
      StructField("i_formulation",StringType,true),
      StructField("i_color",StringType,true),
      StructField("i_units",StringType,true),
      StructField("i_container",StringType,true),
      StructField("i_manager_id",IntegerType,true),
      StructField("i_product_name",StringType,true)))
    val df_item = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").schema(schema_item).load("/home/whassan/tmp/csv/item.csv")
    //df_item.show()
    // val store_sales = sqlContext.read.json("/home/whassan/tmp/store_sales.json")
    // val item = sqlContext.read.json("/home/whassan/tmp/item.json")
    df_store_sales.registerTempTable("store_sales_table")
    df_item.registerTempTable("item_table")


    // val sales_items = store_sales.join(item,item("item2") === store_sales("item1"))
    // val sales_items_fil = sales_items.filter(sales_items("category") === 1)
    // sales_items_fil.show()
    // val customer_i_class = sales_items_fil.groupBy("customer").agg( count("item1"))

    val lines = scala.io.Source.fromFile("/home/whassan/hps/query-examples/src/main/scala/q26.sql").mkString
    val fin  = sqlContext.sql(lines)
    fin.show()
    println(fin.queryExecution.logical.numberedTreeString)
    println("\n===================================\n")
    println(fin.queryExecution.optimizedPlan.numberedTreeString)
    println("\nExecuted Plan=====================\n")
    println(fin.queryExecution.executedPlan.numberedTreeString)
    println("\nSpark Plan=====================\n")
    println(fin.queryExecution.sparkPlan.numberedTreeString)
    println("\nStatistics=====================\n")
    println(fin.queryExecution.analyzed.statistics.sizeInBytes)
    println(df_item.queryExecution.analyzed.statistics.sizeInBytes)
    println(df_store_sales.queryExecution.analyzed.statistics.sizeInBytes)
    println(fin.queryExecution.toString)
    println("DONE with Query 26")
  }
}

// val sales_items = store_sales.join(item,item("item2") === store_sales("item1"))
//     val sales_items_fil = sales_items.filter(sales_items("category") === 1)
//     sales_items_fil.show()
//     val customer_i_class = sales_items_fil.groupBy("customer").agg( count("item1"))
//     customer_i_class.show()


//     //Import Sql Implicit conversions
//     import sqlContext.implicits._
//     import org.apache.spark.sql.Row
//     import org.apache.spark.sql.types.{StructField,StructField,StringType,DoubleType}

//     //Create Schema RDD
//     val schema_string = "cid,id1,id2,id3,id4,id5,id6,id7,id8,id9,id10,id11,id12,id13,id14,id15"
//     val schema_rdd = StructType(schema_string.split(",").map(fieldName => StructField(fieldName, DoubleType, true)) )

//     //Create Empty DataFrame
//     val empty_df = sqlContext.createDataFrame(sc.emptyRDD[Row], schema_rdd) 

//     //Some Operations on Empty Data Frame
//     empty_df.show()
//     println(empty_df.count())

//     //You can register a Table on Empty DataFrame, it's empty table though
//     empty_df.registerTempTable("empty_table")

//     //let's check it ;)
//     val res = sqlContext.sql("select count(CASE WHEN item_table.item1=1 THEN 1 ELSE NULL END) AS id1 from store_sales_table item_table")
//     res.show()
//     println(res.queryExecution.analyzed.numberedTreeString)
    //val dd = sqlContext.sql("CREATE TABLE ddd ( cid  double,  id1  double,  id2  double,  id3  double,  id4  double,  id5  double,  id6  double,  id7  double,  id8  double,  id9  double,  id10 double,  id11 double,  id12 double,  id13 double,  id14 double,  id15 double)").collect().foreach(println)
