SELECT
  ss.ss_customer_sk AS cid,
  count(CASE WHEN i.i_class_id=1  THEN 1 ELSE NULL END) AS id1,
  count(CASE WHEN i.i_class_id=2  THEN 1 ELSE NULL END) AS id2,
  count(CASE WHEN i.i_class_id=3  THEN 1 ELSE NULL END) AS id3,
  count(CASE WHEN i.i_class_id=4  THEN 1 ELSE NULL END) AS id4,
  count(CASE WHEN i.i_class_id=5  THEN 1 ELSE NULL END) AS id5,
  count(CASE WHEN i.i_class_id=6  THEN 1 ELSE NULL END) AS id6,
  count(CASE WHEN i.i_class_id=7  THEN 1 ELSE NULL END) AS id7,
  count(CASE WHEN i.i_class_id=8  THEN 1 ELSE NULL END) AS id8,
  count(CASE WHEN i.i_class_id=9  THEN 1 ELSE NULL END) AS id9,
  count(CASE WHEN i.i_class_id=10 THEN 1 ELSE NULL END) AS id10,
  count(CASE WHEN i.i_class_id=11 THEN 1 ELSE NULL END) AS id11,
  count(CASE WHEN i.i_class_id=12 THEN 1 ELSE NULL END) AS id12,
  count(CASE WHEN i.i_class_id=13 THEN 1 ELSE NULL END) AS id13,
  count(CASE WHEN i.i_class_id=14 THEN 1 ELSE NULL END) AS id14,
  count(CASE WHEN i.i_class_id=15 THEN 1 ELSE NULL END) AS id15
FROM store_sales_table ss
INNER JOIN item_table i 
  ON (ss.ss_item_sk = i.i_item_sk AND i.i_category IN ('Music')
  AND ss.ss_customer_sk IS NOT NULL
)
GROUP BY ss.ss_customer_sk
HAVING count(ss.ss_item_sk) > 10
ORDER BY cid
