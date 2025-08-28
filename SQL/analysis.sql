------------
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'green_tripdata';

------------
SELECT *
FROM green_tripdata
LIMIT 100;

------------
SELECT COUNT(*)
FROM green_tripdata;

------------
SELECT
    "lpep_pickup_datetime",      -- 接載時間
    "lpep_dropoff_datetime",     -- 下車時間
    "trip_distance",             -- 行程距離
    "fare_amount",               -- 車資
    "tip_amount",                -- 小費
    "total_amount"               -- 總金額
FROM
    green_tripdata
ORDER BY
    "total_amount" DESC          -- 依照總金額降序排序
LIMIT 10;                        -- 只顯示前 10 筆

------------
SELECT
    CAST("lpep_pickup_datetime" AS DATE) AS trip_date, -- 將時間戳轉換為日期
    COUNT(*) AS total_trips,                             -- 每日行程總數
    AVG("trip_distance") AS avg_trip_distance,           -- 每日平均行程距離
    AVG("total_amount") AS avg_total_amount              -- 每日平均總金額
FROM
    green_tripdata
GROUP BY
    trip_date                                            -- 依照日期分組
ORDER BY
    trip_date;                                           -- 依照日期排序
	
------------
SELECT
    "payment_type",                     -- 支付類型
    COUNT(*) AS number_of_trips,        -- 該支付類型下的行程數量
    AVG("tip_amount") AS avg_tip_amount -- 該支付類型下的平均小費金額
FROM
    green_tripdata
GROUP BY
    "payment_type"                      -- 依照支付類型分組
ORDER BY
    number_of_trips DESC;               -- 依照行程數量降序排序
	
------------
SELECT
    "lpep_pickup_datetime",
    "fare_amount",
    "tip_amount",
    ("tip_amount" / "fare_amount") * 100 AS tip_percentage -- 計算小費佔車資的百分比
FROM
    green_tripdata
WHERE
    "fare_amount" > 0 AND "tip_amount" > 0 -- 排除車資和小費為零的行程，避免除以零錯誤
ORDER BY
    tip_percentage DESC
LIMIT 5;

------------