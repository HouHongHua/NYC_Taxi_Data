SELECT
    "VendorID",
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount", -- 小費是我們的預測目標，所以需要包含
    "tolls_amount",
    "ehail_fee", -- 根據您的 green_taxi.dtypes，這個欄位存在
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "cbd_congestion_fee"
FROM
    green_tripdata -- 從綠色計程車資料表加載數據
WHERE
    "payment_type" = 1 AND       -- 篩選信用卡支付 (請根據實際編碼確認)
    "fare_amount" > 0 AND        -- 車資必須大於 0
    "tip_amount" >= 0 AND        -- 小費金額不能是負數
    "trip_distance" > 0 AND      -- 行程距離必須大於 0
    "total_amount" > 0;          -- 總金額必須大於 0
