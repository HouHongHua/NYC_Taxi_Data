import pandas as pd
import numpy as np
import glob
import os
from sqlalchemy import create_engine
import gc
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import joblib 

# 載入 .env 檔案中的環境變數
load_dotenv()

# 載入檔案到SQL
def import_monthly_data_to_db(data_path: str, target_months: list, data_name: str, table_name: str):
    # 讀取 DB 連線資訊
    db_user = os.getenv('DB_USER', 'postgres')         
    db_password = os.getenv('DB_PASSWORD', 'your_default_password') 
    db_host = os.getenv('DB_HOST', 'localhost')       
    db_port = os.getenv('DB_PORT', '5432')            
    db_name = os.getenv('DB_NAME', 'NYC_Taxi')        
    table_name = os.getenv('DB_TABLE_NAME', table_name) 

    db_connection_str = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(db_connection_str)

    try:
        file_pattern = os.path.join(data_path, data_name)
        all_parquet_files = glob.glob(file_pattern)

        files_to_import = []
        for file_path in all_parquet_files:
            filename = os.path.basename(file_path)
            try:
                month_str = filename.split('-')[1].split('.')[0]
                if month_str in target_months:
                    files_to_import.append(file_path)
            except IndexError:
                print(f"Warning: '{filename}' Skip month.")

        files_to_import.sort()

        all_dfs = []    
        for file_path in files_to_import:
            filename = os.path.basename(file_path)
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
            print(f"'{filename}' Finish.")
            del df
            gc.collect()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"All Columns: {len(combined_df.columns)}")

        del all_dfs
        gc.collect()

        combined_df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"{target_months} data imported to PostgreSQL.")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        if 'engine' in locals() and engine:
            engine.dispose()
        print("\n End of Process.")

# 載入檔案做訓練
def train_tip_prediction_model(df: pd.DataFrame, training_months: list, testing_months: list):
    if df.empty:
        print("Empty Data")
        return

    TARGET = 'tip_amount'

    features = [
        "lpep_pickup_datetime",
        "lpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "ehail_fee",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "cbd_congestion_fee"
    ]

    df['pickup_hour'] = df['lpep_pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek 
    df['trip_duration_minutes'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60

    df = df[df['trip_duration_minutes'] > 0]
    df = df[df['trip_duration_minutes'] < 180] 

    features.extend(['pickup_hour', 'pickup_day_of_week', 'trip_duration_minutes'])

    features_final = [f for f in features if f not in ['lpep_pickup_datetime', 'lpep_dropoff_datetime']]

    X = df[features_final]
    y = df[TARGET]

    categorical_features = ['RatecodeID', 'pickup_hour', 'pickup_day_of_week']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    df['pickup_month_str'] = df['lpep_pickup_datetime'].dt.strftime('%m')
    train_df = df[df['pickup_month_str'].isin(training_months)].copy()
    test_df = df[df['pickup_month_str'].isin(testing_months)].copy()
    print(f"train_df length: {len(train_df)}")
    print(f"test_df length: {len(test_df)}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    X_train = train_df[features_final]
    y_train = train_df[TARGET]
    X_test = test_df[features_final]
    y_test = test_df[TARGET]

    model.fit(X_train, y_train)
    print("Train Finished.")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    model_output_path = 'src/model_output/green_taxi_tip_prediction_model.joblib'
    joblib.dump(model, model_output_path)
    print(f"\n Save Model: {model_output_path}")

# 連接SQL做資料整理
if __name__ == '__main__':
    RAW_DATA_PATH = os.getenv('RAW_DATA_PATH', './data/raw/')
    PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH', './data/processed/')

    target_months = ['01', '02', '03', '04']
    data_name = 'green_tripdata_2025-*.parquet'
    table_name = 'green_tripdata'

    import_monthly_data_to_db(RAW_DATA_PATH, target_months, data_name, table_name)

    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'your_default_password')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'NYC_Taxi')

    db_connection_str = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(db_connection_str)

    query = """
    SELECT
        "VendorID",
        "lpep_pickup_datetime",
        "lpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "ehail_fee",
        "improvement_surcharge",
        "total_amount",
        "congestion_surcharge",
        "cbd_congestion_fee"
    FROM green_tripdata
    WHERE "payment_type" = 1
    AND "fare_amount" > 0
    AND "tip_amount" >= 0
    AND "trip_distance" > 0
    AND "total_amount" > 0;
    """

    parquet_path = os.path.join(PROCESSED_DATA_PATH, 'green_tripdata_filtered.parquet')
    df = pd.read_sql(query, engine)
    df.to_parquet(parquet_path, index=False)

# 訓練
TRAINING_MONTHS = ['01', '02', '03'] 
TESTING_MONTHS = ['04'] 

train_tip_prediction_model(df, TRAINING_MONTHS, TESTING_MONTHS)
