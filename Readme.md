# NYC Taxi Data Pipeline
This project aims to establish an automated data pipeline for processing New York City taxi data (currently focusing on green taxi data). 

The process includes importing raw Parquet files into a PostgreSQL database, filtering the data, and training a machine learning model to predict tip amounts.

# Project Target
**Data Import**：https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

**Data Cleaning**: Extract and filter data from a repository.

**Machine Learning Prediction**: Train a model to predict the tip_amount(Tips) for taxi trips.

**Automation**: Use GitHub Actions for CI/CD to automate data pipelines and model training.

# Project Structure
```
NYC_TAXI/  
├──.github/  
│   ├──workflows/  
│   |   └──ci_cd.yml               
├── .venv/                          # Python .venv ignore
├── data/                           # ignore
│   ├── processed/                  # filtered Parquet files
│   └── raw/                        # Parquet files 
├── SQL/
│   ├── analysis.sql                
│   ├── create_table.sql            
│   └── predict.sql                 
├── src/
│   ├── __init__.py
│   └── data_pipeline.py            # main code
├── .env                            # setting file, ignore
├── .gitignore                      # Git ignore
├── pyproject.toml                  # toml
└── README.md                       # markdown
```

# Environment Setup (Local)
1. Clone the repository  
```
git clone https://github.com/HouHongHua/NYC_Taxi_Data_Pipeline.git
cd NYC_Taxi_Data_Pipeline
```
2. create a Python .venv
```
python -m venv .venv    
~ Windows
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process  
.\.venv\Scripts\activate  
```
3. Install Dependencies
```
This project uses pyproject.toml to manage dependencies. After activating the virtual environment, install all dependencies:  
pip install -e .
```
4. Set PostgreSQL
```
create a database
```
5. Prepare raw data
```
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
```
6. Configure .env file
```
DB_USER=postgres
DB_PASSWORD=your_actual_db_password 
DB_HOST=localhost
DB_PORT=5432
DB_NAME=NYC_Taxi
DB_TABLE_NAME=green_tripdata
RAW_DATA_PATH=data/raw 
PROCESSED_DATA_PATH=processed 
SQL_DIR=NYC_Taxi/sql 
```

# Make sure the virtual environment is activated  
**1. python src/data_pipeline.py**

```
1.Import the green taxi data from data/raw/ into the green_tripdata table in the NYC_Taxi database.

2.Execute the filter query defined in sql/predict.sql to extract the data from the database.

3.Store the filtered data in data/processed/green_tripdata_filtered.parquet.

4.Split the data into training and testing sets based on the TRAINING_MONTHS and TESTING_MONTHS settings in src/data_pipeline.py.

5.Train a random forest regression model to predict tip_amount

6.Evaluate the model's performance and save the trained model as green_taxi_tip_prediction_model.joblib

7.This project uses GitHub Actions to automate its CI/CD workflow
```

**2. Configure GitHub Secrets**
```
Set your sensitive information as Secrets in your GitHub repository. 

Go to your GitHub repository's Settings -> Secrets and variables -> Actions and add the following Secrets:

DB_USER: postgres

DB_PASSWORD: Your actual database password

DB_HOST: postgres (in the GitHub Actions service)

DB_PORT: 5432

DB_NAME: NYC_Taxi

DB_TABLE_NAME: green_tripdata

RAW_DATA_PATH: /github/workspace/data/raw

PROCESSED_DATA_PATH: /github/workspace/data/processed

SQL_DIR: /github/workspace/sql
```

**3. Data Management (CI/CD Environment)**

    - name: Download data files (IMPORTANT: You MUST adapt this step)
      run: |
        mkdir -p ${{ secrets.RAW_DATA_PATH }}
        mkdir -p ${{ secrets.PROCESSED_DATA_PATH }}
        # Example: Downloading data from a public URL
        wget -P ${{ secrets.RAW_DATA_PATH }} https://www.nyc.gov/html/tlc/downloads/parquet/green_tripdata_2025-01.parquet
        wget -P ${{ secrets.RAW_DATA_PATH }} https://www.nyc.gov/html/tlc/downloads/parquet/green_tripdata_2025-02.parquet
        wget -P ${{ secrets.RAW_DATA_PATH }} https://www.nyc.gov/html/tlc/downloads/parquet/green_tripdata_2025-03.parquet


**4. Push to GitHub**
```
Initial:
cd D:\Users\User\Desktop\Julia\NYC_Taxi
git init
git add .
git commit -m "feat: Initial project setup with data pipeline and model training"
git branch -M main
git remote add origin https://github.com/HouHongHua/NYC_Taxi_Data_Pipeline.git
git push -u origin main
```

```
Then:
cd D:\Users\User\Desktop\Julia\NYC_Taxi
git add .
git commit -m "docs: Update README and chore: Adjust model training/testing months"
git push origin main
```

