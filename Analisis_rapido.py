
"""
COLUMNS:
-date_time : object - CONVERT TO DATE, format AAAA-MM-DD
-service: integer
-user: integer
-user_uid: integer
-user_time_zone: float / integer
-app_raw_latitude: float
-app_raw_timestamp: timestamp  - CONVERT TO DATE, format ISO8601
-app_raw_accuracy: float
-app_raw_longitude: float
-app_raw_distance: float
-app_raw_altitude: float
-app_raw_speed: float

DATABASES DIMENSIONS
1. 7762832 rows 
2. 954771 rows 
3. 1454921 rows 
4. 4525582 rows
5. 20185371 rows
6. 2207969 rows
7. 3747516 rows

OBSERVATIONS:
# - Infinite values (inf) and error values (<0) are found in multiple columns, which should be replaced by NaN for proper handling.
# - date_time and app_raw_timestamp columns should be converted to datetime for better temporal analysis. Around midnight date_time shifts to the next day.
# - Several columns contain missing values (NaN);  app_raw_latitude, app_raw_distance, app_raw_accuracy, app_raw_longitude, app_raw_altitude, app_raw_speed:
        Apply interpolation to app_raw_latitude, app_raw_distance, app_raw_longitude, app_raw_speed
        Apply median imputation to app_raw_accuracy, app_raw_altitude
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

base_path = input('Please enter the base path of the CSV files (include a trailing slash "/"): ')

for i in range(1, 8):
    file_name = f'Locationdistance_eb2prod_{i}.csv'
    file_path = base_path + file_name
    
    try:
        df = pd.read_csv(file_path)

        print(f"\nFile: {file_name}")
        #print(df.head())
        #print(df.info())

        #print("\nStatistical Summary:")
        #print(df.describe())

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['app_raw_latitude'] = df['app_raw_latitude'].apply(lambda x: np.nan if x <= 0 else x)
        df['app_raw_longitude'] = df['app_raw_longitude'].apply(lambda x: np.nan if x <= 0 else x)
        df['app_raw_speed'] = df['app_raw_speed'].apply(lambda x: np.nan if x <= 0 else x)
        df['app_raw_distance'] = df['app_raw_distance'].apply(lambda x: np.nan if x <= 0 else x)

        #missing_values_before = df.isnull().sum()
        #print("Missing values before cleaning:")
        #print(missing_values_before)

        df['date_time'] = pd.to_datetime(df['date_time'], format='%Y-%m-%d', errors='raise')
        df['app_raw_timestamp'] = pd.to_datetime(df['app_raw_timestamp'], format='ISO8601', errors='raise')
        
        def interpolate_values(user_data):
            user_data = user_data.sort_values(by='app_raw_timestamp')

            user_data['app_raw_latitude'] = user_data['app_raw_latitude'].interpolate(method='linear')
            user_data['app_raw_longitude'] = user_data['app_raw_longitude'].interpolate(method='linear')
            user_data['app_raw_speed'] = user_data['app_raw_speed'].interpolate(method='linear')
            user_data['app_raw_distance'] = user_data['app_raw_distance'].interpolate(method='linear')

            user_data['app_raw_latitude'] = user_data['app_raw_latitude'].ffill().bfill()
            user_data['app_raw_longitude'] = user_data['app_raw_longitude'].ffill().bfill()
            user_data['app_raw_speed'] = user_data['app_raw_speed'].ffill().bfill()
            user_data['app_raw_distance'] = user_data['app_raw_distance'].ffill().bfill()

            return user_data

        df = interpolate_values(df)

        imputer = SimpleImputer(strategy='median')

        df['app_raw_accuracy'] = imputer.fit_transform(df[['app_raw_accuracy']])
        df['app_raw_altitude'] = imputer.fit_transform(df[['app_raw_altitude']])

        missing_values_after = df.isnull().sum()
        print("\nMissing values after interpolation and imputation:")
        print(missing_values_after)

    except Exception as e:
        print(f"An unexpected error occurred while processing the file {file_name}: {e}")
