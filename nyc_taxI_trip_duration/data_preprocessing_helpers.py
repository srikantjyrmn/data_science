# Data preprocessing Helpers
""" Functions:
1. Global variable definitions
Distance Functions
1. haversine
2. minowski
3. average_distances (for osrm)
4. cosine_distance

Others
3. get_time_of_day
4. isnan

Data Preprocessing Functions
5. load_data
6. clean_data
7. feature_engineering
8. get_preprocessed_data
9. missing_value_imputation
9. advanced_feature_engineering
10. split_data
11. get_osrm_merged_data
"""

import pandas as pd
import numpy as np
import haversine as hs
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split

train_file_path = './nyc_taxi_trip_duration.csv'
id_var = 'id'
col_dtypes = {
    'vendor_id' : 'category',
    'store_and_fwd_flag' : 'category',
    'passenger_count' : 'int8'
}
datetime_cols = ['pickup_datetime', 'dropoff_datetime']
 
    
## Custom Functions for this dataset
# Distance Functions
def hv(row):
    pickup_loc = (row['pickup_latitude'], row['pickup_longitude'])
    drop_loc = (row['dropoff_latitude'], row['dropoff_longitude'])
    return hs.haversine(pickup_loc, drop_loc)


def minowski(row):
    return abs(row['pickup_latitude'] - row['dropoff_latitude']) + abs(row['pickup_longitude'] - row['dropoff_longitude'])


def average_distances(list):
    float_list = [float(x) for x in list]
    avg = np.mean(float_list)
    return avg


def scpy_cosine(row):
    pickup_loc = (row['pickup_latitude'], row['pickup_longitude'])
    drop_loc = (row['dropoff_latitude'], row['dropoff_longitude'])
    return cosine(pickup_loc, drop_loc)


def get_time_of_day(pickup_hour):
    if 8 <= pickup_hour <= 11:
        return 'morning_rush'
    elif 12 <= pickup_hour <= 18:
        return 'office_hours'
    elif 19 <= pickup_hour <= 21:
        return 'evening_rush'
    elif pickup_hour in [22, 23, 0]:
        return 'night_time'
    else: return 'early_morning'


## Framework Functions
def isnan(x):
    return x != x


def load_data(file_path = ''):
    
    if file_path == '':
        file_path = train_file_path
    elif file_path == 'test':
        file_path = test_file_path

    print(f'Loading {file_path}')    
    df = pd.read_csv(
        file_path,
        index_col = id_var,
        dtype = col_dtypes,
        parse_dates = datetime_cols
    )
    print(f'Data has {df.shape} shape')
    
    return df


def clean_data(df):
    """Function to drop rows. Should be used with caution, will explore utility.
    """
    
    print('Removing Rows for improper location')
    df['proper_latitude_pickup'] = [40 <= x <= 42 for x in df.pickup_latitude]
    df['proper_latitude_drop'] = [40 <= x <= 42 for x in df.dropoff_latitude]
    df['proper_longitude_pickup'] = [-75 <= x <= -73 for x in df.pickup_longitude]
    df['proper_longitude_drop'] = [-75 <= x <= -73 for x in df.dropoff_longitude]
    df['improper_location'] = df.proper_latitude_drop & df. proper_latitude_pickup & df.proper_longitude_pickup & df.proper_longitude_drop
    
    remove_for_improper_loc = df.improper_location
    print(f'{sum(remove_for_improper_loc)}')
    df = df.loc[remove_for_improper_loc]
    print(f'There are now {df.shape[0]} rows')
    
    #remove_2 = df.trip_duration < 60
#    print('Removing Rows for improper duration')
#    print(sum(remove_2))
#    df = df.loc[~remove_2]
#    print('Removing trip_durations more than 70 mins and less than 3minutes, since that is the .995th percentile')
#    df = df.loc[df.trip_duration < 70*60]
#    df = df.loc[df.trip_duration > 3*60]
#    df = df.loc[df.pickup_latitude != df.dropoff_latitude]
#    df = df.loc[df.pickup_longitude != df.dropoff_longitude]
#    print(f'There are now {df.shape[0]} rows')
    return df.drop(['proper_latitude_pickup', 'proper_latitude_drop', 'proper_longitude_pickup', 'proper_longitude_drop', 'improper_location'], axis = 1)


def feature_engineering(df):
    """ Makes new columns needed for modelling.
    input: data-frame
    output: data-frame with tranformed / added features
    """
    print('Making New Features')
    
    # Log Transform y-variable
    df['trip_duration'] = np.log1p(df.trip_duration)
    
    # Time Related
    df['pickup_dow'] = df.pickup_datetime.dt.weekday
    df['pickup_doy'] = df.pickup_datetime.dt.dayofyear
    df['pickup_week'] = df.pickup_datetime.dt.isocalendar().week.astype('int64')
    df['pickup_hour'] = df.pickup_datetime.dt.hour
    
    df['pickup_dow']=df['pickup_dow'].astype('uint8')
    df['pickup_doy']=df['pickup_doy'].astype('uint16')
    df['pickup_week']=df['pickup_week'].astype('uint8')
    df['pickup_hour']=df['pickup_hour'].astype('uint8')
    
    ## Binning Times - do more carefully. How will you select better split?
    df['day_type'] = ['weekend' if x in [0,6] else 'weekday' for x in df.pickup_dow]
    df['time_of_day'] = [get_time_of_day(x) for x in df.pickup_hour]
    df['time_slot_type'] = df.day_type + '_' + df.time_of_day
    
    df['day_type'] = df['day_type'].astype('category')
    df['time_of_day'] = df['time_of_day'].astype('category')
    df['time_slot_type'] = df['time_of_day'].astype('category')
    
    ## 2.2 Distances
    df['trip_distance'] = df.apply(lambda row: hv(row), axis = 1)
    df['log_distance'] = np.log1p(df.trip_distance)
    
    df['m_distance'] = df.apply(lambda row: minowski(row), axis = 1)
    df['log_m_distance'] = np.log1p(df.m_distance)
    
    
    # Binning Lat-longs
    lat_long_rounding_n = 3
    df['pickup_lat_bin'] = round(df.pickup_latitude, lat_long_rounding_n)
    df['pickup_lon_bin'] = round(df.pickup_longitude, lat_long_rounding_n)
    df['drop_lat_bin'] = round(df.dropoff_latitude, lat_long_rounding_n)
    df['drop_lon_bin'] = round(df.dropoff_longitude, lat_long_rounding_n)
    print(f'Data has {df.shape} shape')
    
    return df


def missing_value_imputation(df):
    return df


def advanced_feature_engineering(df, pickup_lat_bin_count, pickup_lon_bin_count, drop_lat_bin_count, drop_lon_bin_count):
    """ Creates features that is dependent on training set. 
    The test set also has to use data inferred from training set, like traffic at a particular location.
    Inputs: DataFrame (Train/Test) + inferred data from training set
    Output: DF with added columns
    Remarks: The actual steps in this function depends on the project. Function for Framework.
    """
    print('Making new features using Training Set')
    
    # Get Traffic information from training set, since this is an inferred column, not real time
    df['pickup_lat_bin_count'] = [pickup_lat_bin_count[x] for x in df.pickup_lat_bin]
    df['pickup_lon_bin_count'] = [pickup_lon_bin_count[x] for x in df.pickup_lon_bin]
    df['drop_lat_bin_count'] = [drop_lat_bin_count[x] for x in df.drop_lat_bin]
    df['drop_lon_bin_count'] = [drop_lon_bin_count[x] for x in df.drop_lon_bin]
    
    print(f'Added Bin Counts for traffic. Data has {df.shape} shape')
    return df


def split_data(df, target = None, include_cols = [], validation_split = None):
    """ Takes in a cleaned dataframe and return x_train, y_train and x_val, y_val
    """

    print(f'input data has {df.shape[0]} rows')
    if target:
        y = df[target]
        x = df.drop(target, axis = 1)
    else:
        y = []
        x = df
    
    if len(include_cols) > 0:
        print(f'Subsetting Columns {include_cols}')
        x = df[include_cols]
        
    if validation_split is None:
        return x, y
        print(f'output data has {x.shape[0]} rows')
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, random_state = 56, test_size = validation_split)
        print(f'output data has {x_train.shape[0]} + {x_val.shape[0]} rows')
        return x_train, y_train, x_val, y_val
        

def get_preprocessed_data(file_path = '', include_cols = [], ohe = True):
    
    # Get training data to infer these things
    df = load_data(file_path)
    
    print('Cleaning Data')
    df = clean_data(df)
    df = feature_engineering(df)

    pickup_lat_bin_count = df.pickup_lat_bin.value_counts()
    pickup_lon_bin_count = df.pickup_lon_bin.value_counts()
    drop_lat_bin_count = df.drop_lat_bin.value_counts()
    drop_lon_bin_count = df.drop_lon_bin.value_counts()
    
    if file_path == 'test':
        df = load_data(file_path)

    df = advanced_feature_engineering(df,
                                      pickup_lat_bin_count, pickup_lon_bin_count,
                                      drop_lat_bin_count, drop_lon_bin_count)
    

    drop_cols = datetime_cols + ['store_and_fwd_flag']
    print(f'Dropping datetime columns and store_fwd_flag. {drop_cols}')
    df = df.drop(drop_cols, axis = 1)    
    
    
    if len(include_cols) > 0:
        print('Subsetting Columns.')
        df = df[include_cols]
        print(f'Data has {df.shape} shape')
              
    if ohe:
        print('One Hot Encoding')
        df = pd.get_dummies(df)    
        
    return df
        
    
def get_osrm_merged_data(update=True):
    if update:
        df = get_preprocessed_data()
        print('Merging OSRM Data')
        df_osrm_1 = pd.read_csv('./fastest_routes_train_part_1.csv', index_col = 'id')
        df_osrm_2 = pd.read_csv('./fastest_routes_train_part_2.csv', index_col = 'id')
        df_osrm = pd.concat([df_osrm_1,df_osrm_2], axis = 0)
        print('Joining')
        df = df.merge(df_osrm, how = 'left', left_index=True, right_index=True)
        
        # Some feature Engineering and Dropping NAs
        df = df.dropna()
        df['av_step_distance'] = [average_distances(x.split('|')) for x in df['distance_per_step']]
        df['av_step_time'] = [average_distances(x.split('|')) for x in df['travel_time_per_step']]
        df['cosine_distance'] = df.apply(lambda row: scpy_cosine(row), axis = 1)
        print(df.shape)
        
        df.to_parquet('osrm_merged_data.parquet', index=True)
        
    else:
        df = pd.read_parquet('osrm_merged_data.parquet')
        # todo: have to handle column types. - parquet
    return df