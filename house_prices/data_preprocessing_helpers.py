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

train_file_path = './../data/house_prices/train.csv'
test_file_path = './../data/house_prices/test.csv'
id_var = 'Id'
col_dtypes = {
    
}


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
        dtype = col_dtypes
    )

    print(f'Data has {df.shape} shape')
    
    return df


def clean_data(df):
    """Function to drop rows. Should be used with caution, will explore utility.
    """

    return 


def feature_engineering(df):
    """ Makes new columns needed for modelling.
    input: data-frame
    output: data-frame with tranformed / added features
    """
    print('Making New Features')
    var_groups = pd.read_csv('./house_price_variable_groups.csv')
    categorical_columns = list(var_groups.loc[var_groups.dtype == 'category', "variable_name"])

    df['MS_story'] = [0 if x in [20,30,40,45,50, 120, 150] else 1 for x in df['MSSubClass']]
    df['MS_story'] = df['MS_story'].astype('category')
    df['residential'] = 1*(df.MSZoning.str.find('R') == 0)

    df['BsmtScore'] = df.BsmtQual.map({'Ex' : 10, 'Gd' : 8, 'TA' : 5, 'Fa' : 3, 'none' : 0})
    df['BsmtScore2'] = df.BsmtCond.map({'Ex' : 10, 'Gd' : 8, 'TA' : 5, 'Fa' : 3, 'none' : 0})
    df['BsmtScoreTotal'] = df['BsmtScore'] + df['BsmtScore2']

    df['ExterScore'] = df.ExterQual.map({'Ex' : 10, 'Gd' : 8, 'TA' : 5, 'Fa' : 3, 'none' : 0})
    df['FireplaceScore'] = df.FireplaceQu.map({'Ex' : 10, 'Gd' : 8, 'TA' : 5, 'Fa' : 3, 'Po' : 1, 'none' : 0})
    df['KitchenScore'] = df.FireplaceQu.map({'Ex' : 10, 'Gd' : 8, 'TA' : 5, 'Fa' : 3, 'Po' : 1, 'none' : 0})
    df['has_garage'] = 10.0*(df.GarageQual != 'none')

    df['QualScore'] = df['BsmtScoreTotal'] + df['ExterScore'] + df['FireplaceScore'] 
    df['QualScore2'] = df['ExterScore'] + df['OverallQual'] 

    df['BsmtScore']=df['BsmtScore'].fillna(0)
    df['QualScore']=df['QualScore'].fillna(0)
    df['QualScore2']=df['QualScore2'].fillna(0)
    df['BsmtFullBath']=df['BsmtFullBath'].fillna(0)
    df['BsmtHalfBath']=df['BsmtHalfBath'].fillna(0)
    df['age'] = df.YrSold - df.YearBuilt
    df['GarageArea']=df['GarageArea'].fillna(0)
    df['FireplaceScore'] = df['FireplaceScore'].fillna(0)
    df['has_pool'] = ~df.PoolQC.isna()
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['BsmtScore2'] = df['BsmtScore2'].fillna(0)
    df['KitchenScore'] = df['KitchenScore'].fillna(0)
    df['BsmtScoreTotal'] = df['BsmtScoreTotal'].fillna(0)

    df['remodel'] = ~df['YearRemodAdd'].isna()

    print('Setting Column Type')
    df[categorical_columns] = df[categorical_columns].astype('category')

    return df


def missing_value_imputation(df):
    return df


def advanced_feature_engineering(df):
    """ Creates features that is dependent on training set. 
    The test set also has to use data inferred from training set, like traffic at a particular location.
    Inputs: DataFrame (Train/Test) + inferred data from training set
    Output: DF with added columns
    Remarks: The actual steps in this function depends on the project. Function for Framework.
    """
    print('Making new features using Training Set')
    neighborhood_scores = pd.read_csv('neighbourhood_scores.csv', index_col='cat_label')
    df['NeighborhoodScore'] = df['Neighborhood'].apply(lambda x: neighborhood_scores.loc[x, 'neighbourhood_score'])


    return df


def split_data(df, target = None, include_cols = [], ohe = False, validation_split = None):
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
        print(f'Subsetting {len(include_cols)} Columns')
        x = df[include_cols]

    if ohe:
        print('One Hot Encoding')
        x = pd.get_dummies(x)
        
    if validation_split is None:
        return x, y
        print(f'output data has {x.shape[0]} rows')
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, random_state = 56, test_size = validation_split)
        print(f'output data has {x_train.shape[0]} + {x_val.shape[0]} rows')
        return x_train, y_train, x_val, y_val
        

def get_preprocessed_data(file_path = '', include_cols = [], ohe = False):
    
    # Get training data to infer these things
    df = load_data(file_path)

    print('Feature Engineering')
    df = feature_engineering(df)
    df = advanced_feature_engineering(df)  
    
    if len(include_cols) > 0:
        print('Subsetting Columns.')
        df = df[include_cols]
        print(f'Data has {df.shape} shape')
              
    if ohe:
        print('One Hot Encoding')
        df = pd.get_dummies(df)    
        
    return df
