import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print('Loading from customer_churn folder')

train_file_path = './churn_prediction.csv'
id_var = 'customer_id'

col_dtypes = {
    'gender' : 'category',
    'occupation' : 'category',
    'city' : 'category',
    'branch_code' : 'category',
    'churn' : 'bool',
    'customer_nw_category' : 'category'
}

def load_data(file_path = train_file_path):
    
    if file_path == 'test':
        file_path = test_file_path

    print(f'Loading {file_path}')    
    df = pd.read_csv(
        file_path,
        index_col = id_var,
        dtype = col_dtypes
    )
    print(f'Data has {df.shape} shape')
    
    return df

def load_preprocessed_data(file_path = train_file_path):
    df = load_data(file_path)
    df = feature_engineering(df)
    df = missing_value_imputation(df)
    return df
    
## Custom Functions for this dataset

## Framework Functions
def isnan(x):
    return x != x


def missing_value_imputation(df):
    df['gender'] = df.gender.cat.add_categories('na').fillna('na')
    df['city'] = df.city.cat.add_categories('na').fillna('na')
    df['dependents'] = df['dependents'].fillna(0)
    df['occupation'] = df['occupation'].fillna('self_employed')
    df['days_since_last_transaction'] = df['days_since_last_transaction'].fillna(999)
    return df


def advanced_feature_engineering(df, median_ages, last_name_survival, cabin_survival, categories):
    """ Function to load raw data and deliver a DataFrame ready for modelling.
    Inputs: DataFrame, Dictionary to set Data types and drop columns
    Output: DF ready for modelling
    Remarks: The actual steps in this function depends on the project. Function for Framework.
    """
    return df




def feature_engineering(df):
    """ Makes new columns needed for modelling.
    input: data-frame
    output: data-frame with tranformed / added features
    """
    df['balance_ratio'] = df['current_month_balance']/df['previous_month_end_balance']
    df['debit_ratio'] = df['current_month_debit']/df['previous_month_debit']
    df['credit_ratio'] = df['current_month_credit']/df['previous_month_credit']
    df['age_group'] = pd.cut(df.age, [0,18,25,30,60,100])
    df['liquid_ratio'] = pd.cut(round(df.current_month_debit/df.average_monthly_balance_prevQ,0), [-100000, -2,-1,0,1,2,100,10000])

    return df


def get_preprocessed_data(file_path = train_file_path, include_cols = [], ohe = True):
    
    # Get training data to infer these things
    df = load_data(file_path)        
    df = feature_engineering(df)
    df = missing_value_imputation(df)
    df = advanced_feature_engineering(df)
    
    if len(include_cols) > 0:
        print('Subsetting Columns.')
        df = df[include_cols]
        print(f'Data has {df.shape} shape')
              
    if ohe:
        print('One Hot Encoding')
        df = pd.get_dummies(df)    
        
    return df
 
    
## Framework Functions
def isnan(x):
    return x != x


def advanced_feature_engineering(df):
    """ Creates features that is dependent on training set. 
    The test set also has to use data inferred from training set, like traffic at a particular location.
    Inputs: DataFrame (Train/Test) + inferred data from training set
    Output: DF with added columns
    Remarks: The actual steps in this function depends on the project. Function for Framework.
    """
    
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
        print('Subsetting Columns')
        x = df[include_cols]
        
    if ohe:
        print('One Hot Encoding')
        x = pd.get_dummies(x,drop_first=True)
        
    if validation_split is None:
        return x, y
        print(f'output data has {x.shape[0]} rows')
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, random_state = 56, test_size = validation_split, stratify = y)
        print(f'output data has {x_train.shape[0]} + {x_val.shape[0]} rows')
        return x_train, y_train, x_val, y_val
        