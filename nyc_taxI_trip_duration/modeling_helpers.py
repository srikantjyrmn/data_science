# Modelling helpers
"""
Functions:
1. scorer - RMSE for this dataset
2. evaluate_predictions - evaluate more error metrics
3. scale_x -  function to return scaled DF. [TODO: Evaluate different scaling methods.]
4. build_evaluate_model - 
"""
import pandas as pd
import data_preprocessing_helpers as dp
from sklearn.linear_model import LinearRegression as lin_reg
from sklearn.tree import DecisionTreeRegressor as dtree
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns

scorer = make_scorer(mean_squared_error, squared = False)

def evaluate_predictions(preds, actuals):
    """Evaluate a set of predictions against actuals and returns the required Error metrics
    """
    # DF
    preds_df = pd.DataFrame(preds, columns = {'predictions'}, index = actuals.index)
    preds_df['actuals'] = actuals
    preds_df['error'] = actuals - preds
    
    # Plot
    sns.scatterplot(preds_df, x  = 'predictions', y = 'actuals')
    
    # Errors
    mape = mean_absolute_percentage_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    rmsle = mean_squared_log_error(actuals, preds, squared=False)
    rmse = mean_squared_error(actuals, preds, squared=False)
    r2 = r2_score(actuals, preds)
    
    errors = {
        'rmse' : rmse,
        'r2': r2,
        'rmsle' : rmsle,
        'mape': mape,
        'mae': mae
    }
    
    print(f'RMSE: {rmse}, r2: {r2}, MAPE: {mape}, MAE: {mae}, RMSLE : {rmsle}')
    
    return errors, preds_df


def scale_x(x, xval):
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = pd.DataFrame(scaler.transform(x), columns = scaler.feature_names_in_, index = x.index)
    xval_scaled = pd.DataFrame(scaler.transform(xval), columns = scaler.feature_names_in_, index = xval.index)
    return x_scaled, xval_scaled


def build_evaluate_model(model, df, select_cols, validation_split = 0.25, scale = False):
    """ Builds given model and returns performance metrics as specified globally.
    Steps: Split given dataset into train and test
    """

    print(f'Building & Evaluating a {model}')
    print('-----')
    
    # Split into train and validation
    x, y, xval, yval = dp.split_data(df, target = 'trip_duration', include_cols = select_cols, validation_split = validation_split)
    
    if scale:
        x, xval = scale_x(x, xval)
    
    # Fit model and Evaluate Predictions
    model.fit(x,y)
    preds = model.predict(xval)
    print(f'Model Scores: {model.score(x,y)}, {model.score(xval,yval)}')
    
    scores, pred_df = evaluate_predictions(actuals = yval, preds = preds)
        
    # Cross Validation
    print('Performing Cross Validation')
    x, y = dp.split_data(df, target = 'trip_duration', include_cols = select_cols)      

    cv_scores = cross_val_score(model, x, y, scoring = scorer)
    scores['cv_scores'] = cv_scores
    print(f'CV Mean RMSE: {cv_scores.mean()}, Std: {cv_scores.std()}')
    print('-----')
    
    return scores, model, pred_df