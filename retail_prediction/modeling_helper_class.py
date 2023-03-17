import pandas as pd
import data_preprocessing_helpers as dp

import numpy as np

from sklearn.tree import DecisionTreeRegressor as dtree
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.linear_model import LinearRegression as lin_reg

import pandas as pd
import data_preprocessing_helpers as dp 
from sklearn.tree import DecisionTreeRegressor as dtree
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import RandomizedSearchCV

class model_object:
    def __init__(self, model, df, target, scorer):
        print('Welcome to model object')
        self.base_model = model
        self.df = df
        self.x = df[[x for x in df.columns if x != target]]
        self.y = df[target]
        self.target = target
        self.scorer = scorer
        self.model_logs = pd.DataFrame(columns = ['estimator', 'hyperparameters','features', 'festure_selection_strategy', 'cv_scores', 'mean_cv', 'std_cv'])
        self.params_grid = self.get_params_grid('dt')
        return 


    def get_params_grid(self, model_name):
        if model_name == 'dt':
            params_grid = {
                'max_depth' : range(1,100),
                'max_features' : range(1,1000),
                #'min_samples_split' : range(1,1000),
                #'min_samples_leaf' : range(1,1000),
                'n_estimators' : range(1,1000)
            }
        else:
            print('Not yet Supported')
        return params_grid

    def log_model(self, cv_scores):
        scores_to_log = {
            'estimator' : self.model,
            'hyperparameters' : self.model.params,
            'features' : self.model.feature_names_in,
            'feature_sel_strat' : None,
            'cv_scores' : cv_scores,
            'mean_cvs' : cv_scores.mean(),
            'std_cvs' : cv_scores.std()
        }

        return


    def feature_selection(self, model, strategy):
        if strategy == 'rfecv':
            print('Performing RFE CV')
            rfe = RFECV(model, 
                      scoring = self.scorer, 
                      n_jobs=-1,
                      step = 3
                      ).fit(self.x, self.y)
            
            rfe_rankings = pd.DataFrame(rfe.ranking_, 
                                        index = rfe.feature_names_in_, columns = ['feature_rfe_rank'])
            
        elif strategy == 'rfe':
            print('RFE without CV')
            #rfe = RFE()
        elif strategy == 'select_from_model':
            rfe = SelectFromModel(self.best_model).fit(self.x, self.y)
            rfe_rankings = pd.DataFrame(rfe.ranking_, 
                                        index = rfe.feature_names_in_, columns = ['feature_rfe_rank'])
            
        else:
            print('Nothing here')
            
        return rfe, rfe_rankings

        # Strategies: RFECV, SelectFromModel


    def cross_validate(self, model, x, y):
        print(f'Validating Model {model} with {len(x.columns)} features')
        cv_scores = cross_val_score(model, x, y, 
                                    scoring = self.scorer, n_jobs = -1)
        print(f'Mean: {cv_scores.mean()}, STD: {cv_scores.std()}')
        print('_'*25)
        return cv_scores.mean(), cv_scores.std(), cv_scores
    
    def tune_hyperparams(self, model, x, y, n_iter):
        print(f'Tuning {model}, with  {len(x.columns)} features. {n_iter} Iterations')

        tuning_res = RandomizedSearchCV(model, 
                                 self.params_grid, 
                                 scoring = self.scorer, 
                                 n_iter = n_iter, 
                                 n_jobs = -1).fit(x, y)
        return tuning_res
    

    def fit_evaluate_model(self, model, x, y):
        print(f"Fitting model {model}, with {len(x.columns)} features")
        fit_model = model.fit(x,y)

        cvm, cvs, _ = self.cross_validate(model, x, y)

        return fit_model, cvm, cvs
    

    def tune_model(self, model, x, y, n_iter):

        tuning_res = self.tune_hyperparams(model, x, y, n_iter)
        tuning_res_df = pd.DataFrame(tuning_res.cv_results_)

        tuned_model = tuning_res.best_estimator_

        cvm_tuned = tuning_res_df.loc[tuning_res_df.rank_test_score == 1,
                                                          'mean_test_score'].iloc[0]
        cvs_tuned = tuning_res_df.loc[tuning_res_df.rank_test_score == 1,
                                                          'std_test_score'].iloc[0]

        print(f'Tuned Baseline Score: {cvm_tuned}, {cvs_tuned}')

        return tuned_model, cvm_tuned, cvs_tuned, tuning_res
    

    def select_features(self, model, x, y, strategy = 'rfecv'):
        print(f'Selecting Features from x with {len(x.columns)} features')
        feat_sel, feature_ranks = self.feature_selection(model, strategy = strategy)
        x_sel = pd.DataFrame(feat_sel.transform(x), 
                             columns=feat_sel.get_feature_names_out())
        print(f'Selected {len(feat_sel.get_feature_names_out())} features')

        return x_sel, feature_ranks, feat_sel


    def get_best_model(self):
        n_iter = 10
        
        # 1. Cross Validate Base Model
        self.model_bl, self.cvm_bl, self.cvs_bl = self.fit_evaluate_model(self.base_model, self.x, self.y)
        print(f'Baseline Score: {self.cvm_bl}, {self.cvs_bl}')
        
        # 2. Small Random Search for decent model
        print('Tuning')
        self.model_bl_tuned_1, self.cvm_bl_tuned_1, self.cvs_bl_tuned_1, self.tuning_res_bl = self.tune_model(self.base_model, self.x, self.y, n_iter = n_iter)
        self.model_bl_tuned, self.cvm_bl_tuned, self.cvs_bl_tuned = self.fit_evaluate_model(self.model_bl_tuned_1, self.x, self.y)

        
        # 3.1 Feature Selection
        print('FeatSelecting on BaseLine Model')
        self.x_small, self.feature_ranks_bl, self.feat_sel_bl = self.select_features(self.model_bl, self.x, self.y, strategy = 'rfecv')
        self.model_blfs, self.cvm_blfs, self.cvs_blfs = self.fit_evaluate_model(self.model_bl, self.x_small, self.y)
        print(f'FS - BL Score : {self.cvm_blfs}, {self.cvs_blfs}')
        
        # 3.2 Grid Search for best HyperParameters
        print('Tuning Feature Selected BaseLine Model')
        self.model_blfs_tuned_1, self.cvm_blfs_tuned_1, self.cvs_blfs_tuned_1, self.tuning_res_blfs = self.tune_model(self.model_bl, self.x_small, self.y, n_iter = n_iter)
        self.model_blfs_tuned, self.cvm_blfs_tuned, self.cvs_blfs_tuned = self.fit_evaluate_model(self.model_blfs_tuned_1, self.x_small, self.y)
        print(f'FS - BL - Tuned Score: {self.cvm_blfs_tuned_1}, {self.cvs_blfs_tuned_1}')
        print(f'FS - BL - Tuned Score Again: {self.cvm_blfs_tuned}, {self.cvs_blfs_tuned}') 
        

        # 4.1 Feature Selection for tuned model
        print('FeatSelecting on Tuned Model')
        self.x_small_tuned, self.feature_ranks_tuned, self.feat_sel_tuned = self.select_features(self.model_bl_tuned, self.x, self.y, strategy = 'rfecv')
        self.model_tunedfs, self.cvm_tunedfs, self.cvs_tunedfs = self.fit_evaluate_model(self.model_bl_tuned, self.x_small_tuned, self.y)
        print(f'FS - Tuned Score : {self.cvm_tunedfs}, {self.cvs_tunedfs}')
        
        # 4.2 Grid Search for best HyperParameters
        print('Tuning Feature Selected Tuned Model')
        self.model_tunedfs_tuned_1, self.cvm_tunedfs_tuned_1, self.cvs_tunedfs_tuned_1, self.tuning_res_tunedfs = self.tune_model(self.model_bl_tuned, self.x_small_tuned, self.y, n_iter = n_iter)
        self.model_tunedfs_tuned, self.cvm_tunedfs_tuned, self.cvs_tunedfs_tuned = self.fit_evaluate_model(self.model_tunedfs_tuned_1, self.x_small_tuned, self.y)
        print(f'FS - Tuned - Tuned Score: {self.cvm_tunedfs_tuned_1}, {self.cvs_tunedfs_tuned_1}')
        print(f'FS - Tuned - Tuned Score Again: {self.cvm_tunedfs_tuned}, {self.cvs_tunedfs_tuned}') 

        return 