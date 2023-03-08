import pandas as pd
import data_preprocessing_helpers as dp
import modeling_helpers as mh

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
import modeling_helpers as mh
# import cross_val_score

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
        return cv_scores
    
    def tune_hyperparams(self, model, x, y, n_iter):
        print(f'Tuning {model}, with  {len(x.columns)} features')

        tuning_res = RandomizedSearchCV(model, 
                                 self.params_grid, 
                                 scoring = self.scorer, 
                                 n_iter = n_iter, 
                                 n_jobs = -1).fit(x, y)
        return tuning_res

    def get_best_model(self):
        n_iter = 100
        
        # 1. Cross Validate Base Model
        baseline_cvs = self.cross_validate(self.base_model, self.x, self.y)
        self.model_bl = self.base_model.fit(self.x, self.y)

        self.cvm_baseline = baseline_cvs.mean()
        self.cvs_baseline = baseline_cvs.std()

        print(f'Baseline Score: {self.cvm_baseline}, {self.cvs_baseline}')
        
        # 2. Small Random Search for decent model
        print('Tuning')
        baseline_tuning_res = self.tune_hyperparams(self.base_model, self.x, self.y, n_iter)
        
        baseline_tuning_res_df = pd.DataFrame(baseline_tuning_res.cv_results_)
        self.model_bl_tuned = baseline_tuning_res.best_estimator_

        self.cvm_bl_tuned = baseline_tuning_res_df.loc[baseline_tuning_res_df.rank_test_score == 1,
                                                          'mean_test_score'].iloc[0]
        self.cvs_bl_tuned = baseline_tuning_res_df.loc[baseline_tuning_res_df.rank_test_score == 1,
                                                          'std_test_score'].iloc[0]
        

        print(f'Tuned Baseline Score: {self.cvm_bl_tuned}, {self.cvs_bl_tuned}')


        # 3. Feature Selection
        print('FeatSelecting')
        feature_sel_bl, self.feature_ranks_bl = self.feature_selection(self.model_bl, strategy = 'rfecv')
        self.feat_sel_bl = feature_sel_bl

        self.x_small = pd.DataFrame(feature_sel_bl.transform(self.x), 
                                          columns=feature_sel_bl.get_feature_names_out())
        
        cvscores_bl_featsel = self.cross_validate(self.model_bl, self.x_small, self.y)

        self.cvm_bl_fs = cvscores_bl_featsel.mean()
        self.cvs_bl_fs = cvscores_bl_featsel.std()
        print(f'FS - BL Score : {self.cvm_bl_fs}, {self.cvs_bl_fs}')

        
        # 4. Grid Search for best HyperParameters
        print('Tuning')
        self.blfs_tuning_res = self.tune_hyperparams(self.model_bl, self.x_small, self.y, n_iter)
        self.model_blfs_tuned = self.blfs_tuning_res.best_estimator_
        
        cvscores_blfs_tuned = pd.DataFrame(self.blfs_tuning_res.cv_results_)

        self.cvm_blfs_tuned_1 = cvscores_blfs_tuned.loc[cvscores_blfs_tuned.rank_test_score == 1,'mean_test_score'].iloc[0]
        self.cvs_blfs_tuned_1 = cvscores_blfs_tuned.loc[cvscores_blfs_tuned.rank_test_score == 1,'std_test_score'].iloc[0]
        print(f'FS - BL - Tuned Score: {self.cvm_blfs_tuned_1}, {self.cvs_blfs_tuned_1}')

        cvscores_blfs_tuned_2 = self.cross_validate(
            model = self.model_blfs_tuned,
            x = self.x_small, 
            y = self.y)
        
        self.cvm_blfs_tuned = cvscores_blfs_tuned_2.mean()
        self.cvs_blfs_tuned = cvscores_blfs_tuned_2.std()    
        print(f'Scoring Again: {self.cvm_blfs_tuned}, {self.cvs_blfs_tuned}')

        # 5. Feature Selection for tuned model
        print('Selecting Features for Tuned Model')
        feature_sel_tuned, self.feature_ranks_tuned = self.feature_selection(self.model_bl_tuned, strategy = 'rfecv')


        self.x_small_tuned = pd.DataFrame(feature_sel_tuned.transform(self.x), 
                                          columns=feature_sel_tuned.get_feature_names_out())
        
        cvscores_tuned_featsel = self.cross_validate(self.model_bl_tuned, self.x_small_tuned, self.y)

        self.cvm_tuned_fs = cvscores_tuned_featsel.mean()
        self.cvs_tuned_fs = cvscores_tuned_featsel.std()
        print(f'FS - Tuned - Tuned Score : {self.cvm_tuned_fs}, {self.cvs_tuned_fs}')
        
        # 4. Grid Search for best HyperParameters
        print('Tuning')
        self.tunedfs_tuning_res = self.tune_hyperparams(self.model_bl_tuned, self.x_small_tuned, self.y, n_iter)
        self.model_tunedfs_tuned = self.tunedfs_tuning_res.best_estimator_
        
        cvscores_tunedfs_tuned_1 = pd.DataFrame(self.tunedfs_tuning_res.cv_results_)

        self.cvm_tunedfs_tuned_1 = cvscores_tunedfs_tuned_1.loc[cvscores_tunedfs_tuned_1.rank_test_score == 1,'mean_test_score'].iloc[0]
        self.cvs_tunedfs_tuned_1 = cvscores_tunedfs_tuned_1.loc[cvscores_tunedfs_tuned_1.rank_test_score == 1,'std_test_score'].iloc[0]
        print(f'Tuned Score: {self.cvm_tunedfs_tuned_1}, {self.cvs_tunedfs_tuned_1}')

        cvscores_tunedfs_tuned = self.cross_validate(
            model = self.model_tunedfs_tuned,
            x = self.x_small_tuned, 
            y = self.y)
        
        self.cvm_tunedfs_tuned = cvscores_tunedfs_tuned.mean()
        self.cvs_tunedfs_tuned = cvscores_tunedfs_tuned.std()    
        print(f'Scoring Again: {self.cvm_tunedfs_tuned}, {self.cvs_tunedfs_tuned}')

        return 