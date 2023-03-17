#!/usr/bin/env python

# Vision: Automated Data Analysis
# Idea1:
# for classification problems, create basic model with the groups in the data.
# for churn-data: age_group, occupation, gender : Precision (churn_rate), Recall (whatever). This is the most basic decision tree.

# This contains functions to conduct EDA on any dataset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

"""
List of Functions:
Univariate Analysis
1. explore_numerical_variables
2. explore_categorical_variables
3. UVA_outlier

Bivariate Analysis
1. BVA_categorical_plot
2. BVA_cont_cat
3. explore_numerical_relationships
4. get_group_survival

Plotting [TODO]

Utils
1. log_x_1_transform

Statistical Tests
1. TwoSampZ
2. TwoSampT

Describe DataSet
1. Describe_variable
2. Descrtibe_dataset

"""

def describe_dataset(df, variables):
    """ Describe Basic Stats of mentioned variables in DF
    Input : 
        df = DataFrame for which basic stats are required
        variables = Variables for which basic stats are required
    Output :
        DataFrame with num_rows, perc_missing and min-max
    Remarks : 
        1. Assumes that dataTypes have already been set properly
        2. Currently works for Categorical, Numerical, [TODO] test for Datetime as well
    """

    # Initialize DF to store results
    df_describe = pd.DataFrame()
    
    # Loop through all variables
    for var in variables:
        # Get basic stats and append into one DF
        var_describe = describe_variable(df, var)
        df_describe = pd.concat([var_describe, df_describe])
    
    # Sort Values to highlight columns with high missing values
    df_describe = df_describe.sort_values('perc_missing', ascending= False)
    return df_describe
    
    

def describe_variable(df, var):
    """ Describe Basic Stats of one variable in DF
    Input : 
        df = DataFrame for which basic stats are required
        var = Variable for which basic stats are required
    Output :
        DataFrame with 1 row, with figures for num_rows, perc_missing and min-max for the column/variable
    Remarks : 
        1. Assumes that dataTypes have already been set properly
        2. Currently works for Categorical, Numerical, [TODO] test for Datetime as well
    """
    
    # Initialize Return DataFrame
    return_df = pd.DataFrame({
        'variable_name' : var,
        'dtype' : 'category' if df[var].dtype == 'category' else 'numeric',
        'unique_values' : len(df[var].unique()),
        'missing_values' : df[var].isna().sum(),
        'num_rows' : df[var].shape[0],
    }, index = range(0, 1))
    
    return_df['perc_missing'] = return_df.missing_values/return_df.num_rows
    return_df = return_df.sort_values('perc_missing', axis = 0)
    
    if df[var].dtype != 'category':
        return_df['min'] = np.min(df[var])
        return_df['mean'] = np.mean(df[var])
        return_df['median'] = np.median(df[var])
        return_df['max'] = np.max(df[var])
        
    return return_df


def get_descriptive_statistics(df, var):
    """
    Input : df, variable_name
    Output : Descriptive Statistics (min, max, mean, median etc.)
    """
    
    ret_dict = {
        'min' : df[var].min(),
        'max' : df[var].max(),
        'range' : df[var].max()-df[var].min(),
        'mean' : df[var].mean(),
        'median' : df[var].median(),
        'st_dev' : df[var].std(),
        'skew' : df[var].skew(),
        'kurt' : df[var].kurtosis()
    }
    
    ret_df = pd.DataFrame(ret_dict, index = [0])
    return ret_df


def plot_descriptive_statistics(df, var):
    desc_stats = get_descriptive_statistics(df, var)
    # get Values from Dict
    mini = desc_stats['min'][0]
    maxi = desc_stats['max'][0]
    mean = desc_stats['mean'][0]
    median = desc_stats['median'][0]
    st_dev = desc_stats['st_dev'][0]
    kurt = desc_stats['kurt'][0]
    skew = desc_stats['skew'][0]
    ran = desc_stats['range'][0]
    
    a = plt.figure()
    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev
    
    sns.kdeplot(df[var], fill=True)
        
    sns.lineplot(x = points, y=[0,0], color = 'black', label = "std_dev")
    sns.scatterplot(x = [mini,maxi], y = [0,0], color = 'orange', marker = "|", s = 250, alpha = 1)
    sns.scatterplot(x = [mini,maxi], y = [0,0], color = 'orange', label = "min/max")
#    sns.lineplot(x = [mini,mini], y = [-1,1], color = 'orange')
#    sns.lineplot(x = [maxi,maxi], y = [-1,1], color = 'orange', label = "min/max")
    
    sns.scatterplot(x = [mean],  y = [0], color = 'red', label = "mean")
    sns.scatterplot(x = [mean],  y = [0], color = 'red', marker = "|", s = 250, alpha = 1)
#    sns.lineplot(x = [mean, mean],  y = [-1,1], color = 'red', label = "mean")
    
    
    sns.scatterplot(x = [median],  y = [0], color = 'blue', label = "median")
    sns.scatterplot(x = [median],  y = [0], color = 'blue', marker = "|", s = 250, alpha = 1)
    plt.xlabel('{}'.format(var), fontsize = 20)
    plt.ylabel('density')
    plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format(
        (round(points[0],2),round(points[1],2)),
                                                                                                   round(kurt,2),
                                                                                                   round(skew,2),
                                                                                                   (round(mini,2),round(maxi,2),round(ran,2)),
                                                                                                   round(mean,2),
                                                                                                   round(median,2)))
    
    return
    

def explore_numerical_variables(df, var_group, plot = True):

    size = len(var_group)
    plt.figure(figsize = (7*size,2), dpi = 100)
    for var in var_group:
        plot_descriptive_statistics(df, var)
        
        
    
def explore_categorical_variables(data, var_group):
    '''
    Univariate_Analysis_categorical
    takes a group of variables (category) and plot/print all the value_counts and barplot.
    '''
    # setting figure_size
    size = len(var_group)
    plt.figure(figsize = (7*size,5), dpi = 100)

    # for every variable
    for j,i in enumerate(var_group):
        norm_count = data[i].value_counts(normalize = True)
        n_uni = data[i].nunique()

        #Plotting the variable with every information
        plt.subplot(1,size,j+1)
        sns.barplot(x = norm_count, y = norm_count.index , order = norm_count.index)
        plt.xlabel('fraction/percent', fontsize = 20)
        plt.ylabel('{}'.format(i), fontsize = 20)
        plt.title('n_uniques = {} \n value counts \n {};'.format(n_uni,norm_count))
        
        
def UVA_outlier(data, var_group, include_outlier = True):
  '''
  Univariate_Analysis_outlier:
  takes a group of variables (INTEGER and FLOAT) and plot/print boplot and descriptives\n
  Runs a loop: calculate all the descriptives of i(th) variable and plot/print it \n\n

  data : dataframe from which to plot from\n
  var_group : {list} type Group of Continuous variables\n
  include_outlier : {bool} whether to include outliers or not, default = True\n
  '''

  size = len(var_group)
  plt.figure(figsize = (7*size,4), dpi = 100)
  
  #looping for each variable
  for j,i in enumerate(var_group):
    
    # calculating descriptives of variable
    quant25 = data[i].quantile(0.25)
    quant75 = data[i].quantile(0.75)
    IQR = quant75 - quant25
    med = data[i].median()
    whis_low = med-(1.5*IQR)
    whis_high = med+(1.5*IQR)

    # Calculating Number of Outliers
    outlier_high = len(data[i][data[i]>whis_high])
    outlier_low = len(data[i][data[i]<whis_low])

    if include_outlier == True:
      print(include_outlier)
      #Plotting the variable with every information
      plt.subplot(1,size,j+1)
      sns.boxplot(data[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('With Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))
      
    else:
      # replacing outliers with max/min whisker
      data2 = data[var_group][:]
      data2[i][data2[i]>whis_high] = whis_high+1
      data2[i][data2[i]<whis_low] = whis_low-1
      
      # plotting without outliers
      plt.subplot(1,size,j+1)
      sns.boxplot(data2[i], orient="v")
      plt.ylabel('{}'.format(i))
      plt.title('Without Outliers\nIQR = {}; Median = {} \n 2nd,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))
        
        
def BVA_categorical_plot(data, tar, cat):
  '''
  take data and two categorical variables,
  calculates the chi2 significance between the two variables 
  and prints the result with countplot & CrossTab
  '''
  #isolating the variables
  data = data[[cat,tar]][:]

  #forming a crosstab
  table = pd.crosstab(data[tar],data[cat],)
  f_obs = np.array([table.iloc[0][:].values,
                    table.iloc[1][:].values])

  #performing chi2 test
  from scipy.stats import chi2_contingency
  chi, p, dof, expected = chi2_contingency(f_obs)
  
  #checking whether results are significant
  if p<0.05:
    sig = True
  else:
    sig = False

  #plotting grouped plot
  sns.countplot(x=cat, hue=tar, data=data)
  plt.title("p-value = {}\n difference significant? = {}\n".format(round(p,8),sig))

  #plotting percent stacked bar plot
  #sns.catplot(ax, kind='stacked')
  ax1 = data.groupby(cat)[tar].value_counts(normalize=True).unstack()
  ax1.plot(kind='bar', stacked='True',title=str(ax1))
  int_level = data[cat].value_counts()
    
    
def BVA_cont_cat(data, cont, cat, category):
  #creating 2 samples
    x1 = data[cont][data[cat]==category][:]
    x2 = data[cont][~(data[cat]==category)][:]
  
    #calculating descriptives
    n1, n2 = x1.shape[0], x2.shape[0]
    m1, m2 = x1.mean(), x2.mean()
    std1, std2 = x1.std(), x2.std()

    #calculating p-values
    t_p_val = TwoSampT(m1, m2, std1, std2, n1, n2)
    z_p_val = TwoSampZ(m1, m2, std1, std2, n1, n2)

    #table
    table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc = np.mean)

    #plotting
    plt.figure(figsize = (15,6), dpi=140)

    #barplot
    plt.subplot(1,2,1)
    sns.barplot(x=[str(category),'not {}'.format(category)], y=[m1, m2])
    plt.ylabel('mean {}'.format(cont))
    plt.xlabel(cat)
    plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                z_p_val,
                                                                table))

    # boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=cat, y=cont, data=data)
    plt.title('categorical boxplot')
    
    
def explore_numerical_relationship(df, var_1, var_2, log_transform = False):
    correlation = df[[var_1, var_2]].corr()[var_1][1].round(2)
    
    if log_transform:
        df = log_x1_transform(df, [var_1, var_2])
    # make scatter plot
    sns.scatterplot(data = df, x = var_1, y = var_2)
    plot_title = "Corr = {}".format(correlation)
    plt.title(plot_title)
    return correlation


def get_group_survival(df, grp, target):
    
    # Init
    df_base = pd.DataFrame()
    
    # Overall Numbers
    total_in_boat =  df[target].sum()

    # Grouped Numbers
    grouped_totals = df.groupby(grp).count()[target].reset_index()
    grouped_survived = df.groupby(grp).sum()[target].reset_index()

    grouped_totals['Total'] = grouped_totals[target]
    grouped_totals = grouped_totals.drop(target, axis = 1)

    df_base = grouped_totals.merge(grouped_survived)

    df_base['survival_rate'] = df_base.Survived/df_base.Total
    df_base['total_in_boat'] = df.count()['Survived']
    df_base['total_survived_in_boat'] = df.Survived.sum()

    df_base['survival_rate_in_boat'] = df_base.total_survived_in_boat/df_base.total_in_boat
    df_base['index_group_member'] = df_base.survival_rate/df_base.survival_rate_in_boat
    df_base['surface_member'] = df_base.index_group_member > 1.5
    
    df_base['survival_contribution'] = df_base.Survived/df_base.total_survived_in_boat
    df_base['chance_survival_rate'] = 1.0/df_base.shape[0]
    df_base['surface_member_2'] = df_base.survival_contribution > df_base['chance_survival_rate']
    df_base['f1'] = 2*df_base.survival_rate*df_base.survival_contribution/(df_base.survival_rate+df_base.survival_contribution)
    # TODO Cumulative NUmbers, if needed
#    df_base = df_base.sort_values(['survival_contribution'], ascending=False)
    df_base = df_base.sort_values(['survival_rate'], ascending=False)
    df_base['cum_survival_rate'] = df_base.Survived.cumsum()/df_base.Total.cumsum()
    df_base['cum_coverage'] = df_base.Survived.cumsum()/df_base.total_survived_in_boat
    df_base['cum_f1'] = df_base.cum_coverage*df_base.cum_survival_rate/(df_base.cum_coverage+df_base.cum_survival_rate)
    
    plt.figure()
    sns.lineplot(x = range(0,len(df_base)), y = df_base.cum_coverage, label = 'Coverage')
    sns.lineplot(x = range(0,len(df_base)), y = df_base.cum_survival_rate, label = 'Survival Rate')

    diff = list(abs(df_base['cum_survival_rate']- df_base['cum_coverage']))
    min_diff = min(diff)
    diff.index(min_diff)
    
    opt_sr = max(df_base['cum_survival_rate'][(diff.index(min_diff)-1):(diff.index(min_diff)+1)])
    opt_cov = max(df_base['cum_coverage'][(diff.index(min_diff)-1):(diff.index(min_diff)+1)])
    opt_f1 = max(df_base['cum_f1'][(diff.index(min_diff)-1):(diff.index(min_diff)+1)])
    sns.lineplot(y = opt_cov, x = range(0,len(df_base)))
    
    plot_title = '+'.join(grp)+":"+str(round(opt_sr,2))+','+str(round(opt_cov,2)) +','+str(round(opt_f1,2))
    plt.title(plot_title)
#    plt.line(df_base.cum_survival_rate)
    return df_base, [opt_sr, opt_cov, opt_f1]
    
def TwoSampZ(X1, X2, sigma1, sigma2, N1, N2):
  '''
  takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sampled Z-Test
  '''
  from numpy import sqrt, abs, round
  from scipy.stats import norm
  ovr_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
  z = (X1 - X2)/ovr_sigma
  pval = 2*(1 - norm.cdf(abs(z)))
  return pval


def TwoSampT(X1, X2, sd1, sd2, n1, n2):
  '''
  takes mean, standard deviation, and number of observations and returns p-value calculated for 2-sample T-Test
  '''
  from numpy import sqrt, abs, round
  from scipy.stats import t as t_dist
  ovr_sd = sqrt(sd1**2/n1 + sd2**2/n2)
  t = (X1 - X2)/ovr_sd
  df = n1+n2-2
  pval = 2*(1 - t_dist.cdf(abs(t),df))
  return pval
    

def log_x1_transform(df, var_group, make_new_column = False):

    for var in var_group:
        if make_new_column:
            new_col_name = var + '_log'
        else: new_col_name = var

        mini=1

        if df[var].min()<0:
            mini =  abs(df[var].min()) + 1
            print(mini)

        df[new_col_name] = [i+mini for i in df[var]]
        df[new_col_name] = df[new_col_name].map(lambda x : np.log(x))
    
    return df


def set_dtype(df, cols, dtype):
    for col in cols:
        df[col] = df[col].astype(dtype)
    return df